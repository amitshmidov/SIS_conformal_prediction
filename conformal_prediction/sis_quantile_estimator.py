import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, tensor
from numpy import random
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import pickle
from torch.nn import functional as F
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import Dataset, Sampler, DataLoader, random_split
from torch.optim import Adam, SGD

from collections import defaultdict

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

import pytorch_lightning as pl
from operator import itemgetter
from argparse import Namespace
from mlp import MLP
from typing import List, Optional


class SISQuantileEstimator(pl.LightningModule):
    def __init__(self, quantiles, featurizer, num_features, num_labels, classifier_dims, lr=1e-3):
        super(SISQuantileEstimator, self).__init__()
        self.register_buffer('quantiles', tensor(quantiles, requires_grad=False))
        self.featurizer: nn.Module = featurizer
        self.num_labels = num_labels
        classifier_dims.append(len(quantiles))
        acts = [*['relu'] * (len(classifier_dims) - 1), 'none']
        self.classifier = MLP(in_dim=num_features + num_labels, dims=classifier_dims, nonlins=acts)
        assert self.classifier.out_dim == len(quantiles)
        self.learning_rate = lr
        # print(f"init----device: {self.device} gpu: {show_gpu(str(self.device))}")

    def forward(self, x, y) -> torch.Tensor:
        # print(f'x {x} y {y}')
        # self.featurizer.eval()
        # with torch.no_grad():
        features = self.featurizer.extract_features(x)
        label_vec = F.one_hot(y, self.num_labels)
        vec = torch.cat((features, label_vec), dim=1)
        return self.classifier(vec)

    def get_sis_quantile(self, x, y, quantile):
        quan_idx = (self.quantiles == quantile).nonzero()[0]
        if quan_idx.numel() != 1:
            return None
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        preds: Tensor = self(x, y)
        return preds.index_select(dim=1, index=quan_idx)

    def loss(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        # print(f'preds {preds.shape} target {target.shape} quan {self.quantiles}')
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
            # print(f'preds {i} {preds[:, i].shape} target {target.shape} quan {self.quantiles})')
            # losses.append(F.mse_loss(preds[:, i], target).reshape(1, 1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        # print(f'losses {losses} loss {loss}')
        return loss

    def training_step(self, batch_data, batch_idx):
        return self.step(batch_data, 'train')

    def step(self, batch_data, stage):
        x = batch_data['image']
        y = batch_data['label']
        sis = batch_data['sis']
        # print(f"sis: {sis.shape}")
        preds = self(x, y)
        # print(f'sis {sis} stage {stage} base model eval {self.featurizer.training}')
        self.logger.experiment[f"/{stage}/std"].log(torch.std(preds).item())
        loss = self.loss(preds, sis)

        # print(f"standard deviation: {std}")
        # self.logger.experiment[f"{self.run_name}/train/acc"].log(batch_accuracy.item())
        # # self.logger.experiment[f"{self.run_name}/train/std"].log(std)
        self.logger.experiment[f"/{stage}/loss"].log(loss)
        # self.logger.experiment[f"train/{self.run_name}/task{i}_std"].log(torch.std(out[:, 1].detach()).item())
        self.log(f'{stage}_loss', loss)
        return {'loss': loss, 'preds': preds.detach(), 'sis': sis}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end(outputs, 'train')

    def epoch_end(self, outputs: EPOCH_OUTPUT, stage) -> None:
        epoch_preds = torch.cat([output['preds'] for output in outputs], dim=0)

        epoch_sis = torch.cat([output['sis'] for output in outputs], dim=0).unsqueeze(1)
        assert epoch_preds.size(0) == epoch_sis.size(0)
        # print(f"preds {epoch_preds.shape} sis {epoch_sis.shape}")
        quantiles_cdf = (epoch_sis < epoch_preds).float().mean(dim=0)
        epoch_error = F.mse_loss(quantiles_cdf, self.quantiles, reduction='sum')
        print(f'epoch_preds {epoch_preds.shape} error {epoch_error} stage {stage} cdf {quantiles_cdf.detach().tolist()}')
        self.logger.experiment[f"/{stage}/epoch_error"].log(epoch_error.item())

    # def test_step(self, batch_data, batch_idx):
    #     return self.step(batch_data, 'test')

    # def test_epoch_end(self, outputs: list) -> None:
    #     self.epoch_end(outputs, 'test')

    def validation_step(self, batch_data, batch_idx):
        return self.step(batch_data, 'val')

    def validation_epoch_end(self, outputs: list) -> None:
        self.epoch_end(outputs, 'val')

    def configure_optimizers(self):
        print(f'CHOSEN LR: {self.learning_rate}')
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def total_steps(self):
        return (self.hparams.train_samples_to_load // self.hparams.train_samples_per_dom_batch) * self.hparams.epochs


class SISDataModule(pl.LightningDataModule):
    def __init__(self, dataset_type, train_args, test_args, hparams: Namespace):
        super().__init__()
        self.train_batch_size = hparams.train_batch_size
        self.dataset_type = dataset_type
        self.train_args = train_args
        self.test_args = test_args
        self.hparams.update(vars(hparams))
        self.num_samples = 0
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        print(f"Setup STAGE = {stage}")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            training_set: Dataset = self.dataset_type(**self.train_args)
            val_len = int(self.hparams.val_split_ratio * len(training_set))
            self.train_set, self.val_set = random_split(training_set, [len(training_set) - val_len, val_len])
            print(f'train len {len(self.train_set)} val {len(self.val_set)}')
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = self.dataset_type(**self.test_args)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        if self.val_set is not None:
            return DataLoader(self.val_set, batch_size=self.train_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.test_batch_size, shuffle=False, num_workers=4)

class SisDataset(Dataset):
    def __init__(self, sis_path, images_path, load_n_samples=None, max_labels=3):
        dict_keys = ['masks_sizes', 'corrects', 'pred_labels']
        with open(sis_path, 'rb') as file:
            data_dict: dict = pickle.load(file)
        with open(images_path, 'rb') as file:
            self.images = tensor(pickle.load(file))
        images_idx = np.arange(self.images.shape[0]).reshape(-1, 1)
        data_array = np.stack([data_dict[key].T for key in dict_keys], axis=2)
        n_labels = data_array.shape[1]
        self.images_indices = np.expand_dims(np.repeat(images_idx, n_labels, axis=1), axis=2)
        data_array = np.concatenate([data_array, self.images_indices], axis=2)

        data_array = data_array[:, :max_labels, :]  # filters labels
        data_array = data_array[data_array[:, :, 0] > 0, :].astype(int)  # filters sis=0
        data_array = data_array[:load_n_samples]

        self.sis = tensor(data_array[:, 0]).float()
        self.correct = tensor(data_array[:, 1])
        self.labels = tensor(data_array[:, 2])
        self.images_indices = data_array[:, 3]
        print(f'LEN {len(self)}')

    def __getitem__(self, item):
        return {'image': self.images[self.images_indices[item]],
                'label': self.labels[item],
                'sis': self.sis[item],
                'correct': self.correct[item]}

    def __len__(self):
        return self.sis.shape[0]

