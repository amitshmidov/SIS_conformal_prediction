import numpy as np
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, tensor
from numpy import random
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from torch.nn import functional as F
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.optim import Adam, SGD

from collections import defaultdict

from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

import pytorch_lightning as pl
from operator import itemgetter
from argparse import Namespace
from mlp import MLP
from typing import List, Optional


class SISQuantileEstimator(pl.LightningModule):
    def __init__(self, quantiles, featurizer, num_features, num_labels, classifier_dims):
        super(SISQuantileEstimator, self).__init__()
        self.quantiles: Tensor = tensor(quantiles, requires_grad=False)
        self.featurizer = featurizer
        self.num_labels = num_labels
        acts = [*['relu'] * (len(classifier_dims) - 1), 'none']
        self.classifier = MLP(in_dim=num_features + num_labels, dims=classifier_dims, nonlins=acts)
        assert self.classifier.out_dim == len(quantiles)

        # print(f"init----device: {self.device} gpu: {show_gpu(str(self.device))}")

    def forward(self, x, y) -> torch.Tensor:
        features = self.featurizer(x)
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
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

    def training_step(self, batch_data):
        return self.step(batch_data, 'train')

    def step(self, batch_data, stage):
        x = batch_data['x']
        y = batch_data['y']
        sis = batch_data['sis']
        # print(f"labels: {labels.shape} inputs: {input_ids.shape}")
        preds = self(x, y)
        loss = self.loss(preds, sis)

        # print(f"standard deviation: {std}")
        # self.logger.experiment[f"{self.run_name}/train/acc"].log(batch_accuracy.item())
        # # self.logger.experiment[f"{self.run_name}/train/std"].log(std)
        # self.logger.experiment[f"{self.run_name}/train/loss"].log(loss)
        # self.logger.experiment[f"train/{self.run_name}/task{i}_std"].log(torch.std(out[:, 1].detach()).item())
        self.log(f'{stage}_loss', loss)
        return {'loss': loss, 'preds': preds, 'sis': sis}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end(outputs, 'train')

    def epoch_end(self, outputs: EPOCH_OUTPUT, stage) -> None:
        epoch_preds = torch.cat([output['preds'] for output in outputs], dim=0)
        epoch_sis = torch.cat([output['preds'] for output in outputs], dim=0)
        assert epoch_preds.size(0) == epoch_sis.size(0)
        quantiles_cdf = (epoch_sis < epoch_preds).mean(dim=0)
        epoch_error = F.mse_loss(quantiles_cdf, self.quantiles, reduction='sum')
        self.logger.experiment[f"{self.run_name}/{stage}/epoch_error"].log(epoch_error.item())

    def test_step(self, batch_data):
        return self.step(batch_data, 'test')

    def test_epoch_end(self, outputs: list) -> None:
        self.epoch_end(outputs, 'test')

    def validation_step(self, batch_data):
        return self.step(batch_data, 'val')

    def validation_epoch_end(self, outputs: list) -> None:
        self.test_epoch_end(outputs)

    def configure_optimizers(self):
        print(f'CHOSEN LR: {self.learning_rate}')
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def total_steps(self):
        return (self.hparams.train_samples_to_load // self.hparams.train_samples_per_dom_batch) * self.hparams.epochs

class SISDataModule(pl.LightningDataModule):
    def __init__(self, dataset_type, train_args, test_args, hparams: Namespace):
        super().__init__()
        self.train_batch_size = hparams.tune_batch_size if test_args else hparams.train_batch_size
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

        print(f"STAGE = {stage}")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = self.dataset_type(**self.train_args)

            if self.test_args:
                self.val_set = self.dataset_type(**self.test_args)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_set = self.dataset_type(**self.test_args)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        if not self.test_args:
            return None
        return DataLoader(self.val_set, batch_size=self.hparams.test_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.hparams.test_batch_size, shuffle=False, num_workers=4)

def train_estimator():
    featurizer: nn.Module
    data_module = SISDataModule()
    model = SISQuantileEstimator([0.05, 0.9], featurizer, num_features=50, num_labels=10, classifier_dims=[100, 2])
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(gpus=hparams.gpus, auto_lr_find=False, max_epochs=hparams.epochs,
                         replace_sampler_ddp=False, logger=logger, auto_scale_batch_size=False,
                         enable_checkpointing=False, callbacks=[early_stop_callback])
    trainer.fit(model, data_module)


if __name__ == '__main__':
    train_estimator()