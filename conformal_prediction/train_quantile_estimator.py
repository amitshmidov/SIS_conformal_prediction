import argparse
import functools

from pytorch_lightning.loggers import NeptuneLogger

import inference_util
from config import DEVICE
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, tensor
from numpy import random
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from conformal_prediction.sis_quantile_estimator import SISDataModule, SISQuantileEstimator, SisDataset
import pytorch_lightning as pl

from model.resnet_cifar import ResNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, required=True, help='Path to saved base model')
    parser.add_argument('--save_model_to', type=str, required=False, help='Path to saved model directory')
    parser.add_argument('--images_path', type=str, required=True, help='Path to images_data')
    parser.add_argument('--sis_path', type=str, required=True,
                        help='Path sis data')
    # parser.add_argument('--dataset', required=True, choices=DATASET_OPTIONS,
    #                     help='Dataset name')
    parser.add_argument('--data_path', required=False, help='Dataset path')
    parser.add_argument('--sis_threshold', type=float, default=0.0,
                        help='SIS threshold (default: 0 to run backward selection on all images)')
    parser.add_argument('--accelerator', type=str, required=False, choices=['cpu', 'gpu', 'mps'], default='gpu',
                        help='device type')
    parser.add_argument('--device', type=int, required=False, default=0, help='device idx')
    parser.add_argument('--max_epochs', type=int, required=False, default=100, help='epochs')
    parser.add_argument('-q', '--quantiles', type=str, required=False, default='0.5,0.9', help='quantiles to estimate')
    parser.add_argument('-d', '--mlp_dims', type=str, required=False, default='1000,1000,500,500,200', help='classifier dims')
    parser.add_argument('-s', '--samples_limit', type=int, required=False, help='training samples')
    # parser.add_argument('--mps', action='store_true', default=False,
    #                     help='use mps')

    args = parser.parse_args()
    print(args)
    quantiles = [float(item) for item in args.quantiles.split(',')]
    mlp_dims = [int(item) for item in args.mlp_dims.split(',')]
    logger = NeptuneLogger(
        project="nivkook9/ConformalSIS",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0MjlmZDIyNy0yNzYzLTRmNjYtODkxZC0zOGYyMjAxNWE4NzcifQ==",
        # format "<WORKSPACE/PROJECT>"
        log_model_checkpoints=False
    )
    logger.log_hyperparams(args)
    train_args = {'sis_path': args.sis_path, 'images_path': args.images_path, 'load_n_samples': args.samples_limit}
    data_module = SISDataModule(SisDataset, train_args, {}, argparse.Namespace(train_batch_size=5,
                                                                               val_split_ratio=0.2))
    # sis_out_dir = "../saved_models/resnet18_rep1"
    base_model: ResNet = inference_util.load_saved_model(args.base_model_path + "/")
    # featurizer = functools.partial(base_model.extract_features)
    model = SISQuantileEstimator(quantiles, base_model, num_features=512, num_labels=10, classifier_dims=mlp_dims)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.device], auto_lr_find=True, max_epochs=args.max_epochs,
                         replace_sampler_ddp=False, logger=logger, auto_scale_batch_size=False,
                         enable_checkpointing=False, callbacks=[], overfit_batches=0, check_val_every_n_epoch=args.max_epochs)
    trainer.tune(model, data_module)
    print(f'{model.learning_rate} = LR')
    trainer.fit(model, data_module)
    if args.save_model_to is not None:
        trainer.save_checkpoint(args.saved_model_dir)
