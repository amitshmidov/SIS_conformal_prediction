import argparse
from config import DEVICE
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor, tensor
from numpy import random
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from conformal_prediction.sis_quantile_estimator import SISDataModule, SISQuantileEstimator
import pytorch_lightning as pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir', type=str, required=False,
                        help='Path to saved model directory')
    # parser.add_argument('--dataset', required=True, choices=DATASET_OPTIONS,
    #                     help='Dataset name')
    parser.add_argument('--data_path', required=False, help='Dataset path')
    parser.add_argument('--sis_threshold', type=float, default=0.0,
                        help='SIS threshold (default: 0 to run backward selection on all images)')
    parser.add_argument('--accelerator', type=str, required=False, choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help='device type')
    parser.add_argument('--device', type=int, required=False, default=0, help='device idx')
    parser.add_argument('--max_epochs', type=int, required=False, default=5, help='epochs')
    # parser.add_argument('--mps', action='store_true', default=False,
    #                     help='use mps')

    args = parser.parse_args()
    print(args)


    featurizer: nn.Module
    data_module = SISDataModule()
    model = SISQuantileEstimator([0.05, 0.9], featurizer, num_features=50, num_labels=10, classifier_dims=[100, 2])
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    trainer = pl.Trainer(accelerator=args.accelerator, devices=[args.device], auto_lr_find=False, max_epochs=args.max_epochs,
                         replace_sampler_ddp=False, logger=logger, auto_scale_batch_size=False,
                         enable_checkpointing=False, callbacks=[early_stop_callback])
    trainer.fit(model, data_module)
    if args.saved_model_dir is not None:
        trainer.save_checkpoint(args.saved_model_dir)
