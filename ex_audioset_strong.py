import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
import wandb
import transformers
import random

from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.m2d.M2D_wrapper import M2DWrapper
from models.asit.ASIT_wrapper import ASiTWrapper
from models.prediction_wrapper import PredictionsWrapper
from helpers.augment import frame_shift, time_mask, mixup, filter_augmentation, mixstyle, RandomResizeCrop

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from helpers.utils import worker_init_fn

from data_util.audioset_strong import get_training_dataset, get_validation_dataset
from data_util.audioset_strong import get_temporal_count_balanced_sample_weights, get_uniform_sample_weights, \
    get_weighted_sampler
from data_util import audioset_classes


class PLModule(pl.LightningModule):
    def __init__(self, config, encoder):
        super().__init__()
        self.config = config
        self.encoder = encoder

        if config.pretrained == "scratch":
            checkpoint = None
        elif config.pretrained == "ssl":
            checkpoint = "ssl"
        elif config.pretrained == "weak":
            checkpoint = "weak"
        elif config.pretrained == "strong":
            checkpoint = "strong_1"
        else:
            raise ValueError(f"Unknown pretrained checkpoint: {config.pretrained}")

        # load transformer model
        if config.model_name == "BEATs":
            beats = BEATsWrapper()
            model = PredictionsWrapper(beats, checkpoint=f"BEATs_{checkpoint}" if checkpoint else None)
        elif config.model_name == "ATST-F":
            atst = ATSTWrapper()
            model = PredictionsWrapper(atst, checkpoint=f"ATST-F_{checkpoint}" if checkpoint else None)
        elif config.model_name == "fpasst":
            fpasst = FPaSSTWrapper()
            model = PredictionsWrapper(fpasst, checkpoint=f"fpasst_{checkpoint}" if checkpoint else None)
        elif config.model_name == "M2D":
            m2d = M2DWrapper()
            model = PredictionsWrapper(m2d, checkpoint=f"M2D_{checkpoint}" if checkpoint else None)
        elif config.model_name == "ASIT":
            asit = ASiTWrapper()
            model = PredictionsWrapper(asit, checkpoint=f"ASIT_{checkpoint}" if checkpoint else None)
        else:
            raise NotImplementedError(f"Model {config.model_name} not (yet) implemented")

        self.model = model

        # prepare ingredients for knowledge distillation
        assert 0 <= config.distillation_loss_weight <= 1, "Lambda for Knowledge Distillation must be between 0 and 1."
        self.strong_loss = nn.BCEWithLogitsLoss()

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, batch):
        x = batch["audio"]
        mel = self.model.mel_forward(x)
        y_strong, _ = self.model(mel)
        return y_strong

    def get_optimizer(
            self, lr, adamw=False, weight_decay=0.01, betas=(0.9, 0.999)
    ):
        # we split the parameters into two groups, one for the pretrained model and one for the downstream model
        # we also split each of them into <=1 dimensional and >=2 dimensional parameters, so we can only
        # apply weight decay to the >=2 dimensional parameters, thus excluding biases and batch norms, an idea from NanoGPT
        params_leq1D = []
        params_geq2D = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.ndimension() >= 2:
                    params_geq2D.append(param)
                else:
                    params_leq1D.append(param)

        param_groups = [
            {'params': params_leq1D, 'lr': lr},
            {'params': params_geq2D, 'lr': lr, 'weight_decay': weight_decay},
        ]

        if weight_decay > 0:
            assert adamw
        assert len(param_groups) > 0
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return torch.optim.Adam(param_groups, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=20000,
            lr_end=2e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = self.get_optimizer(self.config.max_lr, adamw=self.config.adamw,
                                       weight_decay=self.config.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(optimizer, num_training_steps,
                                          schedule_mode=self.config.schedule_mode,
                                          lr_end=self.config.lr_end)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """

        x = train_batch["audio"]
        labels = train_batch['strong']
        pseudo_labels = train_batch['pseudo_strong']

        mel = self.model.mel_forward(x)

        # time rolling
        if self.config.frame_shift_range > 0:
            mel, labels, pseudo_labels = frame_shift(
                mel,
                labels,
                pseudo_labels=pseudo_labels,
                net_pooling=self.encoder.net_pooling,
                shift_range=self.config.frame_shift_range
            )

        # mixup
        if self.config.mixup_p > random.random():
            mel, labels, pseudo_labels = mixup(
                mel,
                targets=labels,
                pseudo_strong=pseudo_labels
            )

        # mixstyle
        if self.config.mixstyle_p > random.random():
            mel = mixstyle(
                mel
            )

        # time masking
        if self.config.max_time_mask_size > 0:
            mel, labels, pseudo_labels = time_mask(
                mel,
                labels,
                pseudo_labels=pseudo_labels,
                net_pooling=self.encoder.net_pooling,
                max_mask_ratio=self.config.max_time_mask_size
            )

        # frequency masking
        if self.config.filter_augment_p > random.random():
            mel, _ = filter_augmentation(
                mel
            )

        # frequency warping
        if self.config.freq_warp_p > random.random():
            mel = mel.squeeze(1)
            mel = self.freq_warp(mel)
            mel = mel.unsqueeze(1)

        # forward through network; use strong head
        y_hat_strong, _ = self.model(mel)

        strong_supervised_loss = self.strong_loss(y_hat_strong, labels)

        if self.config.distillation_loss_weight > 0:
            strong_distillation_loss = self.strong_loss(y_hat_strong, pseudo_labels)
        else:
            strong_distillation_loss = torch.tensor(0., device=y_hat_strong.device, dtype=y_hat_strong.dtype)

        loss = self.config.distillation_loss_weight * strong_distillation_loss \
               + (1 - self.config.distillation_loss_weight) * strong_supervised_loss

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        self.log("train/strong_supervised_loss", strong_supervised_loss.detach().cpu())
        self.log("train/strong_distillation_loss", strong_distillation_loss.detach().cpu())

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, _, y = val_batch
        x = self.mel_forward(x)
        y_hat, _ = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        preds = torch.sigmoid(y_hat)
        results = {'val_loss': loss, "preds": preds, "targets": y}
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs])
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0)

        all_preds = self.all_gather(preds).reshape(-1, preds.shape[-1]).cpu().float().numpy()
        all_targets = self.all_gather(targets).reshape(-1, targets.shape[-1]).cpu().float().numpy()
        all_loss = self.all_gather(loss).reshape(-1, )

        try:
            average_precision = metrics.average_precision_score(
                all_targets, all_preds, average=None)
        except ValueError:
            average_precision = np.array([np.nan] * 527)
        try:
            roc = metrics.roc_auc_score(all_targets, all_preds, average=None)
        except ValueError:
            roc = np.array([np.nan] * 527)
        logs = {'val/loss': torch.as_tensor(all_loss).mean().cuda(),
                'val/ap': torch.as_tensor(average_precision).mean().cuda(),
                'val/roc': torch.as_tensor(roc).mean().cuda()
                }
        self.log_dict(logs, sync_dist=False)
        self.validation_step_outputs.clear()


def train(config):
    # Train Models on temporally-strong portion of AudioSet.

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="PTSED",
        notes="Pre-Training Transformers for Sound Event Detection on AudioSet Strong.",
        tags=["AudioSet Strong", "Sound Event Detection", "Pseudo Labels", "Knowledge Disitillation"],
        config=config,
        name=config.experiment_name
    )

    # encoder manages encoding and decoding of model predictions
    encoder = ManyHotEncoder(audioset_classes.as_strong_train_classes)

    train_set = get_training_dataset(encoder, wavmix_p=config.wavmix_p)  # TODO: pseudo label location
    eval_set = get_validation_dataset(encoder)

    if config.use_balanced_sampler:
        sample_weights = get_temporal_count_balanced_sample_weights(train_set, save_folder="resources")
    else:
        sample_weights = get_uniform_sample_weights(train_set)

    train_sampler = get_weighted_sampler(sample_weights)

    # train dataloader
    train_dl = DataLoader(dataset=train_set,
                          sampler=train_sampler,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=False)

    # eval dataloader
    eval_dl = DataLoader(dataset=eval_set,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config, encoder)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0)

    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, eval_dl)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet_Strong")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--precision', type=int, default=16)

    # model
    parser.add_argument('--model_name', type=str, default="ATST-F")  # used also for training
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    parser.add_argument('--pretrained', type=str, default="weak")

    # training
    parser.add_argument('--n_epochs', type=int, default=120)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--distillation_loss_weight', type=float, default=0.9)
    parser.add_argument('--epoch_len', type=int, default=100000)

    # augmentation
    parser.add_argument('--wavmix_p', type=float, default=0.8)
    parser.add_argument('--freq_warp_p', type=float, default=0.8)
    parser.add_argument('--filter_augment_p', type=float, default=0.8)
    parser.add_argument('--frame_shift_range', type=float, default=0.125)  # in seconds
    parser.add_argument('--mixup_p', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.3)
    parser.add_argument('--max_time_mask_size', type=float, default=0.0)

    # optimizer
    parser.add_argument('--adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # lr schedule
    parser.add_argument('--schedule_mode', type=str, default="cos")
    parser.add_argument('--max_lr', type=float, default=0.003)
    parser.add_argument('--lr_end', type=float, default=2e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)

    # knowledge distillation
    parser.add_argument('--pseudo_label_file', type=str,
                        default=os.path.join("resources", "pseudo_labels.hdf5"))

    args = parser.parse_args()
    train(args)
