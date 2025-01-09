import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import wandb
import transformers
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import sed_scores_eval

from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.m2d.M2D_wrapper import M2DWrapper
from models.asit.ASIT_wrapper import ASiTWrapper
from models.prediction_wrapper import PredictionsWrapper
from helpers.augment import frame_shift, time_mask, mixup, filter_augmentation, mixstyle, RandomResizeCrop
from helpers.utils import worker_init_fn
from data_util.audioset_strong import get_training_dataset, get_eval_dataset
from data_util.audioset_strong import get_temporal_count_balanced_sample_weights, get_uniform_sample_weights, \
    get_weighted_sampler
from data_util.audioset_classes import as_strong_train_classes, as_strong_eval_classes


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
            model = PredictionsWrapper(beats, checkpoint=f"BEATs_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name == "ATST-F":
            atst = ATSTWrapper()
            model = PredictionsWrapper(atst, checkpoint=f"ATST-F_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name == "fpasst":
            fpasst = FPaSSTWrapper()
            model = PredictionsWrapper(fpasst, checkpoint=f"fpasst_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name == "M2D":
            m2d = M2DWrapper()
            model = PredictionsWrapper(m2d, checkpoint=f"M2D_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        elif config.model_name == "ASIT":
            asit = ASiTWrapper()
            model = PredictionsWrapper(asit, checkpoint=f"ASIT_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type)
        else:
            raise NotImplementedError(f"Model {config.model_name} not (yet) implemented")

        self.model = model

        # prepare ingredients for knowledge distillation
        assert 0 <= config.distillation_loss_weight <= 1, "Lambda for Knowledge Distillation must be between 0 and 1."
        self.strong_loss = nn.BCEWithLogitsLoss()

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.val_durations_df = pd.read_csv(f"resources/eval_durations.csv",
                                            sep=",", header=None, names=["filename", "duration"])
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []

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
        if 'pseudo_strong' in train_batch:
            pseudo_labels = train_batch['pseudo_strong']
        else:
            # create dummy pseudo labels
            pseudo_labels = torch.zeros_like(labels)
            assert self.config.distillation_loss_weight == 0

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
        # bring ground truth into shape needed for evaluation
        for f, gt_string in zip(val_batch["filename"], val_batch["gt_string"]):
            f = f[:-len(".mp3")]
            events = [e.split(";;") for e in gt_string.split("++")]
            self.val_ground_truth[f] = [(float(e[0]), float(e[1]), e[2]) for e in events]
            self.val_duration[f] = self.val_durations_df[self.val_durations_df["filename"] == f]["duration"].values[0]

        y_hat_strong = self(val_batch)
        y_strong = val_batch["strong"]

        loss = self.strong_loss(y_hat_strong, y_strong)
        self.val_loss.append(loss.cpu())

        scores_raw, scores_postprocessed, prediction_dfs = batched_decode_preds(
            y_hat_strong.float(),
            val_batch['filename'],
            self.encoder,
            median_filter=self.config.median_window
        )

        self.val_predictions_strong.update(
            scores_postprocessed
        )

    def on_validation_epoch_end(self):
        gt_unique_events = set([e[2] for f, events in self.val_ground_truth.items() for e in events])
        train_unique_events = set(self.encoder.labels)
        # evaluate on all classes that are in both train and test sets (407 classes)
        class_intersection = gt_unique_events.intersection(train_unique_events)

        assert len(class_intersection) == len(set(as_strong_train_classes).intersection(as_strong_eval_classes)) == 407, \
            f"Intersection unique events. Expected: {len(set(as_strong_train_classes).intersection(as_strong_eval_classes))}," \
            f" Actual: {len(class_intersection)}"

        # filter ground truth according to class_intersection
        val_ground_truth = {fid: [event for event in self.val_ground_truth[fid] if event[2] in class_intersection]
                            for fid in self.val_ground_truth}
        # drop audios without events - aligned with DESED evaluation procedure
        val_ground_truth = {fid: events for fid, events in val_ground_truth.items() if len(events) > 0}
        # keep only corresponding audio durations
        audio_durations = {
            fid: self.val_duration[fid] for fid in val_ground_truth.keys()
        }

        # filter files in predictions
        as_strong_preds = {
            fid: self.val_predictions_strong[fid] for fid in val_ground_truth.keys()
        }
        # filter classes in predictions
        unused_classes = list(set(self.encoder.labels).difference(class_intersection))
        for f, df in as_strong_preds.items():
            df.drop(columns=list(unused_classes), axis=1, inplace=True)

        segment_based_pauroc = sed_scores_eval.segment_based.auroc(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            max_fpr=0.1,
            segment_length=1.0,
            num_jobs=1
        )

        psds1 = sed_scores_eval.intersection_based.psds(
            as_strong_preds,
            val_ground_truth,
            audio_durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            num_jobs=1
        )

        # "val/psds1_macro_averaged" is psds1 without penalization for performance
        #  variations across classes
        logs = {"val/loss": torch.as_tensor(self.val_loss).mean().cuda(),
                "val/psds1": psds1[0],
                "val/psds1_macro_averaged": np.array([v for k, v in psds1[1].items()]).mean(),
                "val/pauroc": segment_based_pauroc[0]['mean'],
                }

        self.log_dict(logs, sync_dist=False)
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []


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
    encoder = ManyHotEncoder(as_strong_train_classes)

    train_set = get_training_dataset(encoder, wavmix_p=config.wavmix_p,
                                     pseudo_labels_file=config.pseudo_labels_file)
    eval_set = get_eval_dataset(encoder)

    if config.use_balanced_sampler:
        sample_weights = get_temporal_count_balanced_sample_weights(train_set, save_folder="resources")
    else:
        sample_weights = get_uniform_sample_weights(train_set)

    train_sampler = get_weighted_sampler(sample_weights, epoch_len=config.epoch_len)

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
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch
                         )

    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, eval_dl)

    wandb.finish()


def evaluate(config):
    # only evaluation of pre-trained models
    # encoder manages encoding and decoding of model predictions
    encoder = ManyHotEncoder(as_strong_train_classes)
    eval_set = get_eval_dataset(encoder)

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
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch)

    # start evaluation
    trainer.validate(pl_module, eval_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet_Strong")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)

    # model
    parser.add_argument('--model_name', type=str, default="ATST-F")  # used also for training
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    parser.add_argument('--pretrained', type=str, choices=["scratch", "ssl", "weak", "strong"],
                        default="weak")
    parser.add_argument('--seq_model_type', type=str, choices=["rnn"],
                        default=None)

    # training
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--distillation_loss_weight', type=float, default=0.0)
    parser.add_argument('--epoch_len', type=int, default=100000)
    parser.add_argument('--median_window', type=int, default=12)

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
    parser.add_argument('--max_lr', type=float, default=7e-5)
    parser.add_argument('--lr_end', type=float, default=2e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)

    # knowledge distillation
    parser.add_argument('--pseudo_labels_file', type=str,
                        default=None)

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)
