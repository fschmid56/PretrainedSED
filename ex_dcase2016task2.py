import argparse
import random
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from data_util.dcase2016task2 import (get_training_dataset, get_validation_dataset, get_test_dataset,
                                      label_vocab_nlabels, label_vocab_as_dict)
from helpers.augment import frame_shift, time_mask, mixup, filter_augmentation, mixstyle, RandomResizeCrop
from helpers.score import get_events_for_all_files, combine_target_events, EventBasedScore, SegmentBasedScore
from helpers.utils import worker_init_fn
from models.asit.ASIT_wrapper import ASiTWrapper
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.m2d.M2D_wrapper import M2DWrapper
from models.prediction_wrapper import PredictionsWrapper


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

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
                                       seq_model_type=config.seq_model_type,
                                       n_classes_strong=self.config.n_classes)
        elif config.model_name == "ATST-F":
            atst = ATSTWrapper()
            model = PredictionsWrapper(atst, checkpoint=f"ATST-F_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type,
                                       n_classes_strong=self.config.n_classes)
        elif config.model_name == "fpasst":
            fpasst = FPaSSTWrapper()
            model = PredictionsWrapper(fpasst, checkpoint=f"fpasst_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type,
                                       n_classes_strong=self.config.n_classes)
        elif config.model_name == "M2D":
            m2d = M2DWrapper()
            model = PredictionsWrapper(m2d, checkpoint=f"M2D_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type,
                                       n_classes_strong=self.config.n_classes,
                                       embed_dim=m2d.m2d.cfg.feature_d)
        elif config.model_name == "ASIT":
            asit = ASiTWrapper()
            model = PredictionsWrapper(asit, checkpoint=f"ASIT_{checkpoint}" if checkpoint else None,
                                       seq_model_type=config.seq_model_type,
                                       n_classes_strong=self.config.n_classes)
        else:
            raise NotImplementedError(f"Model {config.model_name} not (yet) implemented")

        self.model = model
        self.strong_loss = nn.BCEWithLogitsLoss()

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        task_path = Path(self.config.task_path)
        label_vocab, nlabels = label_vocab_nlabels(task_path)
        self.label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

        self.idx_to_label: Dict[int, str] = {
            idx: label for (label, idx) in self.label_to_idx.items()
        }

        self.event_onset_200ms_fms = EventBasedScore(
            label_to_idx=self.label_to_idx,
            name="event_onset_200ms_fms",
            scores=("f_measure", "precision", "recall"),
            params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.2}
        )

        self.event_onset_50ms_fms = EventBasedScore(
            label_to_idx=self.label_to_idx,
            name="event_onset_50ms_fms",
            scores=("f_measure", "precision", "recall"),
            params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.05}
        )

        self.segment_1s_er = SegmentBasedScore(
            label_to_idx=self.label_to_idx,
            name="segment_1s_er",
            scores=("error_rate",),
            params={"time_resolution": 1.0},
            maximize=False,
        )

        self.postprocessing_grid = {
            "median_filter_ms": [
                250
            ],
            "min_duration": [
                125
            ]
        }

        self.preds, self.tgts, self.fnames, self.timestamps = [], [], [], []

    def forward(self, audio):
        mel = self.model.mel_forward(audio)
        y_strong, _ = self.model(mel)
        return y_strong

    def separate_params(self):
        pt_params = []
        seq_params = []
        head_params = []

        for name, p in self.named_parameters():
            name = name[len("model."):]
            if name.startswith('model'):
                # the transformer
                pt_params.append(p)
            elif name.startswith('seq_model'):
                # the optional sequence model
                seq_params.append(p)
            elif name.startswith('strong_head') or name.startswith('weak_head'):
                # the prediction head
                head_params.append(p)
            else:
                raise ValueError(f"Unexpected key in model: {name}")

        if self.model.has_separate_params():
            # split parameters into groups according to their depth in the network
            # based on this, we can apply layer-wise learning rate decay
            pt_params = self.model.separate_params()
        else:
            if self.config.lr_decay != 1.0:
                raise ValueError(f"Model has no separate_params function. Can't apply layer-wise lr decay, but "
                                 f"learning rate decay is set to {self.config.lr_decay}.")

        return pt_params, seq_params, head_params

    def get_optimizer(
            self,
            lr,
            lr_decay=1.0,
            transformer_lr=None,
            transformer_frozen=False,
            adamw=False,
            weight_decay=0.01,
            betas=(0.9, 0.999)
    ):
        pt_params, seq_params, head_params = self.separate_params()

        param_groups = [
            {'params': head_params, 'lr': lr},  # model head (besides base model and seq model)
        ]

        if transformer_frozen:
            for p in pt_params + seq_params:
                if isinstance(p, list):
                    for p_i in p:
                        p_i.detach_()
                else:
                    p.detach_()
        else:
            if transformer_lr is None:
                transformer_lr = lr
            if isinstance(pt_params, list) and isinstance(pt_params[0], list):
                # apply lr decay
                scale_lrs = [transformer_lr * (lr_decay ** i) for i in range(1, len(pt_params) + 1)]
                param_groups = param_groups + [{"params": pt_params[i], "lr": scale_lrs[i]} for i in
                                               range(len(pt_params))]
            else:
                param_groups.append(
                    {'params': pt_params, 'lr': transformer_lr},  # pretrained model
                )
            param_groups.append(
                {'params': seq_params, 'lr': lr},  # pretrained model
            )

        # do not apply weight decay to biases and batch norms
        param_groups_split = []
        for param_group in param_groups:
            params_1D, params_2D = [], []
            lr = param_group['lr']
            for param in param_group['params']:
                if param.ndimension() >= 2:
                    params_2D.append(param)
                elif param.ndimension() <= 1:
                    params_1D.append(param)
            param_groups_split += [{'params': params_2D, 'lr': lr, 'weight_decay': weight_decay},
                                   {'params': params_1D, 'lr': lr}]
        if weight_decay > 0:
            assert adamw
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups_split, lr=lr, weight_decay=weight_decay, betas=betas)
        return torch.optim.Adam(param_groups_split, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=4000,
            lr_end=1e-7,
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
        optimizer = self.get_optimizer(self.config.max_lr,
                                       lr_decay=self.config.lr_decay,
                                       transformer_lr=self.config.transformer_lr,
                                       transformer_frozen=self.config.transformer_frozen,
                                       adamw=False if self.config.no_adamw else True,
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

        audios, labels, fnames, timestamps = train_batch

        if self.config.transformer_frozen:
            self.model.model.eval()
            self.model.seq_model.eval()
        mel = self.model.mel_forward(audios)

        # time rolling
        if self.config.frame_shift_range > 0:
            mel, labels = frame_shift(
                mel,
                labels,
                shift_range=self.config.frame_shift_range
            )

        # mixup
        if self.config.mixup_p > random.random():
            mel, labels = mixup(
                mel,
                targets=labels
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

        loss = self.strong_loss(y_hat_strong, labels)

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)

        return loss

    def _score_step(self, batch):
        audios, labels, fnames, timestamps = batch

        strong_preds = self.forward(audios)

        self.preds.append(strong_preds)
        self.tgts.append(labels)
        self.fnames.append(fnames)
        self.timestamps.append(timestamps)

    def _score_epoch_end(self, name="val"):
        preds = torch.cat(self.preds)
        tgts = torch.cat(self.tgts)
        fnames = [item for sublist in self.fnames for item in sublist]
        timestamps = torch.cat(self.timestamps)
        val_loss = self.strong_loss(preds, tgts)
        self.log(f"{name}/loss", val_loss, prog_bar=True)

        # the following function expects one prediction per timestamp (sequence dimension must be flattened)
        seq_len = preds.size(-1)
        preds = rearrange(preds, 'bs c t -> (bs t) c').float()
        timestamps = rearrange(timestamps, 'bs t -> (bs t)').float()
        fnames = [fname for fname in fnames for _ in range(seq_len)]

        predicted_events_by_postprocessing = get_events_for_all_files(
            preds,
            fnames,
            timestamps,
            self.idx_to_label,
            self.postprocessing_grid
        )

        # we only have one postprocessing configurations (aligned with HEAR challenge)
        key = list(predicted_events_by_postprocessing.keys())[0]
        predicted_events = predicted_events_by_postprocessing[key]

        # load ground truth for test fold
        task_path = Path(self.config.task_path)
        test_target_events = combine_target_events(["valid" if name == "val" else "test"], task_path)
        onset_fms = self.event_onset_200ms_fms(predicted_events, test_target_events)
        onset_fms_50 = self.event_onset_50ms_fms(predicted_events, test_target_events)
        segment_1s_er = self.segment_1s_er(predicted_events, test_target_events)

        self.log(f"{name}/onset_fms", onset_fms[0][1])
        self.log(f"{name}/onset_fms_50", onset_fms_50[0][1])
        self.log(f"{name}/segment_1s_er", segment_1s_er[0][1])

        # free buffers
        self.preds, self.tgts, self.fnames, self.timestamps = [], [], [], []

    def validation_step(self, batch, batch_idx):
        self._score_step(batch)

    def on_validation_epoch_end(self):
        self._score_epoch_end(name="val")

    def test_step(self, batch, batch_idx):
        self._score_step(batch)

    def on_test_epoch_end(self):
        self._score_epoch_end(name="test")


def train(config):
    # Example for fine-tuning pre-trained transformers on a downstream task.

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="PTSED",
        notes="Downstream Training on office sound event detection.",
        tags=["DCASE 2016 Task 2", "Sound Event Detection"],
        config=config,
        name=config.experiment_name
    )

    train_set = get_training_dataset(config.task_path, wavmix_p=config.wavmix_p)
    val_ds = get_validation_dataset(config.task_path)
    test_ds = get_test_dataset(config.task_path)

    # train dataloader
    train_dl = DataLoader(dataset=train_set,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    # validation dataloader
    valid_dl = DataLoader(dataset=val_ds,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=False,
                          drop_last=False)

    # test dataloader
    test_dl = DataLoader(dataset=test_ds,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         shuffle=False,
                         drop_last=False)

    # create pytorch lightening module
    pl_module = PLModule(config)

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
    trainer.fit(
        pl_module,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )

    test_results = trainer.test(pl_module, dataloaders=test_dl)
    print(test_results)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parser. ')

    # general
    parser.add_argument('--task_path', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default="DCASE2016Task2")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)

    # model
    parser.add_argument('--model_name', type=str,
                        choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"],
                        default="ATST-F")  # used also for training
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    parser.add_argument('--pretrained', type=str, choices=["scratch", "ssl", "weak", "strong"],
                        default="strong")
    parser.add_argument('--seq_model_type', type=str, choices=["rnn"],
                        default=None)
    parser.add_argument('--n_classes', type=int, default=11)

    # training
    parser.add_argument('--n_epochs', type=int, default=300)

    # augmentation
    parser.add_argument('--wavmix_p', type=float, default=0.5)
    parser.add_argument('--freq_warp_p', type=float, default=0.0)
    parser.add_argument('--filter_augment_p', type=float, default=0.0)
    parser.add_argument('--frame_shift_range', type=float, default=0.0)  # in seconds
    parser.add_argument('--mixup_p', type=float, default=0.5)
    parser.add_argument('--mixstyle_p', type=float, default=0.0)
    parser.add_argument('--max_time_mask_size', type=float, default=0.0)

    # optimizer
    parser.add_argument('--no_adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--transformer_frozen', action='store_true', dest='transformer_frozen',
                        default=False,
                        help='Disable training for the transformer.')

    # lr schedule
    parser.add_argument('--schedule_mode', type=str, default="cos")
    parser.add_argument('--max_lr', type=float, default=1.06e-4)
    parser.add_argument('--transformer_lr', type=float, default=None)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--lr_end', type=float, default=1e-7)
    parser.add_argument('--warmup_steps', type=int, default=100)

    args = parser.parse_args()
    train(args)
