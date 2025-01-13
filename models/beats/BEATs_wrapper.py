import torch

from models.beats.BEATs import BEATsConfig, BEATs
from models.transformer_wrapper import BaseModelWrapper


class BEATsWrapper(BaseModelWrapper):
    def __init__(self):
        super().__init__()
        cfg = BEATsConfig()
        self.beats = BEATs(cfg)

    def mel_forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            mel = self.beats.preprocess(x)
        mel = mel.unsqueeze(1).transpose(2, 3)
        return mel

    def forward(self, x):
        x = x.transpose(2, 3)
        features = self.beats.extract_features(x, do_preprocess=False)[0]
        return features

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.named_parameters():
            if ".layers.0." in k:
                pt_params[0].append(p)
            elif ".layers.1." in k:
                pt_params[1].append(p)
            elif ".layers.2." in k:
                pt_params[2].append(p)
            elif ".layers.3." in k:
                pt_params[3].append(p)
            elif ".layers.4." in k:
                pt_params[4].append(p)
            elif ".layers.5." in k:
                pt_params[5].append(p)
            elif ".layers.6." in k:
                pt_params[6].append(p)
            elif ".layers.7." in k:
                pt_params[7].append(p)
            elif ".layers.8." in k:
                pt_params[8].append(p)
            elif ".layers.9." in k:
                pt_params[9].append(p)
            elif ".layers.10." in k:
                pt_params[10].append(p)
            elif ".layers.11." in k:
                pt_params[11].append(p)
            elif (".post_extract_proj." in k or ".patch_embedding." in k or '.pos_conv.' in k
                  or 'beats.layer_norm.' in k or "beats.encoder.layer_norm." in k):
                pt_params[0].append(p)
            else:
                raise ValueError(f"Check separate params for BEATs! Unknown key: {k}")
        return list(reversed(pt_params))
