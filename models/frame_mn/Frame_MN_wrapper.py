from models.frame_passt.preprocess import AugmentMelSTFT
from models.transformer_wrapper import BaseModelWrapper
from models.frame_mn.model import get_model


class FrameMNWrapper(BaseModelWrapper):
    def __init__(self, width_mult=1.0) -> None:
        super().__init__()
        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=16_000,
            win_length=400,
            hopsize=160,
            n_fft=512,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
            fast_norm=True,
            preamp=True,
            padding="center",
            periodic_window=False,
        )

        self.frame_mn = get_model(
            width_mult=width_mult
        )

    def mel_forward(self, x):
        return self.mel(x)

    def forward(self, x):
        return self.frame_mn(x)

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.named_parameters():
            if any(['cls_token' in k,
                    'pos_embed' in k,
                    'norm_stats' in k,
                    'patch_embed' in k]):
                pt_params[0].append(p)
            elif 'blocks.0.' in k:
                pt_params[0].append(p)
            elif 'blocks.1.' in k:
                pt_params[1].append(p)
            elif 'blocks.2.' in k:
                pt_params[2].append(p)
            elif 'blocks.3.' in k:
                pt_params[3].append(p)
            elif 'blocks.4.' in k:
                pt_params[4].append(p)
            elif 'blocks.5.' in k:
                pt_params[5].append(p)
            elif 'blocks.6.' in k:
                pt_params[6].append(p)
            elif 'blocks.7.' in k:
                pt_params[7].append(p)
            elif 'blocks.8.' in k:
                pt_params[8].append(p)
            elif 'blocks.9.' in k:
                pt_params[9].append(p)
            elif 'blocks.10.' in k:
                pt_params[10].append(p)
            elif 'blocks.11.' in k:
                pt_params[11].append(p)
            elif 'asit.norm.weight' in k or 'asit.norm.bias' in k:
                pt_params[11].append(p)
            else:
                raise ValueError(f"Check separate params for ASiT! Unknown key: {k}")
        return list(reversed(pt_params))
