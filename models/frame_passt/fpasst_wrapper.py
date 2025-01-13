from models.frame_passt.fpasst import get_model
from models.frame_passt.preprocess import AugmentMelSTFT
from models.transformer_wrapper import BaseModelWrapper


class FPaSSTWrapper(BaseModelWrapper):
    def __init__(self):
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
        )
        self.fpasst = get_model(
            arch="passt_deit_bd_p16_384",
            n_classes=527,
            pos_embed_length=250,
            frame_patchout=0,
            in_channels=16
        )

    def mel_forward(self, x):
        return self.mel(x)

    def forward(self, x):
        return self.fpasst(x)

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.fpasst.named_parameters():
            if k in ['cls_token',
                     'dist_token',
                     'new_pos_embed',
                     'freq_new_pos_embed',
                     'time_new_pos_embed',
                     'conv_in_1.weight',
                     'conv_in_1.bias',
                     'conv_in_2.weight',
                     'conv_in_2.bias',
                     'conv_in_3.weight',
                     'conv_in_3.bias',
                     'patch_embed.proj.weight',
                     'patch_embed.proj.bias',
                     ]:
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
            elif k in ['norm.weight', 'norm.bias']:
                pt_params[11].append(p)
            else:
                raise ValueError(f"Check separate params for frame-passt! Unexpected key: {k}")
        return list(reversed(pt_params))
