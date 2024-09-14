import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from models.atstframe.audio_transformer import FrameASTModel
from models.transformer_wrapper import BaseModelWrapper


class ATSTWrapper(BaseModelWrapper):
    def __init__(self, atst_dropout=0.0) -> None:
        super().__init__()
        self.atst_mel = ATSTMel()
        self.atst = FrameASTModel(atst_dropout=atst_dropout)
        self.fake_length = torch.tensor([1001])
        self.cls_embed = None

    def mel_forward(self, x):
        return self.atst_mel(x)

    def forward(self, spec):
        atst_x = self.atst.get_intermediate_layers(
            spec,
            self.fake_length.to(spec).repeat(len(spec)),
            1,
            scene=False
        )
        return atst_x

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.named_parameters():
            if k in ['atst.mask_embed', 'atst.pos_embed', 'atst.patch_embed.patch_embed.weight',
                     'atst.patch_embed.patch_embed.bias'] or "blocks.0." in k:
                pt_params[0].append(p)
            elif "blocks.1." in k:
                pt_params[1].append(p)
            elif "blocks.2." in k:
                pt_params[2].append(p)
            elif "blocks.3." in k:
                pt_params[3].append(p)
            elif "blocks.4." in k:
                pt_params[4].append(p)
            elif "blocks.5." in k:
                pt_params[5].append(p)
            elif "blocks.6." in k:
                pt_params[6].append(p)
            elif "blocks.7." in k:
                pt_params[7].append(p)
            elif "blocks.8" in k:
                pt_params[8].append(p)
            elif "blocks.9." in k:
                pt_params[9].append(p)
            elif "blocks.10." in k:
                pt_params[10].append(p)
            elif "blocks.11." in k or ".norm_frame." in k:
                pt_params[11].append(p)
            else:
                raise ValueError(f"Unexpected parameters name: {k}")
        return list(reversed(pt_params))


class ATSTMel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel_transform = MelSpectrogram(
            16000,
            f_min=60,
            f_max=7800,
            hop_length=160,
            win_length=1024,
            n_fft=1024,
            n_mels=64
        )
        self.amp_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.scaler = MinMax(min=-79.6482, max=50.6842)

    def amp2db(self, spec):
        return self.amp_to_db(spec).clamp(min=-50, max=80)

    def forward(self, audio):
        with torch.autocast(device_type="cuda", enabled=False):
            spec = self.mel_transform(audio)
        spec = self.scaler(self.amp2db(spec))
        spec = spec.unsqueeze(1)
        return spec


class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(CustomAudioTransform):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, input):
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_) / (max_ - min_) * 2. - 1.
        return input
