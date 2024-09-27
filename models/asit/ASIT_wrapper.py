from models.asit.data_transformations import DataAugmentation
from models.asit.vision_transformer import vit_base
from models.transformer_wrapper import BaseModelWrapper


class ASiTWrapper(BaseModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.asit_mel = DataAugmentation()
        self.asit = vit_base(
            patch_size=[16, 16],
            audio_size=[128, 592],
            stride=[16, 16],
            in_chans=1,
            num_classes=0
        )

    def mel_forward(self, x):
        return self.asit_mel(x)

    def forward(self, spec):
        return self.asit(spec)

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
            elif 'backbone.norm.weight' in k or 'backbone.norm.bias' in k:
                pt_params[11].append(p)
            else:
                raise ValueError("Check layer-wise learning for frame-passt!")
        return list(reversed(pt_params))
