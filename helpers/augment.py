import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

def frame_shift(mels, labels, embeddings=None, pseudo_labels=None,
                net_pooling=4, shift_range=0.125):
    bsz, channels, n_bands, frames = mels.shape
    abs_shift_mel = int(frames * shift_range)

    if embeddings is not None:
        embed_frames = embeddings.shape[-1]
        embed_pool_fact = frames / embed_frames

    for bindx in range(bsz):
        shift = int(random.gauss(0, abs_shift_mel))
        mels[bindx] = torch.roll(mels[bindx], shift, dims=-1)
        label_shift = -abs(shift) / net_pooling if shift < 0 else shift / net_pooling
        label_shift = round(label_shift)
        labels[bindx] = torch.roll(labels[bindx], label_shift, dims=-1)

        if pseudo_labels is not None:
            pseudo_labels[bindx] = torch.roll(pseudo_labels[bindx], label_shift, dims=-1)

        if embeddings is not None:
            embed_shift = -abs(shift) / embed_pool_fact if shift < 0 else shift / embed_pool_fact
            embed_shift = round(embed_shift)
            embeddings[bindx] = torch.roll(embeddings[bindx], embed_shift, dims=-1)

    out_args = [mels]
    if embeddings is not None:
        out_args.append(embeddings)
    out_args.append(labels)
    if pseudo_labels is not None:
        out_args.append(pseudo_labels)
    return tuple(out_args)


def time_mask(features, labels, embeddings=None, pseudo_labels=None, net_pooling=None,
              min_mask_ratio=0.05, max_mask_ratio=0.2):
    _, _, n_frame = labels.shape

    if embeddings is not None:
        embed_frames = embeddings.shape[-1]
        embed_pool_fact = embed_frames / n_frame

    t_width = torch.randint(low=int(n_frame * min_mask_ratio), high=int(n_frame * max_mask_ratio), size=(1,))
    t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
    features[:, :, :, t_low * net_pooling:(t_low+t_width)*net_pooling] = 0
    labels[:, :, t_low:t_low+t_width] = 0

    if pseudo_labels is not None:
        labels[:, :, t_low:t_low + t_width] = 0

    if embeddings is not None:
        low = round((t_low * embed_pool_fact).item())
        high = round(((t_low + t_width) * embed_pool_fact).item())
        embeddings[..., low:high] = 0

    out_args = [features]

    if embeddings is not None:
        out_args.append(embeddings)
    out_args.append(labels)
    if pseudo_labels is not None:
        out_args.append(pseudo_labels)
    return tuple(out_args)


def mixup(data, embeddings=None, targets=None, pseudo_strong=None, alpha=0.2, beta=0.2, return_mix_coef=False):
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta, size=batch_size)
        c = np.maximum(c, 1 - c)

        perm = torch.randperm(batch_size)
        cd = torch.tensor(c, dtype=data.dtype, device=data.device).view(batch_size, *([1] * (data.ndim - 1)))
        mixed_data = cd * data + (1 - cd) * data[perm, :]

        if embeddings is not None:
            ce = torch.tensor(c, dtype=embeddings.dtype, device=embeddings.device).view(batch_size, *([1] * (embeddings.ndim - 1)))
            mixed_embeddings = ce * embeddings + (1 - ce) * embeddings[perm, :]

        if targets is not None:
            ct = torch.tensor(c, dtype=data.dtype, device=data.device).view(batch_size, *([1] * (targets.ndim - 1)))
            mixed_target = torch.clamp(
                ct * targets + (1 - ct) * targets[perm, :], min=0, max=1
            )

        if pseudo_strong is not None:
            cp = torch.tensor(c, dtype=pseudo_strong.dtype, device=pseudo_strong.device).view(batch_size,
                                                                                              *([1] * (pseudo_strong.ndim - 1)))
            mixed_pseudo_strong = cp * pseudo_strong + (1 - cp) * pseudo_strong[perm, :]

    out_args = [mixed_data]
    if embeddings is not None:
        out_args.append(mixed_embeddings)
    if targets is not None:
        out_args.append(mixed_target)
    if pseudo_strong is not None:
        out_args.append(mixed_pseudo_strong)

    if return_mix_coef:
        out_args.append(perm)
        out_args.append(c)
    return tuple(out_args)


def filt_aug_(features, db_range=(-6, 6), n_band=(3, 6), min_bw=6):
    batch_size, channels, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
        freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
        for i in range(n_freq_band):
            for j in range(batch_size):
                freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                    torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                   band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
        freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt.unsqueeze(1)
    else:
        return features


def filter_augmentation(features, n_transform=1, filter_db_range=(-6, 6),
                        filter_bands=(3, 6), filter_minimum_bandwidth=6):
    if n_transform == 2:
        feature_list = []
        for _ in range(n_transform):
            features_temp = features
            features_temp = filt_aug_(features_temp, db_range=filter_db_range, n_band=filter_bands,
                                      min_bw=filter_minimum_bandwidth)
            feature_list.append(features_temp)
        return feature_list
    elif n_transform == 1:
        features = filt_aug_(features, db_range=filter_db_range, n_band=filter_bands,
                             min_bw=filter_minimum_bandwidth)
        return [features, features]
    else:
        return [features, features]


def mixstyle(x, alpha=0.4, eps=1e-6):
    batch_size = x.size(0)

    # frequency-wise statistics
    f_mu = x.mean(dim=3, keepdim=True)
    f_var = x.var(dim=3, keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device, dtype=x.dtype)  # sample instance-wise convex weights
    lmda = torch.max(lmda, 1-lmda)
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics
    return x


class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.0), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        # spec_output = []
        # for lms in specs:
        # lms = lms.unsqueeze(0)
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                            .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(crop.unsqueeze(1), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True).squeeze(1)
            # spec_output.append(lms.float())
        return lms.float() # torch.concat(lms, dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string
