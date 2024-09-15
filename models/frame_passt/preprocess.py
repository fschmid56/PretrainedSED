import torch
import torch.nn as nn
import torchaudio

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


class AugmentMelSTFT(nn.Module):
    def __init__(
            self,
            n_mels=128,
            sr=32000,
            win_length=None,
            hopsize=320,
            n_fft=1024,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=1,
            fmax_aug_range=1,
            fast_norm=False,
            preamp=True,
            padding="center",
            periodic_window=True,
    ):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        if win_length is None:
            win_length = n_fft

        if isinstance(win_length, list) or isinstance(win_length, tuple):
            assert isinstance(n_fft, list) or isinstance(n_fft, tuple)
            assert len(win_length) == len(n_fft)
        else:
            win_length = [win_length]
            n_fft = [n_fft]

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.preamp = preamp
        for win_l in self.win_length:
            self.register_buffer(
                f"window_{win_l}",
                torch.hann_window(win_l, periodic=periodic_window),
                persistent=False,
            )
        assert (
                fmin_aug_range >= 1
        ), f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert (
                fmin_aug_range >= 1
        ), f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1]]]), persistent=False
        )
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=False)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=False)
        self.fast_norm = fast_norm
        self.padding = padding
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.iden = nn.Identity()

    def forward(self, x):
        if self.preamp:
            x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient)
        x = x.squeeze(1)

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()

        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mels = []
        for n_fft, win_length in zip(self.n_fft, self.win_length):
            x_temp = x
            if self.padding == "same":
                pad = win_length - self.hopsize
                self.iden(x_temp)  # printing
                x_temp = torch.nn.functional.pad(x_temp, (pad // 2, pad // 2), mode="reflect")
                self.iden(x_temp)  # printing

            x_temp = torch.stft(
                x_temp,
                n_fft,
                hop_length=self.hopsize,
                win_length=win_length,
                center=self.padding == "center",
                normalized=False,
                window=getattr(self, f"window_{win_length}"),
                return_complex=True
            )
            x_temp = torch.view_as_real(x_temp)
            x_temp = (x_temp ** 2).sum(dim=-1)  # power mag

            mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, n_fft, self.sr,
                                                                     fmin, fmax, vtln_low=100.0, vtln_high=-500.,
                                                                     vtln_warp_factor=1.0)
            mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                        device=x.device)

            with torch.cuda.amp.autocast(enabled=False):
                x_temp = torch.matmul(mel_basis, x_temp)

            x_temp = torch.log(torch.clip(x_temp, min=1e-7))

            mels.append(x_temp)

        mels = torch.stack(mels, dim=1)

        if self.training:
            mels = self.freqm(mels)
            mels = self.timem(mels)
        if self.fast_norm:
            mels = (mels + 4.5) / 5.0  # fast normalization

        return mels

    def extra_repr(self):
        return "winsize={}, hopsize={}".format(self.win_length, self.hopsize)
