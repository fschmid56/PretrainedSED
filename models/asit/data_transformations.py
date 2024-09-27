import torch
import torch.nn.functional
import torchaudio


class DataAugmentation(object):
    def __init__(self, data_mean=-4.2677393, data_std=4.5689974, num_mel_bins=128, sample_rate=16000):
        self.data_mean = data_mean
        self.data_std = data_std
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate

    def _wav2fbank(self, waveform):
        waveform = (waveform - waveform.mean())
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate,
                                                  use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.num_mel_bins, dither=0.0,
                                                  frame_shift=10)
        return fbank

    def convert_waveform(self, waveform):
        w = self._wav2fbank(waveform)
        fbank = (w - self.data_mean) / (self.data_std * 2)
        fbank = fbank.unsqueeze(0)
        return fbank

    def __call__(self, batch):
        # apply convert_waveform to each sample of the batch and return the result
        return torch.stack([self.convert_waveform(sample.reshape(1, -1)) for sample in batch]).permute(0, 1, 3, 2)
