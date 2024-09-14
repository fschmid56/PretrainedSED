from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModelWrapper(ABC, nn.Module):
    @abstractmethod
    def mel_forward(self, x):
        """Process input waveform to mel spectrogram."""
        pass

    @abstractmethod
    def forward(self, x):
        """Extract embedding sequence from mel spectrogram."""
        pass

    @abstractmethod
    def separate_params(self):
        """Separate model parameters into predefined groups for layer-wise learning rate decay."""
        pass