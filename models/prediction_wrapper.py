import os

import torch
import torch.nn as nn
from torch.hub import download_url_to_file

from config import RESOURCES_FOLDER, CHECKPOINT_URLS
from models.seq_models import BidirectionalLSTM, BidirectionalGRU


class PredictionsWrapper(nn.Module):
    """
        A wrapper module that adds an optional sequence model and classification heads on top of a transformer.
        It implements equations (1), (2), and (3) in the paper.

        Args:
            base_model (BaseModelWrapper): The base model (transformer) providing sequence embeddings
            checkpoint (str, optional): checkpoint name for loading pre-trained weights. Default is None.
            n_classes_strong (int): Number of classes for strong predictions. Default is 447.
            n_classes_weak (int, optional): Number of classes for weak predictions. Default is None,
                                            which sets it equal to n_classes_strong.
            embed_dim (int, optional): Embedding dimension of the base model output. Default is 768.
            seq_len (int, optional): Desired sequence length. Default is 250 (40 ms resolution).
            seq_model_type (str, optional): Type of sequence model to use.
                                            Default is None, which means no additional sequence model is used.
            head_type (str, optional): Type of classification head. Choices are ["linear", "attention", "None"].
                                       Default is "linear". "None" means that sequence embeddings are returned.
            rnn_layers (int, optional): Number of RNN layers if seq_model_type is "rnn". Default is 2.
            rnn_type (str, optional): Type of RNN to use. Choices are ["BiGRU", "BiLSTM"]. Default is "BiGRU".
            rnn_dim (int, optional): Dimension of RNN hidden state if seq_model_type is "rnn". Default is 256.
            rnn_dropout (float, optional): Dropout rate for RNN layers. Default is 0.0.
        """

    def __init__(self,
                 base_model,
                 checkpoint=None,
                 n_classes_strong=447,
                 n_classes_weak=None,
                 embed_dim=768,
                 seq_len=250,
                 seq_model_type=None,
                 head_type="linear",
                 rnn_layers=2,
                 rnn_type="BiGRU",
                 rnn_dim=2048,
                 rnn_dropout=0.0
                 ):
        super(PredictionsWrapper, self).__init__()
        self.model = base_model
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_classes_strong = n_classes_strong
        self.n_classes_weak = n_classes_weak if n_classes_weak is not None else n_classes_strong
        self.seq_model_type = seq_model_type
        self.head_type = head_type

        if self.seq_model_type == "rnn":
            if rnn_type == "BiGRU":
                self.seq_model = BidirectionalGRU(
                    n_in=self.embed_dim,
                    n_hidden=rnn_dim,
                    dropout=rnn_dropout,
                    num_layers=rnn_layers
                )
            elif rnn_type == "BiLSTM":
                self.seq_model = BidirectionalLSTM(
                    nIn=self.embed_dim,
                    nHidden=rnn_dim,
                    nOut=rnn_dim * 2,
                    dropout=rnn_dropout,
                    num_layers=rnn_layers
                )
            num_features = rnn_dim * 2
        elif self.seq_model_type is None:
            self.seq_model = nn.Identity()
            # no additional sequence model
            num_features = self.embed_dim
        else:
            raise ValueError(f"Unknown seq_model_type: {self.seq_model_type}")

        if self.head_type == "attention":
            assert self.n_classes_strong == self.n_classes_weak, "head_type=='attention' requires number of strong and " \
                                                                 "weak classes to be the same!"

        if self.head_type is not None:
            self.strong_head = nn.Linear(num_features, self.n_classes_strong)
            self.weak_head = nn.Linear(num_features, self.n_classes_weak)
        if checkpoint is not None:
            print("Loading pretrained checkpoint: ", checkpoint)
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt_file = os.path.join(RESOURCES_FOLDER, checkpoint + ".pt")
        if not os.path.exists(ckpt_file):
            download_url_to_file(CHECKPOINT_URLS[checkpoint], ckpt_file)
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)

        # compatibility with uniform wrapper structure we introduced for the public repo
        if 'fpasst' in checkpoint:
            state_dict = {("model.fpasst." + k[len("model."):] if k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'M2D' in checkpoint:
            state_dict = {("model.m2d." + k[len("model."):] if not k.startswith("model.m2d.") and k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'BEATs' in checkpoint:
            state_dict = {("model.beats." + k[len("model.model."):] if k.startswith("model.model")
                           else k): v for k, v in state_dict.items()}
        elif 'ASIT' in checkpoint:
            state_dict = {("model.asit." + k[len("model."):] if k.startswith("model.")
                           else k): v for k, v in state_dict.items()}

        n_classes_weak_in_sd = state_dict['weak_head.bias'].shape[0] if 'weak_head.bias' in state_dict else -1
        n_classes_strong_in_sd = state_dict['strong_head.bias'].shape[0] if 'strong_head.bias' in state_dict else -1
        seq_model_in_sd = any(['seq_model.' in key for key in state_dict.keys()])
        keys_to_remove = []
        strict = True
        expected_missing = 0
        if self.head_type is None:
            # remove all keys related to head
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
        elif self.seq_model_type is not None and not seq_model_in_sd:
            # we want to train a sequence model (e.g., rnn) on top of a
            #   pre-trained transformer (e.g., AS weak pretrained)
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
            num_seq_model_keys = len([key for key in self.seq_model.state_dict()])
            expected_missing = len(keys_to_remove) + num_seq_model_keys
            strict = False
        else:
            # head type is not None
            if n_classes_weak_in_sd != self.n_classes_weak:
                # remove weak head from sd
                keys_to_remove.append('weak_head.bias')
                keys_to_remove.append('weak_head.weight')
                strict = False
            if n_classes_strong_in_sd != self.n_classes_strong:
                # remove strong head from sd
                keys_to_remove.append('strong_head.bias')
                keys_to_remove.append('strong_head.weight')
                strict = False
            expected_missing = len(keys_to_remove)

        # allow missing mel parameters for compatibility
        num_mel_keys = len([key for key in self.state_dict() if 'mel_transform' in key])
        if num_mel_keys > 0:
            expected_missing += num_mel_keys
            strict = False

        state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        assert len(missing) == expected_missing
        assert len(unexpected) == 0

    def separate_params(self):
        if hasattr(self, "separate_params"):
            return self.model.separate_params()
        else:
            raise NotImplementedError("The base model has no 'separate_params' method!'")

    def has_separate_params(self):
        return hasattr(self.model, "separate_params")

    def mel_forward(self, x):
        return self.model.mel_forward(x)

    def forward(self, x):
        # base model is expected to output a sequence (see Eq. (1) in paper)
        # (batch size x sequence length x embedding dimension)
        x = self.model(x)

        # ATST: x.shape: batch size x 250 x 768
        # PaSST: x.shape: batch size x 250 x 768
        # ASiT: x.shape: batch size x 497 x 768
        # M2D: x.shape: batch size x 62 x 3840
        # BEATs: x.shape: batch size x 496 x 768

        assert len(x.shape) == 3

        if x.size(-2) > self.seq_len:
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.seq_len).transpose(1, 2)
        elif x.size(-2) < self.seq_len:
            x = torch.nn.functional.interpolate(x.transpose(1, 2), size=self.seq_len,
                                                mode='linear').transpose(1, 2)

        # Eq. (3) in the paper
        # for teachers this is an RNN, for students it is nn.Identity
        x = self.seq_model(x)

        if self.head_type == "attention":
            # attention head to obtain weak from strong predictions
            # this is typically used for the DESED task, which requires both
            # weak and strong predictions
            strong = torch.sigmoid(self.strong_head(x))
            sof = torch.softmax(self.weak_head(x), dim=-1)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)
            return strong.transpose(1, 2), weak
        elif self.head_type == "linear":
            # simple linear layers as head (see Eq. (3) in the paper)
            # on AudioSet strong, only strong predictions are used
            # on AudioSet weak, only weak predictions are used
            # why both? because we tried to simultaneously train on AudioSet weak and strong (less successful)
            strong = self.strong_head(x)
            weak = self.weak_head(x.mean(dim=1))
            return strong.transpose(1, 2), weak
        else:
            # no head means the sequence is returned instead of strong and weak predictions
            return x
