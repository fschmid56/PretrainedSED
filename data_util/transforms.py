import os

import datasets
import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio

from data_util.audioset_classes import as_strong_train_classes

## Transforms with a similar style to https://github.com/descriptinc/audiotools/blob/master/audiotools/data/transforms.py
logger = datasets.logging.get_logger(__name__)


def target_transform(sample):
    del sample["labels"]
    del sample["label_ids"]
    return sample


def strong_label_transform(sample, strong_label_encoder=None):
    assert strong_label_encoder is not None
    events = pd.DataFrame(sample['events'][0])
    events = events[events['event_label'].isin(set(as_strong_train_classes))]
    strong = strong_label_encoder.encode_strong_df(events).T
    sample["strong"] = [strong]
    sample["event_count"] = [strong.sum(1)]
    sample["gt_string"] = ["++".join([";;".join([str(e[0]), str(e[1]), e[2]]) for e in
                                      zip(sample['events'][0]['onset'], sample['events'][0]['offset'],
                                          sample['events'][0]['event_label'])])]
    del sample['events']
    return sample


class AddPseudoLabelsTransform:
    def __init__(self, pseudo_labels_folder="/share/hel/datasets/as_strong/predictions",
                 pseudo_labels_name="final"):
        self.pseudo_labels_file = os.path.join(pseudo_labels_folder, pseudo_labels_name, "as_strong.hdf5")

        if self.pseudo_labels_file is not None:
            # fetch dict of positions for each example
            self.ex2pseudo_idx = {}
            f = h5py.File(self.pseudo_labels_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2pseudo_idx[fname.decode("UTF-8")] = i
        self._opened_pseudo_hdf5 = None

    @property
    def pseudo_hdf5_file(self):
        if self._opened_pseudo_hdf5 is None:
            self._opened_pseudo_hdf5 = h5py.File(self.pseudo_labels_file, "r")
        return self._opened_pseudo_hdf5

    def add_pseudo_label_transform(self, sample):
        indices = [self.ex2pseudo_idx[fn] for fn in sample['filename']]
        pseudo_strong = [torch.from_numpy(np.stack(self.pseudo_hdf5_file["strong_logits"][index])).float()
                         for index in indices]
        pseudo_strong = [torch.sigmoid(pseudo_strong[i]) for i in range(len(pseudo_strong))]
        sample['pseudo_strong'] = pseudo_strong
        return sample


class SequentialTransform:
    """Apply a sequence of transforms to a batch."""

    def __init__(self, transforms):
        """
        Args:
            transforms: list of transforms to apply
        """
        self.transforms = transforms

    def append(self, transform):
        self.transforms.append(transform)

    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch


class Mp3DecodeTransform:
    def __init__(
            self,
            mp3_bytes_key="mp3_bytes",
            audio_key="audio",
            sample_rate=32000,
            max_length=10.0,
            min_length=None,
            random_sample_crop=True,
            allow_resample=True,
            resampling_method="sinc_interp_kaiser",
            keep_mp3_bytes=False,
            debug_info_key=None,
    ):
        """Decode mp3 bytes to audio waveform

        Args:
            mp3_bytes_key (str, optional): The key to mp3 bytes in the input batch. Defaults to "mp3_bytes".
            audio_key (str, optional): The key to save the decoded audio in the output batch. Defaults to "audio".
            sample_rate (int, optional): The expected output audio_key. Defaults to 32000.
            max_length (int, float, optional): the maximum output audio length in seconds if float, otherwise in samples. Defaults to 10.
            min_length (int, optional): the minimum output audio length in seconds. Defaults to max_length.
            random_sample_crop (bool, optional): Randomly crop the audio to max_length if its longer otherwise return the first crop. Defaults to True.
            allow_resample (bool, optional): Resample the singal if the sampling rate don't match. Defaults to True.
            resampling_method (str, optional): reampling method from torchaudio.transforms.Resample  . Defaults to "sinc_interp_kaiser".
            keep_mp3_bytes (bool, optional): keep the original bytes in the output dict. Defaults to False.

        Raises:
            Exception: if minimp3py is not installed
        """
        self.mp3_bytes_key = mp3_bytes_key
        self.audio_key = audio_key
        self.sample_rate = sample_rate
        self.max_length = max_length
        if min_length is None:
            min_length = max_length
        self.min_length = min_length
        self.random_sample_crop = random_sample_crop
        self.allow_resample = allow_resample
        self.resampling_method = resampling_method
        self.keep_mp3_bytes = keep_mp3_bytes
        self.debug_info_key = debug_info_key
        self.resamplers_cache = {}
        try:
            import minimp3py  # noqa: F401
        except:
            raise Exception(
                "minimp3py is not installed, please install it using: `CFLAGS='-O3 -march=native' pip install https://github.com/f0k/minimp3py/archive/master.zip`"
            )

    def __call__(self, batch):
        import minimp3py

        data_list = batch[self.mp3_bytes_key]
        if self.debug_info_key is not None:
            file_name_list = batch[self.debug_info_key]
        else:
            file_name_list = range(len(data_list))
        audio_list = []
        for data, file_name in zip(data_list, file_name_list):
            try:
                duration, ch, sr = minimp3py.probe(data)
                if isinstance(self.max_length, float):
                    max_length = int(self.max_length * sr)
                else:
                    max_length = int(self.max_length * sr // self.sample_rate)
                offset = 0
                if self.random_sample_crop and duration > max_length:
                    max_offset = max(int(duration - max_length), 0) + 1
                    offset = torch.randint(max_offset, (1,)).item()
                waveform, _ = minimp3py.read(data, start=offset, length=max_length)
                waveform = waveform[:, 0]  # 0 for the first channel only
                if waveform.dtype != "float32":
                    raise RuntimeError("Unexpected wave type")

                waveform = torch.from_numpy(waveform)
                if len(waveform) == 0:
                    logger.warning(
                        f"Empty waveform for {file_name}, duration {duration}, offset {offset}, max_length {max_length}, sr {sr}, ch {ch}"
                    )
                elif sr != self.sample_rate:
                    assert self.allow_resample, f"Unexpected sample rate {sr} instead of {self.sample_rate} at {file_name}"
                    if self.resamplers_cache.get(sr) is None:
                        self.resamplers_cache[sr] = torchaudio.transforms.Resample(
                            sr,
                            self.sample_rate,
                            resampling_method=self.resampling_method,
                        )
                    waveform = self.resamplers_cache[sr](waveform)
                min_length = self.min_length
                if isinstance(self.min_length, float):
                    min_length = int(self.min_length * self.sample_rate)
                if min_length is not None and len(waveform) < min_length:
                    waveform = torch.concatenate(
                        (
                            waveform,
                            torch.zeros(
                                min_length - len(waveform),
                                dtype=waveform.dtype,
                                device=waveform.device,
                            ),
                        ),
                        dim=0,
                    )
                audio_list.append(waveform)
            except Exception as e:
                print(f"Error decoding {file_name}: {e}")
                raise e
        batch[self.audio_key] = audio_list
        batch["sampling_rate"] = [self.sample_rate] * len(audio_list)
        if not self.keep_mp3_bytes:
            del batch[self.mp3_bytes_key]
        return batch
