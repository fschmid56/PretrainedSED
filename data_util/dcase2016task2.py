import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from intervaltree import IntervalTree
from torch.utils.data import Dataset


class FixCropDataset(Dataset):
    """
    Read in a JSON file and return audio and audio filenames
    """

    def __init__(self, data: Dict,
                 audio_dir: Path,
                 sample_rate: int,
                 label_fps: int,
                 label_to_idx: Dict,
                 nlabels: int):
        self.clip_len = 120
        self.target_len = 10
        self.pieces_per_clip = self.clip_len // self.target_len
        self.filenames = list(data.keys())
        self.audio_dir = audio_dir
        assert self.audio_dir.is_dir(), f"{audio_dir} is not a directory"
        self.sample_rate = sample_rate
        # all files are 120 seconds long, split them into 12 x 10 second pieces
        self.pieces = []
        self.labels = []
        self.timestamps = []
        for filename in self.filenames:
            self.pieces += [(filename, i) for i in range(self.pieces_per_clip)]
            labels = data[filename]
            frame_len = 1000 / label_fps
            timestamps = np.arange(label_fps * self.clip_len) * frame_len + 0.5 * frame_len
            timestamp_labels = get_labels_for_timestamps(labels, timestamps)
            ys = []
            for timestamp_label in timestamp_labels:
                timestamp_label_idxs = [label_to_idx[str(event)] for event in timestamp_label]
                y_timestamp = label_to_binary_vector(timestamp_label_idxs, nlabels)
                ys.append(y_timestamp)
            ys = torch.stack(ys)
            frames_per_clip = ys.size(0) // self.pieces_per_clip
            self.labels += [ys[frames_per_clip * i: frames_per_clip * (i + 1)] for i in range(self.pieces_per_clip)]
            self.timestamps += [timestamps[frames_per_clip * i: frames_per_clip * (i + 1)] for i in
                                range(self.pieces_per_clip)]

        assert len(self.labels) == len(self.pieces) == len(self.filenames) * self.pieces_per_clip

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        filename = self.pieces[idx][0]
        piece = self.pieces[idx][1]
        audio_path = self.audio_dir.joinpath(filename)
        audio, sr = sf.read(str(audio_path), dtype=np.float32)
        assert sr == self.sample_rate
        start = self.sample_rate * piece * self.target_len
        end = start + self.sample_rate * self.target_len
        audio = audio[start:end]
        return audio, self.labels[idx].transpose(0, 1), filename, self.timestamps[idx]


class RandomCropDataset(Dataset):
    """
    Read in a JSON file and return audio and audio filenames
    """

    def __init__(self, data: Dict,
                 audio_dir: Path,
                 sample_rate: int,
                 label_fps: int,
                 label_to_idx: Dict,
                 nlabels: int):
        self.clip_len = 120
        self.target_len = 10
        self.pieces_per_clip = self.clip_len // self.target_len
        self.filenames = list(data.keys())
        self.audio_dir = audio_dir
        assert self.audio_dir.is_dir(), f"{audio_dir} is not a directory"
        self.sample_rate = sample_rate
        self.label_fps = label_fps
        # all files are 120 seconds long, randomly crop 10 seconds snippets
        self.labels = []
        self.timestamps = []
        for filename in self.filenames:
            labels = data[filename]
            frame_len = 1000 / label_fps
            timestamps = np.arange(label_fps * self.clip_len) * frame_len + 0.5 * frame_len
            timestamp_labels = get_labels_for_timestamps(labels, timestamps)
            ys = []
            for timestamp_label in timestamp_labels:
                timestamp_label_idxs = [label_to_idx[str(event)] for event in timestamp_label]
                y_timestamp = label_to_binary_vector(timestamp_label_idxs, nlabels)
                ys.append(y_timestamp)
            ys = torch.stack(ys)
            self.labels.append(ys)
            self.timestamps.append(timestamps)

        assert len(self.labels) == len(self.filenames)

    def __len__(self):
        return len(self.filenames) * self.clip_len // self.target_len

    def __getitem__(self, idx):
        idx = idx % len(self.filenames)
        filename = self.filenames[idx]
        audio_path = self.audio_dir.joinpath(filename)
        audio, sr = sf.read(str(audio_path), dtype=np.float32)
        assert sr == self.sample_rate

        # crop random 10 seconds piece
        labels_to_pick = self.target_len * self.label_fps
        max_offset = len(self.labels[idx]) - labels_to_pick + 1
        offset = torch.randint(max_offset, (1,)).item()
        labels = self.labels[idx][offset:offset + labels_to_pick]
        scale = self.sample_rate // self.label_fps
        audio = audio[offset * scale:offset * scale + labels_to_pick * scale]
        timestamps = self.timestamps[idx][offset:offset + labels_to_pick]
        return audio, labels.transpose(0, 1), filename, timestamps


def get_training_dataset(
        task_path,
        sample_rate=16000,
        label_fps=25,
        wavmix_p=0.0,
        random_crop=True
):
    task_path = Path(task_path)

    label_vocab, nlabels = label_vocab_nlabels(task_path)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    train_fold = task_path.joinpath("train.json")
    audio_dir = task_path.joinpath(str(sample_rate), "train")
    train_fold_data = json.load(train_fold.open())
    if random_crop:
        dataset = RandomCropDataset(train_fold_data, audio_dir, sample_rate, label_fps, label_to_idx, nlabels)
    else:
        dataset = FixCropDataset(train_fold_data, audio_dir, sample_rate, label_fps, label_to_idx, nlabels)
    if wavmix_p > 0:
        dataset = MixupDataset(dataset, rate=wavmix_p)
    return dataset


def get_validation_dataset(
        task_path,
        sample_rate=16000,
        label_fps=25,
):
    task_path = Path(task_path)

    label_vocab, nlabels = label_vocab_nlabels(task_path)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    valid_fold = task_path.joinpath("valid.json")
    audio_dir = task_path.joinpath(str(sample_rate), "valid")
    valid_fold_data = json.load(valid_fold.open())
    dataset = FixCropDataset(valid_fold_data, audio_dir, sample_rate, label_fps, label_to_idx, nlabels)
    return dataset


def get_test_dataset(
        task_path,
        sample_rate=16000,
        label_fps=25,
):
    task_path = Path(task_path)

    label_vocab, nlabels = label_vocab_nlabels(task_path)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    test_fold = task_path.joinpath("test.json")
    audio_dir = task_path.joinpath(str(sample_rate), "test")
    test_fold_data = json.load(test_fold.open())
    dataset = FixCropDataset(test_fold_data, audio_dir, sample_rate, label_fps, label_to_idx, nlabels)
    return dataset


def get_labels_for_timestamps(labels: List, timestamps: np.ndarray) -> List:
    # A list of labels present at each timestamp
    tree = IntervalTree()
    # Add all events to the label tree
    for event in labels:
        # We add 0.0001 so that the end also includes the event
        tree.addi(event["start"], event["end"] + 0.0001, event["label"])

    timestamp_labels = []
    # Update the binary vector of labels with intervals for each timestamp
    for j, t in enumerate(timestamps):
        interval_labels: List[str] = [interval.data for interval in tree[t]]
        timestamp_labels.append(interval_labels)
        # If we want to store the timestamp too
        # labels_for_sound.append([float(t), interval_labels])

    assert len(timestamp_labels) == len(timestamps)
    return timestamp_labels


def label_vocab_nlabels(task_path: Path) -> Tuple[pd.DataFrame, int]:
    label_vocab = pd.read_csv(task_path.joinpath("labelvocabulary.csv"))

    nlabels = len(label_vocab)
    assert nlabels == label_vocab["idx"].max() + 1
    return (label_vocab, nlabels)


def label_vocab_as_dict(df: pd.DataFrame, key: str, value: str) -> Dict:
    """
    Returns a dictionary of the label vocabulary mapping the label column to
    the idx column. key sets whether the label or idx is the key in the dict. The
    other column will be the value.
    """
    if key == "label":
        # Make sure the key is a string
        df["label"] = df["label"].astype(str)
        value = "idx"
    else:
        assert key == "idx", "key argument must be either 'label' or 'idx'"
        value = "label"
    return df.set_index(key).to_dict()[value]


def label_to_binary_vector(label: List, num_labels: int) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is multi-hot binary vector
    """
    # Lame special case for multilabel with no labels
    if len(label) == 0:
        # BCEWithLogitsLoss wants float not long targets
        binary_labels = torch.zeros((num_labels,), dtype=torch.float)
    else:
        binary_labels = torch.zeros((num_labels,)).scatter(0, torch.tensor(label), 1.0)

    # Validate the binary vector we just created
    assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label)
    return binary_labels


class MixupDataset(Dataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=0.2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            batch1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            batch2 = self.dataset[idx2]
            x1, x2 = batch1[0], batch2[0]
            y1, y2 = batch1[1], batch2[1]
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            y = (y1 * l + y2 * (1. - l))
            return x, y, batch1[2], batch1[3]
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
