import os
from time import perf_counter
import datasets
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset as TorchDataset,
    DistributedSampler,
    WeightedRandomSampler,
)

from data_util.audioset import DistributedSamplerWrapper
from data_util.audioset_classes import as_strong_train_classes
from data_util.transforms import (
    Mp3DecodeTransform,
    SequentialTransform,
    AddPseudoLabelsTransform,
    strong_label_transform,
    target_transform
)

logger = datasets.logging.get_logger(__name__)


def init_hf_config(max_shard_size="2GB", verbose=True, in_mem_max=None):
    datasets.config.MAX_SHARD_SIZE = max_shard_size
    if verbose:
        datasets.logging.set_verbosity_info()
    if in_mem_max is not None:
        datasets.config.IN_MEMORY_MAX_SIZE = in_mem_max


def get_hf_local_path(path, local_datasets_path=None):
    if local_datasets_path is None:
        local_datasets_path = os.environ.get(
            "HF_DATASETS_LOCAL",
            os.path.join(os.environ.get("HF_DATASETS_CACHE"), "../local"),
        )
    path = os.path.join(local_datasets_path, path)
    return path


class catchtime:
    # context to measure loading time: https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
    def __init__(self, debug_print="Time", logger=logger):
        self.debug_print = debug_print
        self.logger = logger

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        readout = f"{self.debug_print}: {self.time:.3f} seconds"
        self.logger.info(readout)


def merge_overlapping_events(sample):
    events = pd.DataFrame(sample['events'][0])
    events = events.sort_values(by='onset')
    sample['events'] = [None]

    for l in events['event_label'].unique():
        rows = []
        for i, r in events.loc[events['event_label'] == l].iterrows():
            if len(rows) == 0 or rows[-1]['offset'] < r['onset']:
                rows.append(r)
            else:
                onset = min(rows[-1]['onset'], r['onset'])
                offset = max(rows[-1]['offset'], r['offset'])
                rows[-1]['onset'] = onset
                rows[-1]['offset'] = offset
        if sample["events"][0] is None:
            sample['events'][0] = pd.DataFrame(rows)
        else:
            sample["events"][0] = pd.concat([sample['events'][0], pd.DataFrame(rows)])
    return sample


def get_training_dataset(
        label_encoder,
        audio_length=10.0,
        sample_rate=16000,
        wavmix_p=0.0,
        pseudo_labels_folder="/share/hel/datasets/as_strong/predictions",
        pseudo_labels_name="final"
):
    init_hf_config()

    decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    )

    ds_list = []

    with catchtime("Loading audioset_strong"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset_strong"))

    # label encode transformation
    if label_encoder is not None:
        # set list of label names to be encoded
        label_encoder.labels = as_strong_train_classes
        encode_label_fun = lambda x: strong_label_transform(x, strong_label_encoder=label_encoder)
    else:
        encode_label_fun = lambda x: x

    add_pseudo_label_transform = AddPseudoLabelsTransform(pseudo_labels_folder=pseudo_labels_folder,
                                                          pseudo_labels_name=pseudo_labels_name).add_pseudo_label_transform

    as_transforms = [
        decode_transform,
        merge_overlapping_events,
        encode_label_fun,
        target_transform,
        add_pseudo_label_transform
    ]

    as_ds.set_transform(SequentialTransform(as_transforms))

    ds_list.append(as_ds["balanced_train"])
    ds_list.append(as_ds["unbalanced_train"])
    dataset = torch.utils.data.ConcatDataset(ds_list)

    if wavmix_p > 0:
        print("Using Wavmix!")
        dataset = MixupDataset(dataset, rate=wavmix_p)
    return dataset


def get_validation_dataset(
        label_encoder,
        audio_length=10.0,
        sample_rate=16000
):
    init_hf_config()
    ds_list = []

    decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    )

    with catchtime(f"Loading audioset:"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset_strong"))

    # label encode transformation
    if label_encoder is not None:
        label_encoder.labels = as_strong_train_classes
        encode_label_fun = lambda x: strong_label_transform(x, strong_label_encoder=label_encoder)
    else:
        encode_label_fun = lambda x: x

    as_transforms = [
        decode_transform,
        merge_overlapping_events,
        encode_label_fun,
        target_transform
    ]
    as_ds.set_transform(SequentialTransform(as_transforms))
    as_ds_eval = (
        as_ds["eval"]
    )
    ds_list.append(as_ds_eval)
    dataset = torch.utils.data.ConcatDataset(ds_list)
    return dataset


def get_full_dataset(label_encoder, audio_length=10.0, sample_rate=16000):
    init_hf_config()

    decode_transform = Mp3DecodeTransform(
        sample_rate=sample_rate, max_length=audio_length, debug_info_key="filename"
    )

    with catchtime(f"Loading audioset:"):
        as_ds = datasets.load_from_disk(get_hf_local_path("audioset_strong"))

    # label encode transformation
    if label_encoder is not None:
        label_encoder.labels = as_strong_train_classes
        encode_label_fun = lambda x: strong_label_transform(x, strong_label_encoder=label_encoder)
    else:
        encode_label_fun = lambda x: x

    as_transforms = [
        decode_transform,
        merge_overlapping_events,
        encode_label_fun,
    ]

    as_ds.set_transform(SequentialTransform(as_transforms))
    ds_list = []
    ds_list.append(as_ds["balanced_train"])
    ds_list.append(as_ds["unbalanced_train"])
    ds_list.append(as_ds["eval"])

    dataset = torch.utils.data.ConcatDataset(ds_list)
    return dataset


def get_uniform_sample_weights(dataset):
    """
    :return: float tensor of shape len(full_training_set) representing the weights of each sample.
    """
    return torch.ones(len(dataset)).float()


def get_temporal_count_balanced_sample_weights(dataset, sample_weight_offset=100,
                                               save_folder="/share/rk8/shared/as_strong"):
    """
    :return: float tensor of shape len(full_training_set) representing the weights of each sample.
    """
    # the order of balanced_train_hdf5, unbalanced_train_hdf5 is important.
    # should match get_full_training_set
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, f"weights_temporal_count_offset_{sample_weight_offset}.pt")
    if os.path.exists(save_file):
        return torch.load(save_file)

    from tqdm import tqdm

    all_y = []
    for sample in tqdm(dataset, desc="Calculating sample weights."):
        all_y.append(sample["event_count"])
    all_y = torch.from_numpy(np.stack(all_y, axis=0))
    per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class

    per_class = sample_weight_offset + per_class  # offset low freq classes
    if sample_weight_offset > 0:
        print(f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}")
    per_class_weights = 1000. / per_class
    all_weight = all_y * per_class_weights
    all_weight = all_weight.sum(dim=1)

    torch.save(all_weight, save_file)
    return all_weight


class MixupDataset(TorchDataset):
    """ Mixing Up wave forms
    """

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            batch1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            batch2 = self.dataset[idx2]
            x1, x2 = batch1['audio'], batch2['audio']
            y1, y2 = batch1['strong'], batch2['strong']
            p1, p2 = batch1['pseudo_strong'], batch2['pseudo_strong']
            l = np.random.beta(self.beta, self.beta)
            l = max(l, 1. - l)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * l + x2 * (1. - l))
            x = x - x.mean()
            batch1['audio'] = x
            batch1['strong'] = (y1 * l + y2 * (1. - l))
            batch1['pseudo_strong'] = (p1 * l + p2 * (1. - l))
            return batch1
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
        self, sampler, dataset, num_replicas=None, rank=None, shuffle: bool = True
    ):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch < 2:
            logger.info(
                f"\n DistributedSamplerWrapper (rank {self.rank}) :  {indices[:3]} \n\n"
            )
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


def get_weighted_sampler(
        samples_weights,
        epoch_len=100_000,
        sampler_replace=False,
):
    num_nodes = int(os.environ.get("WORLD_SIZE", 1))
    ddp = int(os.environ.get("DDP", 1))
    num_nodes = max(ddp, num_nodes)
    rank = int(os.environ.get("NODE_RANK", 0))
    return DistributedSamplerWrapper(
        sampler=WeightedRandomSampler(
            samples_weights, num_samples=epoch_len, replacement=sampler_replace
        ),
        dataset=range(epoch_len),
        num_replicas=num_nodes,
        rank=rank,
    )


if __name__ == "__main__":
    from helpers.encode import ManyHotEncoder

    encoder = ManyHotEncoder([], 10., 160, net_pooling=4, fs=16_000)

    train_ds = get_training_dataset(
        encoder, audio_length=10.0, sample_rate=16_000
    )

    valid_ds = get_validation_dataset(
        encoder, audio_length=10.0, sample_rate=16_000
    )

    print("Len train dataset: ", len(train_ds))
    print("Len valid dataset: ", len(valid_ds))
