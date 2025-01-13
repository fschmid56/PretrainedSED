"""
score functions from: https://hearbenchmark.com/hear-tasks.html
"""

import json
from collections import ChainMap
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any

import more_itertools
import numpy as np
import sed_eval
import torch
from dcase_util.containers import MetaDataContainer
from scipy.ndimage import median_filter
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def validate_score_return_type(ret: Union[Tuple[Tuple[str, float], ...], float]):
    """
    Valid return types for the metric are
        - tuple(tuple(string: name of the subtype, float: the value)): This is the
            case with sed eval metrics. They can return (("f_measure", value),
            ("precision", value), ...), depending on the scores
            the metric should is supposed to return. This is set as `scores`
            attribute in the metric.
        - float: Standard metric behaviour

    The downstream prediction pipeline is able to handle these two types.
    In case of the tuple return type, the value of the first entry in the
    tuple will be used as an optimisation criterion wherever required.
    For instance, if the return is (("f_measure", value), ("precision", value)),
    the value corresponding to the f_measure will be used ( for instance in
    early stopping if this metric is the primary score for the task )
    """
    if isinstance(ret, tuple):
        assert all(
            type(s) == tuple and type(s[0]) == str and type(s[1]) == float for s in ret
        ), (
            "If the return type of the score is a tuple, all the elements "
            "in the tuple should be tuple of type (string, float)"
        )
    elif isinstance(ret, float):
        pass
    else:
        raise ValueError(
            f"Return type {type(ret)} is unexpected. Return type of "
            "the score function should either be a "
            "tuple(tuple) or float. "
        )


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove label_to_idx?
    def __init__(
            self,
            label_to_idx: Dict[str, int],
            name: Optional[str] = None,
            maximize: bool = True,
    ):
        """
        :param label_to_idx: Map from label string to integer index.
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        self.label_to_idx = label_to_idx
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, *args, **kwargs) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Calls the compute function of the metric, and after validating the output,
        returns the metric score
        """
        ret = self._compute(*args, **kwargs)
        validate_score_return_type(ret)
        return ret

    def _compute(
            self, predictions: Any, targets: Any, **kwargs
    ) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Compute the score based on the predictions and targets.
        This is a private function and the metric should be used as a functor
        by calling the `__call__` method which calls this and also validates
        the return type
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class SoundEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
            self,
            label_to_idx: Dict[str, int],
            scores: Tuple[str],
            params: Dict = None,
            name: Optional[str] = None,
            maximize: bool = True,
    ):
        """
        :param scores: Scores to use, from the list of overall SED eval scores.
            The first score in the tuple will be the primary score for this metric
        :param params: Parameters to pass to the scoring function,
                       see inheriting children for details.
        """
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = params
        assert self.score_class is not None

    def _compute(
            self, predictions: Dict, targets: Dict, **kwargs
    ) -> Tuple[Tuple[str, float], ...]:
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        # This will break in Python < 3.6 if the dict order is not
        # the insertion order I think. I'm a little worried about this line
        scores = self.score_class(
            event_label_list=list(self.label_to_idx.keys()), **self.params
        )

        for filename in predictions:
            scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # results_overall_metrics return a pretty large nested selection of scores,
        # with dicts of scores keyed on the type of scores, like f_measure, error_rate,
        # accuracy
        nested_overall_scores: Dict[
            str, Dict[str, float]
        ] = scores.results_overall_metrics()
        # Open up nested overall scores
        overall_scores: Dict[str, float] = dict(
            ChainMap(*nested_overall_scores.values())
        )
        # Return the required scores as tuples. The scores are returned in the
        # order they are passed in the `scores` argument
        return tuple([(score, overall_scores[score]) for score in self.scores])

    @staticmethod
    def sed_eval_event_container(
            x: Dict[str, List[Dict[str, Any]]]
    ) -> MetaDataContainer:
        # Reformat event list for sed_eval
        reference_events = []
        for filename, event_list in x.items():
            for event in event_list:
                reference_events.append(
                    {
                        # Convert from ms to seconds for sed_eval
                        "event_label": str(event["label"]),
                        "event_onset": event["start"] / 1000.0,
                        "event_offset": event["end"] / 1000.0,
                        "file": filename,
                    }
                )
        return MetaDataContainer(reference_events)


class EventBasedScore(SoundEventScore):
    """
    event-based scores - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.EventBasedMetrics


class SegmentBasedScore(SoundEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.SegmentBasedMetrics


def get_events_for_all_files(
        predictions: torch.Tensor,
        filenames: List[str],
        timestamps: torch.Tensor,
        idx_to_label: Dict[int, str],
        postprocessing_grid: Dict[str, List[float]],
        postprocessing: Optional[Tuple[Tuple[str, Any], ...]] = None,
) -> Dict[Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[str, float]]]]]:
    """
    Produces lists of events from a set of frame based label probabilities.
    The input prediction tensor may contain frame predictions from a set of different
    files concatenated together. file_timestamps has a list of filenames and
    timestamps for each frame in the predictions tensor.

    We split the predictions into separate tensors based on the filename and compute
    events based on those individually.

    If no postprocessing is specified (during training), we try a
    variety of ways of postprocessing the predictions into events,
    from the postprocessing_grid including median filtering and
    minimum event length.

    If postprocessing is specified (during test, chosen at the best
    validation epoch), we use this postprocessing.

    Args:
        predictions: a tensor of frame based multi-label predictions.
        filenames: a list of filenames where each entry corresponds
            to a frame in the predictions tensor.
        timestamps: a list of timestamps where each entry corresponds
            to a frame in the predictions tensor.
        idx_to_label: Index to label mapping.
        postprocessing: See above.

    Returns:
        A dictionary from filtering params to the following values:
        A dictionary of lists of events keyed on the filename slug.
        The event list is of dicts of the following format:
            {"label": str, "start": float ms, "end": float ms}
    """
    # This probably could be more efficient if we make the assumption that
    # timestamps are in sorted order. But this makes sure of it.
    assert predictions.shape[0] == len(filenames)
    assert predictions.shape[0] == len(timestamps)
    event_files: Dict[str, Dict[float, torch.Tensor]] = {}
    for i, (filename, timestamp) in enumerate(zip(filenames, timestamps)):
        slug = Path(filename).name

        # Key on the slug to be consistent with the ground truth
        if slug not in event_files:
            event_files[slug] = {}

        # Save the predictions for the file keyed on the timestamp
        event_files[slug][float(timestamp)] = predictions[i]

    # Create events for all the different files. Store all the events as a dictionary
    # with the same format as the ground truth from the luigi pipeline.
    # Ex) { slug -> [{"label" : "woof", "start": 0.0, "end": 2.32}, ...], ...}
    event_dict: Dict[
        Tuple[Tuple[str, Any], ...], Dict[str, List[Dict[str, Union[float, str]]]]
    ] = {}
    if postprocessing:
        postprocess = postprocessing
        event_dict[postprocess] = {}
        for slug, timestamp_predictions in event_files.items():
            event_dict[postprocess][slug] = create_events_from_prediction(
                timestamp_predictions, idx_to_label, **dict(postprocess)
            )
    else:
        postprocessing_confs = list(ParameterGrid(postprocessing_grid))
        for postprocess_dict in tqdm(postprocessing_confs):
            postprocess = tuple(postprocess_dict.items())
            event_dict[postprocess] = {}
            for slug, timestamp_predictions in event_files.items():
                event_dict[postprocess][slug] = create_events_from_prediction(
                    timestamp_predictions, idx_to_label, **postprocess_dict
                )

    return event_dict


def create_events_from_prediction(
        prediction_dict: Dict[float, torch.Tensor],
        idx_to_label: Dict[int, str],
        threshold: float = 0.5,
        median_filter_ms: float = 150,
        min_duration: float = 60.0,
) -> List[Dict[str, Union[float, str]]]:
    """
    Takes a set of prediction tensors keyed on timestamps and generates events.
    (This is for one particular audio scene.)
    We convert the prediction tensor to a binary label based on the threshold value. Any
    events occurring at adjacent timestamps are considered to be part of the same event.
    This loops through and creates events for each label class.
    We optionally apply median filtering to predictions.
    We disregard events that are less than the min_duration milliseconds.

    Args:
        prediction_dict: A dictionary of predictions keyed on timestamp
            {timestamp -> prediction}. The prediction is a tensor of label
            probabilities.
        idx_to_label: Index to label mapping.
        threshold: Threshold for determining whether to apply a label
        min_duration: the minimum duration in milliseconds for an
                event to be included.

    Returns:
        A list of dicts withs keys "label", "start", and "end"
    """
    # Make sure the timestamps are in the correct order
    timestamps = np.array(sorted(prediction_dict.keys()))

    # Create a sorted numpy matrix of frame level predictions for this file. We convert
    # to a numpy array here before applying a median filter.
    predictions = np.stack(
        [prediction_dict[t].detach().cpu().numpy() for t in timestamps]
    )

    # Optionally apply a median filter here to smooth out events.
    ts_diff = np.mean(np.diff(timestamps))
    if median_filter_ms:
        filter_width = int(round(median_filter_ms / ts_diff))
        if filter_width:
            predictions = median_filter(predictions, size=(filter_width, 1))

    # Convert probabilities to binary vectors based on threshold
    predictions = (predictions > threshold).astype(np.int8)

    events = []
    for label in range(predictions.shape[1]):
        for group in more_itertools.consecutive_groups(
                np.where(predictions[:, label])[0]
        ):
            grouptuple = tuple(group)
            assert (
                    tuple(sorted(grouptuple)) == grouptuple
            ), f"{sorted(grouptuple)} != {grouptuple}"
            startidx, endidx = (grouptuple[0], grouptuple[-1])

            start = timestamps[startidx]
            end = timestamps[endidx]
            # Add event if greater than the minimum duration threshold
            if end - start >= min_duration:
                events.append(
                    {"label": idx_to_label[label], "start": start, "end": end}
                )

    # This is just for pretty output, not really necessary
    events.sort(key=lambda k: k["start"])
    return events


def combine_target_events(split_names: List[str], task_path):
    """
    This combines the target events from the list of splits and
    returns the combined target events. This is useful when combining
    multiple folds of data to create the training or validation
    dataloader. For example, in k-fold, the training data-loader
    might be made from the first 4/5 folds, and calling this function
    with [fold00, fold01, fold02, fold03] will return the
    aggregated target events across all the folds
    """
    combined_target_events: Dict = {}
    for split_name in split_names:
        target_events = json.load(
            task_path.joinpath(f"{split_name}.json").open()
        )
        common_keys = set(combined_target_events.keys()).intersection(
            target_events.keys()
        )
        assert len(common_keys) == 0, (
            "Target events from one split should not override "
            "target events from another. This is very unlikely as the "
            "target_event is keyed on the files which are distinct for "
            "each split"
        )
        combined_target_events.update(target_events)
    return combined_target_events
