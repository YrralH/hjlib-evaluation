'''Method-neutral LSV-HR population, entry, and matrix evaluation.'''
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hjlib_dataset_std import (
    VirtualCrowd_Accepted_Track_Span,
    VirtualCrowd_Eval_Population_Selection,
)
from hjlib_evaluation.corrected_crowd_data import Corrected_Crowd_Sequence
from hjlib_evaluation.virtualcrowd_naive_comparison import (
    VC_NAIVE_COMPARISON_PROFILE_ID,
    VirtualCrowd_Naive_Comparison_Result,
    VirtualCrowd_Naive_Comparison_Sequence_Summary,
    evaluate_virtualcrowd_naive_comparison,
    reduce_virtualcrowd_naive_comparison_summaries,
)


class LSVHR_Evaluation_Profile(StrEnum):
    '''Fixed LSV-HR metric profiles selectable by callers.'''

    NAIVE = 'naive'


def require_identity(value: str, name: str) -> str:
    '''Require one non-empty exact string identity.'''
    if type(value) is not str:
        raise TypeError('%s must be an exact str' % name)
    if not value:
        raise ValueError('%s must be non-empty' % name)
    return value


@dataclass(frozen=True, slots=True)
class LSVHR_Evaluation_Population:
    '''One dataset-std population selection projected to exact split scenes.'''

    filtering_id: str
    split_id: str
    rule_id: str
    selection: VirtualCrowd_Eval_Population_Selection
    split_scene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identity(self.filtering_id, 'filtering_id')
        require_identity(self.split_id, 'split_id')
        require_identity(self.rule_id, 'rule_id')
        if type(self.selection) is not VirtualCrowd_Eval_Population_Selection:
            raise TypeError(
                'selection must be a VirtualCrowd_Eval_Population_Selection'
            )
        for scene_id in self.split_scene_ids:
            require_identity(scene_id, 'split scene ID')
        if self.split_scene_ids != tuple(sorted(set(self.split_scene_ids))):
            raise ValueError('split_scene_ids must be sorted unique')
        if not self.split_scene_ids:
            raise ValueError('split_scene_ids must be non-empty')
        selected_scene_ids = {
            span.scene_name for span in self.spans
        }
        if selected_scene_ids != set(self.split_scene_ids):
            raise ValueError(
                'population must select every exact split scene'
            )

    @property
    def spans(self) -> tuple[VirtualCrowd_Accepted_Track_Span, ...]:
        '''Return the canonical selection restricted to exact split scenes.'''
        split_scenes = set(self.split_scene_ids)
        return tuple(
            span for span in self.selection.spans
            if span.scene_name in split_scenes
        )

    @property
    def count_span(self) -> int:
        '''Return the number of selected native-track spans.'''
        return len(self.spans)

    @property
    def count_occurrence(self) -> int:
        '''Return the exact selected person-frame count.'''
        return sum(span.count_frame for span in self.spans)


class LSVHR_Method_Loader(Protocol):
    '''Method-owned normalized-scene loader consumed by evaluation.'''

    def load_scene(self, scene_id: str) -> Corrected_Crowd_Sequence: ...


@dataclass(frozen=True, slots=True)
class LSVHR_Evaluation_Entry:
    '''One official output-entry identity bound to its method loader.'''

    entry_id: str
    loader: LSVHR_Method_Loader

    def __post_init__(self) -> None:
        require_identity(self.entry_id, 'entry_id')
        if not callable(getattr(self.loader, 'load_scene', None)):
            raise TypeError('loader must provide load_scene')


@dataclass(frozen=True, slots=True)
class LSVHR_Entry_Evaluation_Result:
    '''One reduced metric result bound to an official output entry.'''

    profile: LSVHR_Evaluation_Profile
    entry_id: str
    result: VirtualCrowd_Naive_Comparison_Result

    def __post_init__(self) -> None:
        if type(self.profile) is not LSVHR_Evaluation_Profile:
            raise TypeError('profile must be an exact LSVHR_Evaluation_Profile')
        require_identity(self.entry_id, 'entry_id')
        if type(self.result) is not VirtualCrowd_Naive_Comparison_Result:
            raise TypeError('result must be an exact naive comparison result')
        if self.profile is not LSVHR_Evaluation_Profile.NAIVE:
            raise ValueError('unsupported LSV-HR evaluation profile')
        if self.result.profile_id != VC_NAIVE_COMPARISON_PROFILE_ID:
            raise ValueError('result metric profile differs from LSV-HR profile')


def selected_gt_mask_for_lsvhr_population_scene(
        sequence: Corrected_Crowd_Sequence,
        population: LSVHR_Evaluation_Population,
    ) -> NDArray[np.bool_]:
    '''Project one exact split population onto normalized GT occurrences.'''
    if type(population) is not LSVHR_Evaluation_Population:
        raise TypeError('population must be an LSVHR_Evaluation_Population')
    spans = tuple(
        span for span in population.spans
        if span.scene_name == sequence.scene_id
    )
    if not spans:
        raise ValueError(
            'population has no spans for scene %s' % sequence.scene_id
        )
    selected_keys = {
        (frame_id, span.native_track_id)
        for span in spans
        for frame_id in range(span.start_frame, span.end_frame)
    }
    if not selected_keys:
        raise ValueError('population scene selection is empty')
    gt_keys = tuple(
        (int(frame_id), int(track_id))
        for frame_id, track_id in zip(
            sequence.gt_frame_ids.tolist(),
            sequence.gt_track_ids.tolist(),
            strict=True,
        )
    )
    if len(gt_keys) != len(set(gt_keys)):
        raise ValueError('normalized scene contains duplicate GT occurrence keys')
    gt_key_set = set(gt_keys)
    missing = selected_keys - gt_key_set
    if missing:
        raise ValueError(
            'normalized scene is missing %d selected GT keys' % len(missing)
        )
    mask = np.asarray(
        [key in selected_keys for key in gt_keys],
        dtype=np.bool_,
    )
    if int(mask.sum()) != len(selected_keys):
        raise ValueError('selected GT mask count differs from population spans')
    mask.setflags(write=False)
    return mask


def evaluate_lsvhr_virtualcrowd_entry(
        profile: LSVHR_Evaluation_Profile,
        entry: LSVHR_Evaluation_Entry,
        population: LSVHR_Evaluation_Population,
    ) -> LSVHR_Entry_Evaluation_Result:
    '''Evaluate one official entry on one exact VirtualCrowd population.'''
    if type(profile) is not LSVHR_Evaluation_Profile:
        raise TypeError('profile must be an exact LSVHR_Evaluation_Profile')
    if type(entry) is not LSVHR_Evaluation_Entry:
        raise TypeError('entry must be an exact LSVHR_Evaluation_Entry')
    if type(population) is not LSVHR_Evaluation_Population:
        raise TypeError('population must be an LSVHR_Evaluation_Population')
    summaries: list[VirtualCrowd_Naive_Comparison_Sequence_Summary] = []
    for scene_id in population.split_scene_ids:
        sequence = entry.loader.load_scene(scene_id)
        if sequence.scene_id != scene_id:
            raise ValueError('method loader returned the wrong scene identity')
        selected_mask = selected_gt_mask_for_lsvhr_population_scene(
            sequence,
            population,
        )
        if profile is LSVHR_Evaluation_Profile.NAIVE:
            summaries.append(evaluate_virtualcrowd_naive_comparison(
                sequence,
                population.filtering_id,
                population.split_id,
                selected_mask,
            ))
        else:
            raise ValueError('unsupported LSV-HR evaluation profile')
    reduced = reduce_virtualcrowd_naive_comparison_summaries(summaries)
    return LSVHR_Entry_Evaluation_Result(
        profile=profile,
        entry_id=entry.entry_id,
        result=reduced,
    )


def evaluate_lsvhr_virtualcrowd_matrix(
        profile: LSVHR_Evaluation_Profile,
        entries: Sequence[LSVHR_Evaluation_Entry],
        population: LSVHR_Evaluation_Population,
    ) -> tuple[LSVHR_Entry_Evaluation_Result, ...]:
    '''Evaluate a non-empty ordered set of unique official entries.'''
    entry_sequence = tuple(entries)
    if not entry_sequence:
        raise ValueError('LSV-HR evaluation matrix must be non-empty')
    for entry in entry_sequence:
        if type(entry) is not LSVHR_Evaluation_Entry:
            raise TypeError('entries must contain exact LSVHR_Evaluation_Entry')
    entry_ids = tuple(entry.entry_id for entry in entry_sequence)
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError('LSV-HR evaluation entry IDs must be unique')
    results = [
        evaluate_lsvhr_virtualcrowd_entry(profile, entry, population)
        for entry in entry_sequence
    ]
    return tuple(results)


__all__ = [
    'LSVHR_Entry_Evaluation_Result',
    'LSVHR_Evaluation_Entry',
    'LSVHR_Evaluation_Population',
    'LSVHR_Evaluation_Profile',
    'LSVHR_Method_Loader',
    'evaluate_lsvhr_virtualcrowd_entry',
    'evaluate_lsvhr_virtualcrowd_matrix',
    'selected_gt_mask_for_lsvhr_population_scene',
]
