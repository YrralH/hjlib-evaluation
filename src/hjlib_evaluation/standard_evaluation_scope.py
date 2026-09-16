'''Shared TRACK, VISRUN, and FRAME partitions for standard evaluation.'''
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import (
    Corrected_Crowd_Sequence,
    bool_array,
    int_array,
)


@dataclass(frozen=True, slots=True)
class Standard_Evaluation_Direct_Target_Join:
    '''Lightweight exact mapping without copying normalized metric tensors.'''

    sequence: Corrected_Crowd_Sequence
    gt_rows: NDArray[np.int64]
    prediction_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        if type(self.sequence) is not Corrected_Crowd_Sequence:
            raise TypeError('sequence must be a corrected crowd sequence')
        gt_rows = int_array(self.gt_rows, 'direct-target GT rows')
        prediction_rows = int_array(
            self.prediction_rows,
            'direct-target prediction rows',
        )
        if gt_rows.ndim != 1 or prediction_rows.shape != gt_rows.shape:
            raise ValueError('direct-target row mappings must be paired vectors')
        if len(np.unique(gt_rows)) != len(gt_rows) \
                or len(np.unique(prediction_rows)) != len(prediction_rows):
            raise ValueError('direct-target row mappings must be one-to-one')
        if np.any((gt_rows < 0) | (gt_rows >= len(self.sequence.gt_frame_ids))) \
                or np.any(
                    (prediction_rows < 0)
                    | (prediction_rows >= len(self.sequence.prediction_frame_ids))
                ):
            raise ValueError('direct-target row mapping is out of range')
        if not np.array_equal(
                self.sequence.prediction_identity_target_gt_rows[prediction_rows],
                gt_rows,
            ) or not np.array_equal(
                self.sequence.prediction_frame_ids[prediction_rows],
                self.sequence.gt_frame_ids[gt_rows],
            ):
            raise ValueError('direct-target row mapping violates identity/frame')
        object.__setattr__(self, 'gt_rows', gt_rows)
        object.__setattr__(self, 'prediction_rows', prediction_rows)


def standard_direct_target_join(
        sequence: Corrected_Crowd_Sequence,
        selected_gt_mask: NDArray[np.generic],
    ) -> Standard_Evaluation_Direct_Target_Join:
    '''Join selected GT rows without cloning the immutable sequence.'''
    if type(sequence) is not Corrected_Crowd_Sequence:
        raise TypeError('sequence must be a corrected crowd sequence')
    selected = bool_array(selected_gt_mask, 'selected_gt_mask')
    gt_count = len(sequence.gt_frame_ids)
    if selected.shape != (gt_count,):
        raise ValueError('selected_gt_mask must have shape (%d,)' % gt_count)
    gt_rows = np.flatnonzero(selected).astype(np.int64, copy=False)
    targets = sequence.prediction_identity_target_gt_rows
    mapped_prediction_rows = np.flatnonzero(targets >= 0).astype(
        np.int64,
        copy=False,
    )
    target_counts = np.bincount(
        targets[mapped_prediction_rows],
        minlength=gt_count,
    )
    if np.any(target_counts[gt_rows] != 1):
        raise ValueError(
            'every selected GT row must have exactly one direct-target prediction'
        )
    prediction_for_gt = np.full(gt_count, -1, dtype=np.int64)
    prediction_for_gt[targets[mapped_prediction_rows]] = mapped_prediction_rows
    return Standard_Evaluation_Direct_Target_Join(
        sequence,
        gt_rows,
        prediction_for_gt[gt_rows],
    )


@dataclass(frozen=True, slots=True)
class Standard_Evaluation_Scope_Partition:
    '''One immutable joined-row order and its half-open scope offsets.'''

    order: NDArray[np.int64]
    offsets: NDArray[np.int64]

    def __post_init__(self) -> None:
        order = int_array(self.order, 'scope order')
        offsets = int_array(self.offsets, 'scope offsets')
        if order.ndim != 1 or offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError('scope order/offsets must be one-dimensional')
        if offsets[0] != 0 or offsets[-1] != len(order):
            raise ValueError('scope offsets must cover the complete order')
        if np.any(np.diff(offsets) <= 0):
            raise ValueError('scope offsets must be strictly increasing')
        if len(np.unique(order)) != len(order) \
                or np.any((order < 0) | (order >= len(order))):
            raise ValueError('scope order must be one permutation')
        object.__setattr__(self, 'order', order)
        object.__setattr__(self, 'offsets', offsets)

    @property
    def count_scope(self) -> int:
        '''Return the number of non-empty scopes.'''
        return len(self.offsets) - 1


@dataclass(frozen=True, slots=True)
class Standard_Evaluation_Scope_Index:
    '''One exact join with all shared traversal partitions.'''

    join: Standard_Evaluation_Direct_Target_Join
    joined_visrun_labels: NDArray[np.int64]
    tracks: Standard_Evaluation_Scope_Partition
    visruns: Standard_Evaluation_Scope_Partition
    acceleration_runs: Standard_Evaluation_Scope_Partition
    visrun_acceleration_offsets: NDArray[np.int64]
    frames: Standard_Evaluation_Scope_Partition

    def __post_init__(self) -> None:
        if type(self.join) is not Standard_Evaluation_Direct_Target_Join:
            raise TypeError('join must be a direct-target join')
        count = len(self.join.gt_rows)
        if count == 0:
            raise ValueError('standard evaluation scope index cannot be empty')
        labels = int_array(self.joined_visrun_labels, 'joined VISRUN labels')
        if labels.shape != (count,) or np.any(labels < 0):
            raise ValueError('joined VISRUN labels must be non-negative row labels')
        for name in ('tracks', 'visruns', 'acceleration_runs', 'frames'):
            partition = getattr(self, name)
            if type(partition) is not Standard_Evaluation_Scope_Partition:
                raise TypeError('%s must be a scope partition' % name)
            if len(partition.order) != count:
                raise ValueError('%s must cover every joined row' % name)
        mapping = int_array(
            self.visrun_acceleration_offsets,
            'visrun acceleration offsets',
        )
        if mapping.shape != (self.visruns.count_scope + 1,):
            raise ValueError('visrun acceleration mapping shape differs')
        if mapping[0] != 0 \
                or mapping[-1] != self.acceleration_runs.count_scope \
                or np.any(np.diff(mapping) <= 0):
            raise ValueError('visrun acceleration mapping must partition runs')
        expected = canonical_scope_partitions(self.join, labels)
        for name, wanted in zip(
                ('tracks', 'visruns', 'acceleration_runs', 'frames'),
                (*expected[:3], expected[4]),
                strict=True,
            ):
            actual = getattr(self, name)
            if not np.array_equal(actual.order, wanted.order) \
                    or not np.array_equal(actual.offsets, wanted.offsets):
                raise ValueError('%s does not match joined row semantics' % name)
        if not np.array_equal(mapping, expected[3]):
            raise ValueError('VISRUN acceleration mapping differs from semantics')
        object.__setattr__(self, 'joined_visrun_labels', labels)
        object.__setattr__(self, 'visrun_acceleration_offsets', mapping)


def boundary_offsets(boundaries: NDArray[np.bool_]) -> NDArray[np.int64]:
    '''Convert start flags into complete half-open offsets.'''
    starts = np.flatnonzero(boundaries).astype(np.int64, copy=False)
    return np.concatenate((starts, np.array([len(boundaries)], dtype=np.int64)))


def make_partition(
        order: NDArray[np.int64],
        start_flags: NDArray[np.bool_],
    ) -> Standard_Evaluation_Scope_Partition:
    return Standard_Evaluation_Scope_Partition(
        order,
        boundary_offsets(start_flags),
    )


def canonical_scope_partitions(
        join: Standard_Evaluation_Direct_Target_Join,
        joined_labels: NDArray[np.int64],
    ) -> tuple[
        Standard_Evaluation_Scope_Partition,
        Standard_Evaluation_Scope_Partition,
        Standard_Evaluation_Scope_Partition,
        NDArray[np.int64],
        Standard_Evaluation_Scope_Partition,
    ]:
    '''Derive the unique semantic partitions for joined rows and VISRUN labels.'''
    count = len(join.gt_rows)
    joined_frames = join.sequence.gt_frame_ids[join.gt_rows]
    joined_tracks = join.sequence.gt_track_ids[join.gt_rows]
    track_order = np.asarray(
        np.lexsort((joined_frames, joined_tracks)),
        dtype=np.int64,
    )
    ordered_frames = joined_frames[track_order]
    ordered_tracks = joined_tracks[track_order]
    ordered_labels = joined_labels[track_order]

    first = np.zeros(count, dtype=np.bool_)
    first[0] = True
    track_starts = np.array(first, copy=True)
    track_starts[1:] = ordered_tracks[1:] != ordered_tracks[:-1]
    visrun_starts = np.array(first, copy=True)
    visrun_starts[1:] = ordered_labels[1:] != ordered_labels[:-1]
    label_track_pairs = set(zip(
        ordered_labels.tolist(),
        ordered_tracks.tolist(),
        strict=True,
    ))
    if len(label_track_pairs) != len(np.unique(ordered_labels)):
        raise ValueError('one base VISRUN label cannot span GT tracks')
    if int(np.count_nonzero(visrun_starts)) != len(np.unique(ordered_labels)):
        raise ValueError('base VISRUN labels must be contiguous within tracks')

    acceleration_starts = np.array(visrun_starts, copy=True)
    acceleration_starts[1:] |= ordered_frames[1:] != ordered_frames[:-1] + 1
    tracks = make_partition(track_order, track_starts)
    visruns = make_partition(track_order, visrun_starts)
    acceleration_runs = make_partition(track_order, acceleration_starts)
    acceleration_boundary_indices = np.searchsorted(
        acceleration_runs.offsets,
        visruns.offsets,
    ).astype(np.int64, copy=False)

    frame_order = np.asarray(
        np.lexsort((joined_tracks, joined_frames)),
        dtype=np.int64,
    )
    frame_values = joined_frames[frame_order]
    frame_starts = np.zeros(count, dtype=np.bool_)
    frame_starts[0] = True
    frame_starts[1:] = frame_values[1:] != frame_values[:-1]
    frames = make_partition(frame_order, frame_starts)
    return (
        tracks,
        visruns,
        acceleration_runs,
        acceleration_boundary_indices,
        frames,
    )


def build_standard_evaluation_scope_index(
        sequence: Corrected_Crowd_Sequence,
        selected_gt_mask: NDArray[np.generic],
        base_visrun_labels: NDArray[np.generic],
    ) -> Standard_Evaluation_Scope_Index:
    '''Build every traversal index once without copying metric tensors.'''
    join = standard_direct_target_join(sequence, selected_gt_mask)
    count = len(join.gt_rows)
    if count == 0:
        raise ValueError('standard evaluation selection is empty')
    labels_source = np.asarray(base_visrun_labels)
    if not np.issubdtype(labels_source.dtype, np.integer):
        raise TypeError('base_visrun_labels must have integer dtype')
    labels = np.asarray(labels_source, dtype=np.int64)
    if labels.shape != (len(sequence.gt_frame_ids),):
        raise ValueError('base_visrun_labels must align to GT rows')
    joined_labels = labels[join.gt_rows]
    if np.any(joined_labels < 0):
        raise ValueError('every selected GT row must have a base VISRUN label')

    tracks, visruns, acceleration_runs, acceleration_boundary_indices, frames = (
        canonical_scope_partitions(join, joined_labels)
    )
    return Standard_Evaluation_Scope_Index(
        join,
        joined_labels,
        tracks,
        visruns,
        acceleration_runs,
        acceleration_boundary_indices,
        frames,
    )


def standard_evaluation_scope_rows(
        partition: Standard_Evaluation_Scope_Partition,
        scope_index: int,
    ) -> NDArray[np.int64]:
    '''Return a readonly view of joined positions in one scope.'''
    if type(scope_index) is not int \
            or scope_index < 0 or scope_index >= partition.count_scope:
        raise IndexError('scope index is out of range')
    start = int(partition.offsets[scope_index])
    end = int(partition.offsets[scope_index + 1])
    return partition.order[start:end]


__all__ = [
    'Standard_Evaluation_Direct_Target_Join',
    'Standard_Evaluation_Scope_Index',
    'Standard_Evaluation_Scope_Partition',
    'build_standard_evaluation_scope_index',
    'standard_evaluation_scope_rows',
    'standard_direct_target_join',
]
