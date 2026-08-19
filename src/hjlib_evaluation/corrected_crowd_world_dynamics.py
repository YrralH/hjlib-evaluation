'''Additive GT-relative world dynamics for corrected crowd evaluation.'''
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import (
    CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
    Corrected_Crowd_Selected_View_Sequence_Summary,
    Corrected_Crowd_Sequence,
    Float_Array,
    Int_Array,
    JSON_Object,
    bool_array,
    float_array,
    int_array,
    json_int_array,
    require_shape,
    validate_exact_consecutive_window_support,
    validate_corrected_crowd_selected_view_name,
    validate_corrected_crowd_sequence,
)
from hjlib_evaluation.corrected_crowd_protocol import (
    build_visrun_labels,
    evaluate_corrected_crowd_matched_rows,
    iter_exact_matched_segments,
)
from hjlib_evaluation.joint_acceleration import compute_joint_acceleration_errors
from hjlib_evaluation.joint_jerk import compute_joint_jerk_errors


CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION = 1
CORRECTED_CROWD_WORLD_DYNAMICS_METRICS = (
    'ACC-JOINT',
    'ACC-ROOT',
    'JERK-JOINT',
    'JERK-ROOT',
)
CORRECTED_CROWD_WORLD_DYNAMICS_UNITS = (
    'mm/frame^2',
    'mm/frame^2',
    'mm/frame^3',
    'mm/frame^3',
)


def require_nonnegative_exact_int(value: object, name: str) -> int:
    '''Validate and return one non-negative exact integer.'''
    if type(value) is not int or value < 0:
        raise ValueError('%s must be a non-negative exact int' % name)
    return value


def expected_metric_counts(triples: int, quadruples: int) -> NDArray[np.int64]:
    '''Return the four population sizes implied by exact windows.'''
    return np.asarray(
        [24 * triples, triples, 24 * quadruples, quadruples],
        dtype=np.int64,
    )


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_World_Dynamics_Sequence_Summary:
    '''One scene's additive world-dynamics sufficient statistics.'''

    schema_version: int
    scene_id: str
    view_name: str
    selected_gt_count: int
    matched_selected_count: int
    metric_sample_sums: Float_Array
    metric_sample_counts: Int_Array
    accel_exact_consecutive_triple_count: int
    jerk_exact_consecutive_quadruple_count: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION:
            raise ValueError('unsupported world dynamics schema version')
        if type(self.scene_id) is not str or not self.scene_id:
            raise ValueError('scene_id must be non-empty')
        validate_corrected_crowd_selected_view_name(self.view_name)
        selected = require_nonnegative_exact_int(
            self.selected_gt_count,
            'selected_gt_count',
        )
        matched = require_nonnegative_exact_int(
            self.matched_selected_count,
            'matched_selected_count',
        )
        if matched > selected:
            raise ValueError('matched_selected_count cannot exceed selected_gt_count')
        triples = require_nonnegative_exact_int(
            self.accel_exact_consecutive_triple_count,
            'accel_exact_consecutive_triple_count',
        )
        quadruples = require_nonnegative_exact_int(
            self.jerk_exact_consecutive_quadruple_count,
            'jerk_exact_consecutive_quadruple_count',
        )
        validate_exact_consecutive_window_support(matched, triples, quadruples)
        sums = float_array(self.metric_sample_sums, 'metric_sample_sums')
        counts = int_array(self.metric_sample_counts, 'metric_sample_counts')
        shape = (len(CORRECTED_CROWD_WORLD_DYNAMICS_METRICS),)
        require_shape(sums, shape, 'metric_sample_sums')
        require_shape(counts, shape, 'metric_sample_counts')
        if not np.array_equal(counts, expected_metric_counts(triples, quadruples)):
            raise ValueError('metric counts differ from exact-window support')
        if np.any(sums < 0.0):
            raise ValueError('metric sample sums must be non-negative')
        if np.any((counts == 0) & (sums != 0.0)):
            raise ValueError('empty metric populations must have zero sum')
        object.__setattr__(self, 'metric_sample_sums', sums)
        object.__setattr__(self, 'metric_sample_counts', counts)


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_World_Dynamics_Result:
    '''Reduced additive world-dynamics result in display units.'''

    schema_version: int
    view_name: str
    selected_gt_count: int
    matched_selected_count: int
    metric_values: tuple[float | None, ...]
    accel_exact_consecutive_triple_count: int
    jerk_exact_consecutive_quadruple_count: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION:
            raise ValueError('unsupported world dynamics schema version')
        validate_corrected_crowd_selected_view_name(self.view_name)
        selected = require_nonnegative_exact_int(
            self.selected_gt_count,
            'selected_gt_count',
        )
        matched = require_nonnegative_exact_int(
            self.matched_selected_count,
            'matched_selected_count',
        )
        if matched > selected:
            raise ValueError('matched_selected_count cannot exceed selected_gt_count')
        triples = require_nonnegative_exact_int(
            self.accel_exact_consecutive_triple_count,
            'accel_exact_consecutive_triple_count',
        )
        quadruples = require_nonnegative_exact_int(
            self.jerk_exact_consecutive_quadruple_count,
            'jerk_exact_consecutive_quadruple_count',
        )
        validate_exact_consecutive_window_support(matched, triples, quadruples)
        if len(self.metric_values) != len(CORRECTED_CROWD_WORLD_DYNAMICS_METRICS):
            raise ValueError('metric_values has wrong metric count')
        for value in self.metric_values:
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError('metric values must be non-negative finite or None')
        expected_none = (triples == 0, triples == 0, quadruples == 0, quadruples == 0)
        if tuple(value is None for value in self.metric_values) != expected_none:
            raise ValueError('metric availability differs from exact-window support')


def prepare_selected_rows(
    data: Corrected_Crowd_Sequence,
    view_name: str,
    selected_gt_mask: NDArray[np.generic],
) -> tuple[str, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], int]:
    '''Validate one selected population and return its matched row join.'''
    validated_name = validate_corrected_crowd_selected_view_name(view_name)
    selected_mask = bool_array(selected_gt_mask, 'selected_gt_mask')
    if selected_mask.shape != (len(data.gt_frame_ids),):
        raise ValueError('selected_gt_mask must have shape [G]')
    base_visible = np.any(data.gt_visibility_native > 0.0, axis=1)
    if np.any(selected_mask & ~base_visible):
        raise ValueError('selected_gt_mask must be a subset of GT-visible rows')
    matched_selected = selected_mask[data.matched_gt_rows]
    return (
        validated_name,
        data.matched_gt_rows[matched_selected],
        data.matched_prediction_rows[matched_selected],
        build_visrun_labels(data),
        int(np.count_nonzero(selected_mask)),
    )


def collect_world_dynamics_errors(
    data: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    visrun_labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Collect temporal-major acceleration and jerk joint residuals.'''
    acceleration: list[NDArray[np.float64]] = []
    jerk: list[NDArray[np.float64]] = []
    for gt_rows, pred_rows in iter_exact_matched_segments(
        data,
        match_gt,
        match_pred,
        visrun_labels,
    ):
        predicted = data.prediction_joints_world_m[pred_rows]
        reference = data.gt_joints_world_m[gt_rows]
        if len(gt_rows) >= 3:
            acceleration.append(compute_joint_acceleration_errors(predicted, reference))
        if len(gt_rows) >= 4:
            jerk.append(compute_joint_jerk_errors(predicted, reference))
    acceleration_array = (
        np.concatenate(acceleration, axis=0)
        if acceleration
        else np.empty((0, 24), dtype=np.float64)
    )
    jerk_array = (
        np.concatenate(jerk, axis=0)
        if jerk
        else np.empty((0, 24), dtype=np.float64)
    )
    return acceleration_array, jerk_array


def make_world_dynamics_summary(
    data: Corrected_Crowd_Sequence,
    view_name: str,
    selected_count: int,
    match_gt: NDArray[np.int64],
    acceleration: NDArray[np.float64],
    jerk: NDArray[np.float64],
) -> Corrected_Crowd_World_Dynamics_Sequence_Summary:
    '''Build one immutable summary from exact temporal populations.'''
    populations = (
        acceleration.reshape(-1),
        acceleration[:, 0],
        jerk.reshape(-1),
        jerk[:, 0],
    )
    sums = np.asarray([
        math.fsum(float(value) for value in population)
        for population in populations
    ], dtype=np.float64)
    counts = np.asarray([len(population) for population in populations], dtype=np.int64)
    return Corrected_Crowd_World_Dynamics_Sequence_Summary(
        schema_version=CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION,
        scene_id=data.scene_id,
        view_name=view_name,
        selected_gt_count=selected_count,
        matched_selected_count=len(match_gt),
        metric_sample_sums=sums,
        metric_sample_counts=counts,
        accel_exact_consecutive_triple_count=len(acceleration),
        jerk_exact_consecutive_quadruple_count=len(jerk),
    )


def evaluate_corrected_crowd_world_dynamics(
    sequence: Corrected_Crowd_Sequence,
    view_name: str,
    selected_gt_mask: NDArray[np.generic],
) -> Corrected_Crowd_World_Dynamics_Sequence_Summary:
    '''Evaluate one selected population into world-dynamics statistics.'''
    data = validate_corrected_crowd_sequence(sequence)
    name, match_gt, match_pred, labels, selected_count = prepare_selected_rows(
        data,
        view_name,
        selected_gt_mask,
    )
    acceleration, jerk = collect_world_dynamics_errors(
        data,
        match_gt,
        match_pred,
        labels,
    )
    return make_world_dynamics_summary(
        data,
        name,
        selected_count,
        match_gt,
        acceleration,
        jerk,
    )


def evaluate_corrected_crowd_selected_view_and_world_dynamics(
    sequence: Corrected_Crowd_Sequence,
    view_name: str,
    selected_gt_mask: NDArray[np.generic],
) -> tuple[
    Corrected_Crowd_Selected_View_Sequence_Summary,
    Corrected_Crowd_World_Dynamics_Sequence_Summary,
]:
    '''Validate once and evaluate legacy selected-view plus world dynamics.'''
    data = validate_corrected_crowd_sequence(sequence)
    name, match_gt, match_pred, labels, selected_count = prepare_selected_rows(
        data,
        view_name,
        selected_gt_mask,
    )
    acceleration, jerk = collect_world_dynamics_errors(
        data,
        match_gt,
        match_pred,
        labels,
    )
    sums, counts, triples = evaluate_corrected_crowd_matched_rows(
        data,
        match_gt,
        match_pred,
        labels,
        (acceleration, len(acceleration)),
    )
    legacy = Corrected_Crowd_Selected_View_Sequence_Summary(
        schema_version=CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
        scene_id=data.scene_id,
        view_name=name,
        selected_gt_count=selected_count,
        matched_selected_count=len(match_gt),
        metric_sample_sums=sums,
        metric_sample_counts=counts,
        accel_exact_consecutive_triple_count=triples,
    )
    dynamics = make_world_dynamics_summary(
        data,
        name,
        selected_count,
        match_gt,
        acceleration,
        jerk,
    )
    return legacy, dynamics


def reduce_corrected_crowd_world_dynamics_summaries(
    summaries: Sequence[Corrected_Crowd_World_Dynamics_Sequence_Summary],
) -> Corrected_Crowd_World_Dynamics_Result:
    '''Reduce scene dynamics statistics in lexical scene order.'''
    ordered = tuple(sorted(summaries, key=lambda item: item.scene_id))
    if not ordered:
        raise ValueError('world dynamics summary collection is empty')
    if len({item.scene_id for item in ordered}) != len(ordered):
        raise ValueError('world dynamics scene IDs must be unique')
    view_name = ordered[0].view_name
    if any(item.view_name != view_name for item in ordered):
        raise ValueError('world dynamics view names must match')
    counts = np.sum(
        np.stack([item.metric_sample_counts for item in ordered]),
        axis=0,
        dtype=np.int64,
    )
    sums = np.asarray([
        math.fsum(float(item.metric_sample_sums[index]) for item in ordered)
        for index in range(len(CORRECTED_CROWD_WORLD_DYNAMICS_METRICS))
    ], dtype=np.float64)
    values = tuple(
        None if int(counts[index]) == 0 else float(sums[index] / counts[index] * 1000.0)
        for index in range(len(CORRECTED_CROWD_WORLD_DYNAMICS_METRICS))
    )
    return Corrected_Crowd_World_Dynamics_Result(
        schema_version=CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION,
        view_name=view_name,
        selected_gt_count=sum(item.selected_gt_count for item in ordered),
        matched_selected_count=sum(item.matched_selected_count for item in ordered),
        metric_values=values,
        accel_exact_consecutive_triple_count=sum(
            item.accel_exact_consecutive_triple_count for item in ordered
        ),
        jerk_exact_consecutive_quadruple_count=sum(
            item.jerk_exact_consecutive_quadruple_count for item in ordered
        ),
    )


def corrected_crowd_world_dynamics_summary_to_json(
    summary: Corrected_Crowd_World_Dynamics_Sequence_Summary,
) -> JSON_Object:
    '''Serialize one exact world-dynamics scene summary.'''
    return {
        'schema_version': summary.schema_version,
        'scene_id': summary.scene_id,
        'view_name': summary.view_name,
        'metrics': list(CORRECTED_CROWD_WORLD_DYNAMICS_METRICS),
        'metric_units': list(CORRECTED_CROWD_WORLD_DYNAMICS_UNITS),
        'selected_gt_count': summary.selected_gt_count,
        'matched_selected_count': summary.matched_selected_count,
        'metric_sample_sums': summary.metric_sample_sums.tolist(),
        'metric_sample_counts': summary.metric_sample_counts.tolist(),
        'accel_exact_consecutive_triple_count': (
            summary.accel_exact_consecutive_triple_count
        ),
        'jerk_exact_consecutive_quadruple_count': (
            summary.jerk_exact_consecutive_quadruple_count
        ),
    }


def corrected_crowd_world_dynamics_summary_from_json(
    value: Mapping[str, Any],
) -> Corrected_Crowd_World_Dynamics_Sequence_Summary:
    '''Parse one exact world-dynamics scene summary.'''
    expected = {
        'schema_version', 'scene_id', 'view_name', 'metrics', 'metric_units',
        'selected_gt_count', 'matched_selected_count', 'metric_sample_sums',
        'metric_sample_counts', 'accel_exact_consecutive_triple_count',
        'jerk_exact_consecutive_quadruple_count',
    }
    if set(value) != expected:
        raise ValueError('world dynamics summary fields do not match schema')
    if tuple(value['metrics']) != CORRECTED_CROWD_WORLD_DYNAMICS_METRICS:
        raise ValueError('world dynamics metric order does not match schema')
    if tuple(value['metric_units']) != CORRECTED_CROWD_WORLD_DYNAMICS_UNITS:
        raise ValueError('world dynamics metric units do not match schema')
    return Corrected_Crowd_World_Dynamics_Sequence_Summary(
        schema_version=value['schema_version'],
        scene_id=value['scene_id'],
        view_name=value['view_name'],
        selected_gt_count=value['selected_gt_count'],
        matched_selected_count=value['matched_selected_count'],
        metric_sample_sums=np.asarray(value['metric_sample_sums'], dtype=np.float64),
        metric_sample_counts=json_int_array(
            value['metric_sample_counts'],
            'metric_sample_counts',
        ),
        accel_exact_consecutive_triple_count=value[
            'accel_exact_consecutive_triple_count'
        ],
        jerk_exact_consecutive_quadruple_count=value[
            'jerk_exact_consecutive_quadruple_count'
        ],
    )


def corrected_crowd_world_dynamics_result_to_json(
    result: Corrected_Crowd_World_Dynamics_Result,
) -> JSON_Object:
    '''Serialize one reduced world-dynamics result.'''
    return {
        'schema_version': result.schema_version,
        'view_name': result.view_name,
        'metrics': list(CORRECTED_CROWD_WORLD_DYNAMICS_METRICS),
        'metric_units': list(CORRECTED_CROWD_WORLD_DYNAMICS_UNITS),
        'selected_gt_count': result.selected_gt_count,
        'matched_selected_count': result.matched_selected_count,
        'metric_values': list(result.metric_values),
        'accel_exact_consecutive_triple_count': (
            result.accel_exact_consecutive_triple_count
        ),
        'jerk_exact_consecutive_quadruple_count': (
            result.jerk_exact_consecutive_quadruple_count
        ),
    }


def corrected_crowd_world_dynamics_result_from_json(
    value: Mapping[str, Any],
) -> Corrected_Crowd_World_Dynamics_Result:
    '''Parse one reduced world-dynamics result.'''
    expected = {
        'schema_version', 'view_name', 'metrics', 'metric_units',
        'selected_gt_count', 'matched_selected_count', 'metric_values',
        'accel_exact_consecutive_triple_count',
        'jerk_exact_consecutive_quadruple_count',
    }
    if set(value) != expected:
        raise ValueError('world dynamics result fields do not match schema')
    if tuple(value['metrics']) != CORRECTED_CROWD_WORLD_DYNAMICS_METRICS:
        raise ValueError('world dynamics metric order does not match schema')
    if tuple(value['metric_units']) != CORRECTED_CROWD_WORLD_DYNAMICS_UNITS:
        raise ValueError('world dynamics metric units do not match schema')
    raw_values = value['metric_values']
    if not isinstance(raw_values, list):
        raise TypeError('metric_values must be a JSON array')
    return Corrected_Crowd_World_Dynamics_Result(
        schema_version=value['schema_version'],
        view_name=value['view_name'],
        selected_gt_count=value['selected_gt_count'],
        matched_selected_count=value['matched_selected_count'],
        metric_values=tuple(
            None if item is None else float(item)
            for item in cast(list[Any], raw_values)
        ),
        accel_exact_consecutive_triple_count=value[
            'accel_exact_consecutive_triple_count'
        ],
        jerk_exact_consecutive_quadruple_count=value[
            'jerk_exact_consecutive_quadruple_count'
        ],
    )


__all__ = [
    'CORRECTED_CROWD_WORLD_DYNAMICS_METRICS',
    'CORRECTED_CROWD_WORLD_DYNAMICS_SCHEMA_VERSION',
    'CORRECTED_CROWD_WORLD_DYNAMICS_UNITS',
    'Corrected_Crowd_World_Dynamics_Result',
    'Corrected_Crowd_World_Dynamics_Sequence_Summary',
    'corrected_crowd_world_dynamics_result_from_json',
    'corrected_crowd_world_dynamics_result_to_json',
    'corrected_crowd_world_dynamics_summary_from_json',
    'corrected_crowd_world_dynamics_summary_to_json',
    'evaluate_corrected_crowd_selected_view_and_world_dynamics',
    'evaluate_corrected_crowd_world_dynamics',
    'reduce_corrected_crowd_world_dynamics_summaries',
]
