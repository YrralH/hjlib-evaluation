'''Three-scope orchestration and reduction for standard LSV-HR evaluation.'''
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import (
    Corrected_Crowd_Sequence,
    int_array,
)
from hjlib_evaluation.standard_evaluation_metrics import (
    compute_standard_acc_joint_ratio_magnitudes,
    compute_standard_acc_root_ratio_magnitudes,
    compute_standard_acc_root_values,
    compute_standard_mpjpe_world_values,
    compute_standard_oks_vis_values,
    compute_standard_pa_ppds_values,
    compute_standard_ppds_values,
    compute_standard_rt_mpjpe_values,
    compute_standard_rte_world_values,
    compute_standard_sequence_rt_mpjpe_values,
    compute_standard_sequence_t_mpjpe_values,
    compute_standard_t_mpjpe_values,
)
from hjlib_evaluation.standard_evaluation_scope import (
    Standard_Evaluation_Scope_Index,
    build_standard_evaluation_scope_index,
    standard_evaluation_scope_rows,
)


STANDARD_EVALUATION_PROFILE_ID: Final = 'LSVHR_STANDARD_METRICS_V1'
STANDARD_EVALUATION_METRICS: Final = (
    'MPJPE-WORLD',
    'T-MPJPE',
    'RT-MPJPE',
    'SEQ-T-MPJPE-VISRUN',
    'SEQ-RT-MPJPE-VISRUN',
    'SEQ-T-MPJPE-TRACK',
    'SEQ-RT-MPJPE-TRACK',
    'RTE-WORLD',
    'ACC-ROOT',
    'ACC-ROOT-RATIO',
    'ACC-JOINT-RATIO',
    'OKS-VIS',
    'PPDS',
    'PA-PPDS',
)
STANDARD_EVALUATION_METRIC_UNITS: Final = (
    'mm', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm', 'mm',
    'mm/frame^2', 'ratio', 'ratio', 'fraction', 'fraction', 'fraction',
)
STANDARD_MEAN_METRICS: Final = (
    'MPJPE-WORLD',
    'T-MPJPE',
    'RT-MPJPE',
    'SEQ-T-MPJPE-VISRUN',
    'SEQ-RT-MPJPE-VISRUN',
    'SEQ-T-MPJPE-TRACK',
    'SEQ-RT-MPJPE-TRACK',
    'RTE-WORLD',
    'ACC-ROOT',
    'OKS-VIS',
    'PPDS',
    'PA-PPDS',
)
MEAN_METRIC_INDEX: Final = {
    name: index for index, name in enumerate(STANDARD_MEAN_METRICS)
}
FINAL_METRIC_INDEX: Final = {
    name: index for index, name in enumerate(STANDARD_EVALUATION_METRICS)
}
MILLIMETRE_METRICS: Final = frozenset(STANDARD_EVALUATION_METRICS[:9])


def immutable_float_array(
        value: NDArray[np.generic],
        shape: tuple[int, ...],
        name: str,
    ) -> NDArray[np.float64]:
    '''Return one finite, non-negative, immutable float64 array.'''
    source = np.asarray(value)
    if not np.issubdtype(source.dtype, np.number) \
            or np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError('%s must have real numeric dtype' % name)
    array = np.array(source, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError('%s must have shape %s' % (name, shape))
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError('%s must be finite and non-negative' % name)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


def require_identity(value: str, name: str) -> str:
    '''Return one non-empty exact string.'''
    if type(value) is not str:
        raise TypeError('%s must be an exact str' % name)
    if not value:
        raise ValueError('%s must be non-empty' % name)
    return value


def require_nonnegative_int(value: int, name: str) -> int:
    '''Return one non-negative exact integer.'''
    if type(value) is not int:
        raise TypeError('%s must be an exact int' % name)
    if value < 0:
        raise ValueError('%s must be non-negative' % name)
    return value


@dataclass(frozen=True, slots=True)
class Standard_Evaluation_Scene_Summary:
    '''Additive sufficient statistics for one standard-evaluation scene.'''

    profile_id: str
    filtering_id: str
    split_id: str
    scene_id: str
    selected_gt_count: int
    metric_sample_sums: NDArray[np.float64]
    metric_sample_counts: NDArray[np.int64]
    acc_root_ratio_predicted_sum: float
    acc_root_ratio_reference_sum: float
    acc_root_ratio_sample_count: int
    acc_joint_ratio_predicted_sums: NDArray[np.float64]
    acc_joint_ratio_reference_sums: NDArray[np.float64]
    acc_joint_ratio_sample_count: int

    def __post_init__(self) -> None:
        if self.profile_id != STANDARD_EVALUATION_PROFILE_ID:
            raise ValueError('profile_id does not identify standard evaluation')
        for name in ('filtering_id', 'split_id', 'scene_id'):
            require_identity(cast(str, getattr(self, name)), name)
        require_nonnegative_int(self.selected_gt_count, 'selected_gt_count')
        if self.selected_gt_count <= 0:
            raise ValueError('selected_gt_count must be positive')
        sums = immutable_float_array(
            self.metric_sample_sums,
            (len(STANDARD_MEAN_METRICS),),
            'metric_sample_sums',
        )
        counts = int_array(self.metric_sample_counts, 'metric_sample_counts')
        if counts.shape != sums.shape or np.any(counts < 0):
            raise ValueError('metric_sample_counts must be non-negative mean shape')
        predicted = immutable_float_array(
            self.acc_joint_ratio_predicted_sums,
            (24,),
            'acc_joint_ratio_predicted_sums',
        )
        reference = immutable_float_array(
            self.acc_joint_ratio_reference_sums,
            (24,),
            'acc_joint_ratio_reference_sums',
        )
        root_predicted = float(self.acc_root_ratio_predicted_sum)
        root_reference = float(self.acc_root_ratio_reference_sum)
        if not math.isfinite(root_predicted) or root_predicted < 0.0 \
                or not math.isfinite(root_reference) or root_reference < 0.0:
            raise ValueError('root ratio sums must be finite and non-negative')
        require_nonnegative_int(
            self.acc_root_ratio_sample_count,
            'acc_root_ratio_sample_count',
        )
        require_nonnegative_int(
            self.acc_joint_ratio_sample_count,
            'acc_joint_ratio_sample_count',
        )
        if self.acc_root_ratio_sample_count != self.acc_joint_ratio_sample_count:
            raise ValueError('root and joint ratio support counts must match')
        if root_predicted != float(predicted[0]) \
                or root_reference != float(reference[0]):
            raise ValueError('root ratio statistics must equal joint zero')
        if self.acc_root_ratio_sample_count == 0 and (
                root_predicted != 0.0 or root_reference != 0.0
                or np.any(predicted != 0.0) or np.any(reference != 0.0)
            ):
            raise ValueError('zero ratio support requires zero ratio sums')
        if np.any((counts == 0) & (sums != 0.0)):
            raise ValueError('zero mean support requires zero metric sum')
        joint_count = 24 * self.selected_gt_count
        exact_counts = {
            'MPJPE-WORLD': joint_count,
            'T-MPJPE': joint_count,
            'RT-MPJPE': joint_count,
            'SEQ-T-MPJPE-VISRUN': joint_count,
            'SEQ-RT-MPJPE-VISRUN': joint_count,
            'SEQ-T-MPJPE-TRACK': joint_count,
            'SEQ-RT-MPJPE-TRACK': joint_count,
            'RTE-WORLD': self.selected_gt_count,
        }
        for metric_name, expected in exact_counts.items():
            if counts[MEAN_METRIC_INDEX[metric_name]] != expected:
                raise ValueError('%s support count differs' % metric_name)
        if counts[MEAN_METRIC_INDEX['OKS-VIS']] > self.selected_gt_count:
            raise ValueError('OKS-VIS support exceeds selected population')
        if counts[MEAN_METRIC_INDEX['PPDS']] \
                != counts[MEAN_METRIC_INDEX['PA-PPDS']]:
            raise ValueError('PPDS and PA-PPDS support counts must match')
        for metric_name in ('OKS-VIS', 'PPDS', 'PA-PPDS'):
            metric_index = MEAN_METRIC_INDEX[metric_name]
            if sums[metric_index] > float(counts[metric_index]):
                raise ValueError('%s sum cannot exceed support count' % metric_name)
        object.__setattr__(self, 'metric_sample_sums', sums)
        object.__setattr__(self, 'metric_sample_counts', counts)
        object.__setattr__(self, 'acc_root_ratio_predicted_sum', root_predicted)
        object.__setattr__(self, 'acc_root_ratio_reference_sum', root_reference)
        object.__setattr__(self, 'acc_joint_ratio_predicted_sums', predicted)
        object.__setattr__(self, 'acc_joint_ratio_reference_sums', reference)


@dataclass(frozen=True, slots=True)
class Standard_Evaluation_Result:
    '''One globally reduced 14-metric standard evaluation result.'''

    profile_id: str
    filtering_id: str
    split_id: str
    scene_count: int
    selected_gt_count: int
    metric_values: NDArray[np.float64]
    metric_sample_counts: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.profile_id != STANDARD_EVALUATION_PROFILE_ID:
            raise ValueError('profile_id does not identify standard evaluation')
        require_identity(self.filtering_id, 'filtering_id')
        require_identity(self.split_id, 'split_id')
        require_nonnegative_int(self.scene_count, 'scene_count')
        require_nonnegative_int(self.selected_gt_count, 'selected_gt_count')
        if self.scene_count <= 0 or self.selected_gt_count <= 0:
            raise ValueError('scene and selected GT counts must be positive')
        values = immutable_float_array(
            self.metric_values,
            (len(STANDARD_EVALUATION_METRICS),),
            'metric_values',
        )
        counts = int_array(self.metric_sample_counts, 'metric_sample_counts')
        if counts.shape != values.shape or np.any(counts <= 0):
            raise ValueError('every standard metric must have positive support')
        for name in ('OKS-VIS', 'PPDS', 'PA-PPDS'):
            if values[FINAL_METRIC_INDEX[name]] > 1.0:
                raise ValueError('%s cannot exceed one' % name)
        object.__setattr__(self, 'metric_values', values)
        object.__setattr__(self, 'metric_sample_counts', counts)

    def metric(self, name: str) -> float:
        '''Return one named metric value.'''
        try:
            return float(self.metric_values[FINAL_METRIC_INDEX[name]])
        except KeyError as err:
            raise ValueError('unknown standard metric %s' % name) from err


class _Standard_Evaluation_Accumulator:
    '''Mutable scene-local owner of heterogeneous sufficient statistics.'''

    __slots__ = (
        'sums',
        'counts',
        'ratio_predicted',
        'ratio_reference',
        'root_ratio_predicted',
        'root_ratio_reference',
        'root_ratio_count',
        'joint_ratio_count',
    )

    def __init__(self) -> None:
        self.sums = np.zeros(len(STANDARD_MEAN_METRICS), dtype=np.float64)
        self.counts = np.zeros(len(STANDARD_MEAN_METRICS), dtype=np.int64)
        self.ratio_predicted = np.zeros(24, dtype=np.float64)
        self.ratio_reference = np.zeros(24, dtype=np.float64)
        self.root_ratio_predicted = 0.0
        self.root_ratio_reference = 0.0
        self.root_ratio_count = 0
        self.joint_ratio_count = 0

    def add_values(self, metric_name: str, values: NDArray[np.generic]) -> None:
        '''Add one primitive micro population to one mean metric.'''
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise ValueError('%s values must be finite and non-negative' % metric_name)
        index = MEAN_METRIC_INDEX[metric_name]
        self.sums[index] += math.fsum(float(value) for value in array)
        self.counts[index] += int(array.size)

    def add_root_ratio_magnitudes(
            self,
            predicted: NDArray[np.float64],
            reference: NDArray[np.float64],
        ) -> None:
        '''Add one independent root-ratio population.'''
        if predicted.shape != reference.shape or predicted.ndim != 1:
            raise ValueError('root ratio populations must be paired vectors')
        if predicted.size == 0:
            return
        self.root_ratio_predicted += math.fsum(
            float(value) for value in predicted
        )
        self.root_ratio_reference += math.fsum(
            float(value) for value in reference
        )
        self.root_ratio_count += len(predicted)

    def add_joint_ratio_magnitudes(
            self,
            predicted: NDArray[np.float64],
            reference: NDArray[np.float64],
        ) -> None:
        '''Add per-joint ratio numerators and denominators.'''
        if predicted.shape != reference.shape \
                or predicted.ndim != 2 or predicted.shape[1:] != (24,):
            raise ValueError('ratio magnitude populations must have shape (T, 24)')
        if predicted.size == 0:
            return
        for joint in range(24):
            self.ratio_predicted[joint] += math.fsum(
                float(value) for value in predicted[:, joint]
            )
            self.ratio_reference[joint] += math.fsum(
                float(value) for value in reference[:, joint]
            )
        self.joint_ratio_count += len(predicted)


def add_standard_track_metrics(
        accumulator: _Standard_Evaluation_Accumulator,
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> None:
    '''Add both whole-TRACK aligned metric populations.'''
    accumulator.add_values(
        'SEQ-T-MPJPE-TRACK',
        compute_standard_sequence_t_mpjpe_values(predicted_joints, reference_joints),
    )
    accumulator.add_values(
        'SEQ-RT-MPJPE-TRACK',
        compute_standard_sequence_rt_mpjpe_values(predicted_joints, reference_joints),
    )


def add_standard_visrun_metrics(
        accumulator: _Standard_Evaluation_Accumulator,
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
        acceleration_segments: Sequence[
            tuple[NDArray[np.generic], NDArray[np.generic]]
        ],
    ) -> None:
    '''Add base-VISRUN alignment and exact-subrun acceleration metrics.'''
    accumulator.add_values(
        'SEQ-T-MPJPE-VISRUN',
        compute_standard_sequence_t_mpjpe_values(predicted_joints, reference_joints),
    )
    accumulator.add_values(
        'SEQ-RT-MPJPE-VISRUN',
        compute_standard_sequence_rt_mpjpe_values(predicted_joints, reference_joints),
    )
    for predicted_segment, reference_segment in acceleration_segments:
        accumulator.add_values(
            'ACC-ROOT',
            compute_standard_acc_root_values(predicted_segment, reference_segment),
        )
        root_predicted, root_reference = (
            compute_standard_acc_root_ratio_magnitudes(
                predicted_segment,
                reference_segment,
            )
        )
        accumulator.add_root_ratio_magnitudes(
            root_predicted,
            root_reference,
        )
        joint_predicted, joint_reference = (
            compute_standard_acc_joint_ratio_magnitudes(
                predicted_segment,
                reference_segment,
            )
        )
        accumulator.add_joint_ratio_magnitudes(
            joint_predicted,
            joint_reference,
        )


def add_standard_frame_metrics(
        accumulator: _Standard_Evaluation_Accumulator,
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
        predicted_coco17_xy_px: NDArray[np.generic],
        reference_coco17_xy_px: NDArray[np.generic],
        reference_visibility: NDArray[np.generic],
        predicted_coco17_camera_depth_m: NDArray[np.generic],
        reference_bbox_xyxy_px: NDArray[np.generic],
    ) -> None:
    '''Add occurrence and cross-person FRAME metric populations.'''
    accumulator.add_values(
        'MPJPE-WORLD',
        compute_standard_mpjpe_world_values(predicted_joints, reference_joints),
    )
    accumulator.add_values(
        'T-MPJPE',
        compute_standard_t_mpjpe_values(predicted_joints, reference_joints),
    )
    accumulator.add_values(
        'RT-MPJPE',
        compute_standard_rt_mpjpe_values(predicted_joints, reference_joints),
    )
    accumulator.add_values(
        'RTE-WORLD',
        compute_standard_rte_world_values(predicted_joints, reference_joints),
    )
    accumulator.add_values(
        'OKS-VIS',
        compute_standard_oks_vis_values(
            predicted_coco17_xy_px,
            reference_coco17_xy_px,
            reference_visibility,
            predicted_coco17_camera_depth_m,
            reference_bbox_xyxy_px,
        ),
    )
    predicted_pelvis = np.asarray(predicted_joints)[:, 0]
    reference_pelvis = np.asarray(reference_joints)[:, 0]
    accumulator.add_values(
        'PPDS',
        compute_standard_ppds_values(predicted_pelvis, reference_pelvis),
    )
    accumulator.add_values(
        'PA-PPDS',
        compute_standard_pa_ppds_values(predicted_pelvis, reference_pelvis),
    )


def evaluate_standard_evaluation_scene_indexed(
        index: Standard_Evaluation_Scope_Index,
        filtering_id: str,
        split_id: str,
    ) -> Standard_Evaluation_Scene_Summary:
    '''Evaluate one scene with one caller-built shared scope index.'''
    if type(index) is not Standard_Evaluation_Scope_Index:
        raise TypeError('index must be a standard evaluation scope index')
    validated = index.join.sequence
    accumulator = _Standard_Evaluation_Accumulator()
    join = index.join

    for scope in range(index.tracks.count_scope):
        joined_rows = standard_evaluation_scope_rows(index.tracks, scope)
        add_standard_track_metrics(
            accumulator,
            validated.prediction_joints_world_m[join.prediction_rows[joined_rows]],
            validated.gt_joints_world_m[join.gt_rows[joined_rows]],
        )

    for scope in range(index.visruns.count_scope):
        joined_rows = standard_evaluation_scope_rows(index.visruns, scope)
        segments: list[tuple[NDArray[np.generic], NDArray[np.generic]]] = []
        start = int(index.visrun_acceleration_offsets[scope])
        end = int(index.visrun_acceleration_offsets[scope + 1])
        for acceleration_scope in range(start, end):
            acceleration_rows = standard_evaluation_scope_rows(
                index.acceleration_runs,
                acceleration_scope,
            )
            segments.append((
                validated.prediction_joints_world_m[
                    join.prediction_rows[acceleration_rows]
                ],
                validated.gt_joints_world_m[join.gt_rows[acceleration_rows]],
            ))
        add_standard_visrun_metrics(
            accumulator,
            validated.prediction_joints_world_m[join.prediction_rows[joined_rows]],
            validated.gt_joints_world_m[join.gt_rows[joined_rows]],
            segments,
        )

    for scope in range(index.frames.count_scope):
        joined_rows = standard_evaluation_scope_rows(index.frames, scope)
        gt_rows = join.gt_rows[joined_rows]
        prediction_rows = join.prediction_rows[joined_rows]
        add_standard_frame_metrics(
            accumulator,
            validated.prediction_joints_world_m[prediction_rows],
            validated.gt_joints_world_m[gt_rows],
            validated.prediction_coco17_xy_px[prediction_rows],
            validated.gt_coco17_xy_px[gt_rows],
            validated.gt_visibility_native[gt_rows],
            validated.prediction_coco17_camera_depth_m[prediction_rows],
            validated.gt_bbox_xyxy_px[gt_rows],
        )

    return Standard_Evaluation_Scene_Summary(
        profile_id=STANDARD_EVALUATION_PROFILE_ID,
        filtering_id=require_identity(filtering_id, 'filtering_id'),
        split_id=require_identity(split_id, 'split_id'),
        scene_id=validated.scene_id,
        selected_gt_count=len(join.gt_rows),
        metric_sample_sums=accumulator.sums,
        metric_sample_counts=accumulator.counts,
        acc_root_ratio_predicted_sum=accumulator.root_ratio_predicted,
        acc_root_ratio_reference_sum=accumulator.root_ratio_reference,
        acc_root_ratio_sample_count=accumulator.root_ratio_count,
        acc_joint_ratio_predicted_sums=accumulator.ratio_predicted,
        acc_joint_ratio_reference_sums=accumulator.ratio_reference,
        acc_joint_ratio_sample_count=accumulator.joint_ratio_count,
    )


def evaluate_standard_evaluation_scene(
        sequence: Corrected_Crowd_Sequence,
        filtering_id: str,
        split_id: str,
        selected_gt_mask: NDArray[np.generic],
        base_visrun_labels: NDArray[np.generic],
    ) -> Standard_Evaluation_Scene_Summary:
    '''Build one shared index and evaluate one standard scene.'''
    index = build_standard_evaluation_scope_index(
        sequence,
        selected_gt_mask,
        base_visrun_labels,
    )
    return evaluate_standard_evaluation_scene_indexed(
        index,
        filtering_id,
        split_id,
    )


def reduce_standard_evaluation_summaries(
        summaries: Sequence[Standard_Evaluation_Scene_Summary],
    ) -> Standard_Evaluation_Result:
    '''Reduce sorted raw scene statistics before ratios and means.'''
    ordered = tuple(sorted(summaries, key=lambda value: value.scene_id))
    if not ordered:
        raise ValueError('standard evaluation summaries must be non-empty')
    if any(type(value) is not Standard_Evaluation_Scene_Summary for value in ordered):
        raise TypeError('summaries contain an unsupported type')
    if len({value.scene_id for value in ordered}) != len(ordered):
        raise ValueError('scene_id values must be unique')
    first = ordered[0]
    if any(
            value.profile_id != first.profile_id
            or value.filtering_id != first.filtering_id
            or value.split_id != first.split_id
            for value in ordered
        ):
        raise ValueError('scene summary profile identities differ')
    sums = np.asarray([
        math.fsum(float(value.metric_sample_sums[index]) for value in ordered)
        for index in range(len(STANDARD_MEAN_METRICS))
    ], dtype=np.float64)
    counts = np.sum(
        np.stack([value.metric_sample_counts for value in ordered]),
        axis=0,
        dtype=np.int64,
    )
    if np.any(counts <= 0):
        missing = [
            STANDARD_MEAN_METRICS[index]
            for index in np.flatnonzero(counts <= 0)
        ]
        raise ValueError('standard metric support is empty: %s' % ', '.join(missing))
    ratio_predicted = np.asarray([
        math.fsum(
            float(value.acc_joint_ratio_predicted_sums[joint])
            for value in ordered
        )
        for joint in range(24)
    ], dtype=np.float64)
    ratio_reference = np.asarray([
        math.fsum(
            float(value.acc_joint_ratio_reference_sums[joint])
            for value in ordered
        )
        for joint in range(24)
    ], dtype=np.float64)
    ratio_count = sum(value.acc_joint_ratio_sample_count for value in ordered)
    root_ratio_predicted = math.fsum(
        value.acc_root_ratio_predicted_sum for value in ordered
    )
    root_ratio_reference = math.fsum(
        value.acc_root_ratio_reference_sum for value in ordered
    )
    root_ratio_count = sum(value.acc_root_ratio_sample_count for value in ordered)
    if ratio_count <= 0:
        raise ValueError('acceleration-ratio support is empty')
    if np.any(ratio_reference <= 0.0):
        joints = ', '.join(str(value) for value in np.flatnonzero(
            ratio_reference <= 0.0,
        ))
        raise ValueError('ACC ratio reference denominator is not positive: %s' % joints)
    if root_ratio_count != ratio_count \
            or root_ratio_predicted != float(ratio_predicted[0]) \
            or root_ratio_reference != float(ratio_reference[0]):
        raise ValueError('root and joint-zero reduced ratio statistics differ')
    values = np.empty(len(STANDARD_EVALUATION_METRICS), dtype=np.float64)
    final_counts = np.empty(len(STANDARD_EVALUATION_METRICS), dtype=np.int64)
    for metric_name in STANDARD_MEAN_METRICS:
        source_index = MEAN_METRIC_INDEX[metric_name]
        final_index = FINAL_METRIC_INDEX[metric_name]
        value = sums[source_index] / int(counts[source_index])
        values[final_index] = 1000.0 * value \
            if metric_name in MILLIMETRE_METRICS else value
        final_counts[final_index] = counts[source_index]
    root_ratio_index = FINAL_METRIC_INDEX['ACC-ROOT-RATIO']
    joint_ratio_index = FINAL_METRIC_INDEX['ACC-JOINT-RATIO']
    values[root_ratio_index] = root_ratio_predicted / root_ratio_reference
    values[joint_ratio_index] = float(np.mean(ratio_predicted / ratio_reference))
    final_counts[root_ratio_index] = ratio_count
    final_counts[joint_ratio_index] = ratio_count * 24
    return Standard_Evaluation_Result(
        profile_id=STANDARD_EVALUATION_PROFILE_ID,
        filtering_id=first.filtering_id,
        split_id=first.split_id,
        scene_count=len(ordered),
        selected_gt_count=sum(value.selected_gt_count for value in ordered),
        metric_values=values,
        metric_sample_counts=final_counts,
    )


def standard_evaluation_summary_to_json(
        summary: Standard_Evaluation_Scene_Summary,
    ) -> dict[str, object]:
    '''Project one scene summary to named JSON-compatible fields.'''
    return {
        'profile_id': summary.profile_id,
        'filtering_id': summary.filtering_id,
        'split_id': summary.split_id,
        'scene_id': summary.scene_id,
        'selected_gt_count': summary.selected_gt_count,
        'metric_sample_sums': {
            name: float(summary.metric_sample_sums[index])
            for index, name in enumerate(STANDARD_MEAN_METRICS)
        },
        'metric_sample_counts': {
            name: int(summary.metric_sample_counts[index])
            for index, name in enumerate(STANDARD_MEAN_METRICS)
        },
        'acc_root_ratio_predicted_sum': summary.acc_root_ratio_predicted_sum,
        'acc_root_ratio_reference_sum': summary.acc_root_ratio_reference_sum,
        'acc_root_ratio_sample_count': summary.acc_root_ratio_sample_count,
        'acc_joint_ratio_predicted_sums': (
            summary.acc_joint_ratio_predicted_sums.tolist()
        ),
        'acc_joint_ratio_reference_sums': (
            summary.acc_joint_ratio_reference_sums.tolist()
        ),
        'acc_joint_ratio_sample_count': summary.acc_joint_ratio_sample_count,
    }


def standard_evaluation_summary_from_json(
        payload: Mapping[str, object],
    ) -> Standard_Evaluation_Scene_Summary:
    '''Parse one exact named scene-summary projection.'''
    expected = {
        'profile_id', 'filtering_id', 'split_id', 'scene_id',
        'selected_gt_count', 'metric_sample_sums', 'metric_sample_counts',
        'acc_root_ratio_predicted_sum', 'acc_root_ratio_reference_sum',
        'acc_root_ratio_sample_count', 'acc_joint_ratio_predicted_sums',
        'acc_joint_ratio_reference_sums', 'acc_joint_ratio_sample_count',
    }
    if set(payload) != expected:
        raise ValueError('standard scene summary fields differ')
    sums = cast(Mapping[str, object], payload['metric_sample_sums'])
    counts = cast(Mapping[str, object], payload['metric_sample_counts'])
    if set(sums) != set(STANDARD_MEAN_METRICS) \
            or set(counts) != set(STANDARD_MEAN_METRICS):
        raise ValueError('standard mean metric fields differ')
    return Standard_Evaluation_Scene_Summary(
        profile_id=cast(str, payload['profile_id']),
        filtering_id=cast(str, payload['filtering_id']),
        split_id=cast(str, payload['split_id']),
        scene_id=cast(str, payload['scene_id']),
        selected_gt_count=cast(int, payload['selected_gt_count']),
        metric_sample_sums=np.asarray([
            sums[name] for name in STANDARD_MEAN_METRICS
        ]),
        metric_sample_counts=np.asarray([
            counts[name] for name in STANDARD_MEAN_METRICS
        ]),
        acc_root_ratio_predicted_sum=cast(
            float,
            payload['acc_root_ratio_predicted_sum'],
        ),
        acc_root_ratio_reference_sum=cast(
            float,
            payload['acc_root_ratio_reference_sum'],
        ),
        acc_root_ratio_sample_count=cast(
            int,
            payload['acc_root_ratio_sample_count'],
        ),
        acc_joint_ratio_predicted_sums=np.asarray(
            payload['acc_joint_ratio_predicted_sums'],
        ),
        acc_joint_ratio_reference_sums=np.asarray(
            payload['acc_joint_ratio_reference_sums'],
        ),
        acc_joint_ratio_sample_count=cast(
            int,
            payload['acc_joint_ratio_sample_count'],
        ),
    )


def standard_evaluation_result_to_json(
        result: Standard_Evaluation_Result,
    ) -> dict[str, object]:
    '''Project one reduced result to named JSON-compatible fields.'''
    return {
        'profile_id': result.profile_id,
        'filtering_id': result.filtering_id,
        'split_id': result.split_id,
        'scene_count': result.scene_count,
        'selected_gt_count': result.selected_gt_count,
        'metrics': {
            name: float(result.metric_values[index])
            for index, name in enumerate(STANDARD_EVALUATION_METRICS)
        },
        'metric_sample_counts': {
            name: int(result.metric_sample_counts[index])
            for index, name in enumerate(STANDARD_EVALUATION_METRICS)
        },
    }


def standard_evaluation_scene_metrics_to_json(
        summary: Standard_Evaluation_Scene_Summary,
    ) -> dict[str, object]:
    '''Project scene-local metrics while retaining zero-support cells as null.'''
    metrics: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for metric_name in STANDARD_MEAN_METRICS:
        source_index = MEAN_METRIC_INDEX[metric_name]
        count = int(summary.metric_sample_counts[source_index])
        value = None
        if count > 0:
            reduced = float(summary.metric_sample_sums[source_index]) / count
            value = 1000.0 * reduced \
                if metric_name in MILLIMETRE_METRICS else reduced
        metrics[metric_name] = value
        counts[metric_name] = count
    root_count = summary.acc_root_ratio_sample_count
    root_reference = summary.acc_root_ratio_reference_sum
    metrics['ACC-ROOT-RATIO'] = (
        summary.acc_root_ratio_predicted_sum / root_reference
        if root_count > 0 and root_reference > 0.0 else None
    )
    counts['ACC-ROOT-RATIO'] = root_count
    joint_count = summary.acc_joint_ratio_sample_count
    joint_reference = summary.acc_joint_ratio_reference_sums
    metrics['ACC-JOINT-RATIO'] = (
        float(np.mean(
            summary.acc_joint_ratio_predicted_sums / joint_reference,
        ))
        if joint_count > 0 and np.all(joint_reference > 0.0) else None
    )
    counts['ACC-JOINT-RATIO'] = joint_count * 24
    return {
        'profile_id': summary.profile_id,
        'filtering_id': summary.filtering_id,
        'split_id': summary.split_id,
        'scene_count': 1,
        'selected_gt_count': summary.selected_gt_count,
        'metrics': {
            name: metrics[name] for name in STANDARD_EVALUATION_METRICS
        },
        'metric_sample_counts': {
            name: counts[name] for name in STANDARD_EVALUATION_METRICS
        },
    }


def standard_evaluation_result_from_json(
        payload: Mapping[str, object],
    ) -> Standard_Evaluation_Result:
    '''Parse one exact named reduced-result projection.'''
    expected = {
        'profile_id', 'filtering_id', 'split_id', 'scene_count',
        'selected_gt_count', 'metrics', 'metric_sample_counts',
    }
    if set(payload) != expected:
        raise ValueError('standard result fields differ')
    metrics = cast(Mapping[str, object], payload['metrics'])
    counts = cast(Mapping[str, object], payload['metric_sample_counts'])
    if set(metrics) != set(STANDARD_EVALUATION_METRICS) \
            or set(counts) != set(STANDARD_EVALUATION_METRICS):
        raise ValueError('standard result metric fields differ')
    return Standard_Evaluation_Result(
        profile_id=cast(str, payload['profile_id']),
        filtering_id=cast(str, payload['filtering_id']),
        split_id=cast(str, payload['split_id']),
        scene_count=cast(int, payload['scene_count']),
        selected_gt_count=cast(int, payload['selected_gt_count']),
        metric_values=np.asarray([
            metrics[name] for name in STANDARD_EVALUATION_METRICS
        ]),
        metric_sample_counts=np.asarray([
            counts[name] for name in STANDARD_EVALUATION_METRICS
        ]),
    )


__all__ = [
    'STANDARD_EVALUATION_METRICS',
    'STANDARD_EVALUATION_METRIC_UNITS',
    'STANDARD_EVALUATION_PROFILE_ID',
    'Standard_Evaluation_Result',
    'Standard_Evaluation_Scene_Summary',
    'add_standard_frame_metrics',
    'add_standard_track_metrics',
    'add_standard_visrun_metrics',
    'evaluate_standard_evaluation_scene',
    'evaluate_standard_evaluation_scene_indexed',
    'reduce_standard_evaluation_summaries',
    'standard_evaluation_result_from_json',
    'standard_evaluation_result_to_json',
    'standard_evaluation_scene_metrics_to_json',
    'standard_evaluation_summary_from_json',
    'standard_evaluation_summary_to_json',
]
