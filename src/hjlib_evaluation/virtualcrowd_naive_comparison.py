'''Provisional NAIVE comparison for selected VirtualCrowd populations.'''
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import (
    Corrected_Crowd_Sequence,
    bool_array,
    validate_corrected_crowd_sequence,
)
from hjlib_evaluation.corrected_crowd_protocol import COCO17_SIGMAS
from hjlib_evaluation.joint_acceleration import (
    compute_central_acceleration_magnitudes,
)
from hjlib_evaluation.joint_error import (
    compute_joint_position_errors,
    compute_pa_joint_position_errors,
)
from hjlib_evaluation.keypoint_oks import (
    compute_paired_keypoint_oks,
    make_positive_depth_joint_mask,
)


VC_NAIVE_COMPARISON_PROFILE_V1_ID: Final = 'VC_NAIVE_COMPARISON_METRICS_V1'
VC_NAIVE_COMPARISON_PROFILE_V2_ID: Final = 'VC_NAIVE_COMPARISON_METRICS_V2'
VC_NAIVE_COMPARISON_PROFILE_ID: Final = VC_NAIVE_COMPARISON_PROFILE_V2_ID
VC_NAIVE_MATCHED_PROFILE_ID: Final = 'VC_NAIVE_MATCHED_METRICS_V1'
VC_NAIVE_COMPARISON_METRICS_V1: Final = (
    'MPJPE-WORLD',
    'T-MPJPE',
    'OKS-VIS',
    'ACC-ROOT-RATIO',
)
VC_NAIVE_COMPARISON_METRICS_V2: Final = (
    'MPJPE-WORLD',
    'T-MPJPE',
    'PA-MPJPE',
    'OKS-VIS',
    'ACC-ROOT-RATIO',
)
VC_NAIVE_COMPARISON_METRICS: Final = VC_NAIVE_COMPARISON_METRICS_V2


def require_exact_nonnegative_int(value: int, name: str) -> int:
    '''Return one exact non-negative integer.'''
    if type(value) is not int:
        raise TypeError('%s must be an exact int' % name)
    if value < 0:
        raise ValueError('%s must be non-negative' % name)
    return value


def require_finite_nonnegative(value: float, name: str) -> float:
    '''Return one finite non-negative float.'''
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError('%s must be finite and non-negative' % name)
    return converted


def require_identity(value: str, name: str) -> str:
    '''Return one non-empty exact string identity.'''
    if type(value) is not str:
        raise TypeError('%s must be an exact str' % name)
    if not value:
        raise ValueError('%s must be non-empty' % name)
    return value


@dataclass(frozen=True, slots=True)
class VirtualCrowd_Naive_Comparison_Sequence_Summary:
    '''Additive sufficient statistics for one selected VirtualCrowd scene.'''

    profile_id: str
    filtering_id: str
    split_id: str
    scene_id: str
    selected_gt_count: int
    matched_selected_count: int
    mpjpe_world_sum_m: float
    mpjpe_world_count: int
    t_mpjpe_sum_m: float
    t_mpjpe_count: int
    oks_vis_sum: float
    oks_vis_count: int
    acc_root_predicted_sum_m_per_frame2: float
    acc_root_reference_sum_m_per_frame2: float
    acc_root_sample_count: int
    pa_mpjpe_sum_m: float | None = None
    pa_mpjpe_count: int | None = None

    def __post_init__(self) -> None:
        if self.profile_id not in (
                VC_NAIVE_COMPARISON_PROFILE_V1_ID,
                VC_NAIVE_COMPARISON_PROFILE_V2_ID,
            ):
            raise ValueError('profile_id does not identify the naive comparison profile')
        require_identity(self.filtering_id, 'filtering_id')
        require_identity(self.split_id, 'split_id')
        require_identity(self.scene_id, 'scene_id')
        count_names = (
            'selected_gt_count',
            'matched_selected_count',
            'mpjpe_world_count',
            't_mpjpe_count',
            'oks_vis_count',
            'acc_root_sample_count',
        )
        for name in count_names:
            require_exact_nonnegative_int(cast(int, getattr(self, name)), name)
        sum_names = (
            'mpjpe_world_sum_m',
            't_mpjpe_sum_m',
            'oks_vis_sum',
            'acc_root_predicted_sum_m_per_frame2',
            'acc_root_reference_sum_m_per_frame2',
        )
        for name in sum_names:
            require_finite_nonnegative(cast(float, getattr(self, name)), name)
        if self.matched_selected_count != self.selected_gt_count:
            raise ValueError('matched_selected_count must equal selected_gt_count')
        expected_joint_count = 24 * self.selected_gt_count
        if self.mpjpe_world_count != expected_joint_count:
            raise ValueError('mpjpe_world_count must equal 24 * selected_gt_count')
        if self.t_mpjpe_count != expected_joint_count:
            raise ValueError('t_mpjpe_count must equal 24 * selected_gt_count')
        if self.profile_id == VC_NAIVE_COMPARISON_PROFILE_V1_ID:
            if self.pa_mpjpe_sum_m is not None or self.pa_mpjpe_count is not None:
                raise ValueError('V1 naive summary must not contain PA-MPJPE')
        else:
            if self.pa_mpjpe_sum_m is None or self.pa_mpjpe_count is None:
                raise ValueError('V2 naive summary requires PA-MPJPE statistics')
            require_finite_nonnegative(self.pa_mpjpe_sum_m, 'pa_mpjpe_sum_m')
            require_exact_nonnegative_int(self.pa_mpjpe_count, 'pa_mpjpe_count')
            if self.pa_mpjpe_count != expected_joint_count:
                raise ValueError(
                    'pa_mpjpe_count must equal 24 * selected_gt_count')
        if self.oks_vis_count > self.selected_gt_count:
            raise ValueError('oks_vis_count cannot exceed selected_gt_count')
        if self.oks_vis_sum > float(self.oks_vis_count):
            raise ValueError('oks_vis_sum cannot exceed oks_vis_count')
        zero_support_sums = (
            (self.mpjpe_world_count, self.mpjpe_world_sum_m, 'mpjpe_world_sum_m'),
            (self.t_mpjpe_count, self.t_mpjpe_sum_m, 't_mpjpe_sum_m'),
            (self.oks_vis_count, self.oks_vis_sum, 'oks_vis_sum'),
            (
                self.acc_root_sample_count,
                self.acc_root_predicted_sum_m_per_frame2,
                'acc_root_predicted_sum_m_per_frame2',
            ),
            (
                self.acc_root_sample_count,
                self.acc_root_reference_sum_m_per_frame2,
                'acc_root_reference_sum_m_per_frame2',
            ),
        )
        for count, value, name in zero_support_sums:
            if count == 0 and value != 0.0:
                raise ValueError('zero support requires zero %s' % name)
        if self.pa_mpjpe_count == 0 and self.pa_mpjpe_sum_m != 0.0:
            raise ValueError('zero support requires zero pa_mpjpe_sum_m')


@dataclass(frozen=True, slots=True)
class VirtualCrowd_Naive_Comparison_Result:
    '''Reduced provisional NAIVE comparison result.'''

    profile_id: str
    filtering_id: str
    split_id: str
    scene_count: int
    selected_gt_count: int
    matched_selected_count: int
    joint_sample_count: int
    oks_vis_count: int
    acc_root_sample_count: int
    mpjpe_world_mm: float
    t_mpjpe_mm: float
    oks_vis: float
    acc_root_ratio: float
    pa_mpjpe_mm: float | None = None

    def __post_init__(self) -> None:
        if self.profile_id not in (
                VC_NAIVE_COMPARISON_PROFILE_V1_ID,
                VC_NAIVE_COMPARISON_PROFILE_V2_ID,
            ):
            raise ValueError('profile_id does not identify the naive comparison profile')
        require_identity(self.filtering_id, 'filtering_id')
        require_identity(self.split_id, 'split_id')
        count_names = (
            'scene_count',
            'selected_gt_count',
            'matched_selected_count',
            'joint_sample_count',
            'oks_vis_count',
            'acc_root_sample_count',
        )
        for name in count_names:
            require_exact_nonnegative_int(cast(int, getattr(self, name)), name)
        if self.scene_count <= 0:
            raise ValueError('scene_count must be positive')
        if self.selected_gt_count <= 0:
            raise ValueError('selected_gt_count must be positive')
        if self.matched_selected_count != self.selected_gt_count:
            raise ValueError('matched_selected_count must equal selected_gt_count')
        if self.joint_sample_count != 24 * self.selected_gt_count:
            raise ValueError('joint_sample_count must equal 24 * selected_gt_count')
        if not 0 < self.oks_vis_count <= self.selected_gt_count:
            raise ValueError('oks_vis_count must be positive and bounded by population')
        if self.acc_root_sample_count <= 0:
            raise ValueError('acc_root_sample_count must be positive')
        metric_names = (
            'mpjpe_world_mm',
            't_mpjpe_mm',
            'oks_vis',
            'acc_root_ratio',
        )
        for name in metric_names:
            require_finite_nonnegative(cast(float, getattr(self, name)), name)
        if self.profile_id == VC_NAIVE_COMPARISON_PROFILE_V1_ID:
            if self.pa_mpjpe_mm is not None:
                raise ValueError('V1 naive result must not contain PA-MPJPE')
        elif self.pa_mpjpe_mm is None:
            raise ValueError('V2 naive result requires PA-MPJPE')
        else:
            require_finite_nonnegative(self.pa_mpjpe_mm, 'pa_mpjpe_mm')
        if self.oks_vis > 1.0:
            raise ValueError('oks_vis cannot exceed one')


@dataclass(frozen=True, slots=True)
class VirtualCrowd_Naive_Matched_Sequence_Summary:
    '''One scene's association counts and matched-only NAIVE statistics.'''

    profile_id: str
    gt_count: int
    prediction_count: int
    tp: int
    fn: int
    fp: int
    matched: VirtualCrowd_Naive_Comparison_Sequence_Summary

    def __post_init__(self) -> None:
        if self.profile_id != VC_NAIVE_MATCHED_PROFILE_ID:
            raise ValueError('profile_id does not identify NAIVE matched')
        for name in ('gt_count', 'prediction_count', 'tp', 'fn', 'fp'):
            require_exact_nonnegative_int(cast(int, getattr(self, name)), name)
        if self.gt_count != self.tp + self.fn:
            raise ValueError('GT count must equal TP plus FN')
        if self.prediction_count != self.tp + self.fp:
            raise ValueError('prediction count must equal TP plus FP')
        if type(self.matched) \
                is not VirtualCrowd_Naive_Comparison_Sequence_Summary:
            raise TypeError('matched must contain exact NAIVE statistics')
        if self.matched.profile_id != VC_NAIVE_COMPARISON_PROFILE_ID:
            raise ValueError('matched statistics must use current NAIVE profile')
        if self.matched.selected_gt_count != self.tp:
            raise ValueError('matched NAIVE support must equal TP')


@dataclass(frozen=True, slots=True)
class VirtualCrowd_Naive_Matched_Result:
    '''Micro-reduced association quality and matched-only NAIVE metrics.'''

    profile_id: str
    filtering_id: str
    split_id: str
    scene_count: int
    gt_count: int
    prediction_count: int
    tp: int
    fn: int
    fp: int
    precision: float
    recall: float
    f1: float
    joint_sample_count: int
    oks_vis_count: int
    acc_root_sample_count: int
    mpjpe_world_mm: float | None
    t_mpjpe_mm: float | None
    pa_mpjpe_mm: float | None
    oks_vis: float | None
    acc_root_ratio: float | None
    scene_summaries: tuple[VirtualCrowd_Naive_Matched_Sequence_Summary, ...]

    def __post_init__(self) -> None:
        if self.profile_id != VC_NAIVE_MATCHED_PROFILE_ID:
            raise ValueError('profile_id does not identify NAIVE matched')
        require_identity(self.filtering_id, 'filtering_id')
        require_identity(self.split_id, 'split_id')
        for name in (
                'scene_count', 'gt_count', 'prediction_count', 'tp', 'fn',
                'fp', 'joint_sample_count', 'oks_vis_count',
                'acc_root_sample_count',
            ):
            require_exact_nonnegative_int(cast(int, getattr(self, name)), name)
        if self.scene_count <= 0 or self.scene_count != len(self.scene_summaries):
            raise ValueError('scene count must equal nonempty scene summaries')
        if self.gt_count != self.tp + self.fn:
            raise ValueError('GT count must equal TP plus FN')
        if self.prediction_count != self.tp + self.fp:
            raise ValueError('prediction count must equal TP plus FP')
        if self.joint_sample_count != 24 * self.tp:
            raise ValueError('joint support must equal 24 times TP')
        if not 0 <= self.oks_vis_count <= self.tp:
            raise ValueError('OKS support must lie within TP')
        for name in ('precision', 'recall', 'f1'):
            value = require_finite_nonnegative(
                cast(float, getattr(self, name)), name,
            )
            if value > 1.0:
                raise ValueError('%s cannot exceed one' % name)
        for name in (
                'mpjpe_world_mm', 't_mpjpe_mm', 'pa_mpjpe_mm', 'oks_vis',
                'acc_root_ratio',
            ):
            value = cast(float | None, getattr(self, name))
            if value is not None:
                require_finite_nonnegative(value, name)
        if (self.tp == 0) != (self.mpjpe_world_mm is None) \
                or (self.tp == 0) != (self.t_mpjpe_mm is None) \
                or (self.tp == 0) != (self.pa_mpjpe_mm is None):
            raise ValueError('MPJPE metrics must be nullable exactly at zero TP')
        if (self.oks_vis_count == 0) != (self.oks_vis is None):
            raise ValueError('OKS metric and support must agree')
        if self.oks_vis is not None and self.oks_vis > 1.0:
            raise ValueError('oks_vis cannot exceed one')
        summaries = self.scene_summaries
        if any(
                type(value)
                is not VirtualCrowd_Naive_Matched_Sequence_Summary
                for value in summaries
            ):
            raise TypeError('scene_summaries contain an unsupported type')
        if summaries != tuple(
                sorted(summaries, key=lambda value: value.matched.scene_id)
            ) or len({value.matched.scene_id for value in summaries}) \
                != len(summaries):
            raise ValueError('scene summaries must have unique sorted IDs')
        if any(
                value.matched.filtering_id != self.filtering_id
                or value.matched.split_id != self.split_id
                for value in summaries
            ):
            raise ValueError('scene summary protocol identity differs')
        expected_counts = (
            sum(value.gt_count for value in summaries),
            sum(value.prediction_count for value in summaries),
            sum(value.tp for value in summaries),
            sum(value.fn for value in summaries),
            sum(value.fp for value in summaries),
            sum(value.matched.mpjpe_world_count for value in summaries),
            sum(value.matched.oks_vis_count for value in summaries),
            sum(value.matched.acc_root_sample_count for value in summaries),
        )
        if expected_counts != (
                self.gt_count, self.prediction_count, self.tp, self.fn,
                self.fp, self.joint_sample_count, self.oks_vis_count,
                self.acc_root_sample_count,
            ):
            raise ValueError('result counts differ from scene summaries')
        expected_precision = self.tp / self.prediction_count \
            if self.prediction_count else 0.0
        expected_recall = self.tp / self.gt_count if self.gt_count else 0.0
        expected_f1 = (
            2.0 * expected_precision * expected_recall
            / (expected_precision + expected_recall)
            if expected_precision + expected_recall else 0.0
        )
        if (self.precision, self.recall, self.f1) != (
                expected_precision, expected_recall, expected_f1,
            ):
            raise ValueError('matching metrics differ from raw counts')
        matched = tuple(value.matched for value in summaries)
        mpjpe_sum = math.fsum(value.mpjpe_world_sum_m for value in matched)
        t_mpjpe_sum = math.fsum(value.t_mpjpe_sum_m for value in matched)
        pa_mpjpe_sum = math.fsum(
            cast(float, value.pa_mpjpe_sum_m) for value in matched
        )
        oks_sum = math.fsum(value.oks_vis_sum for value in matched)
        acc_predicted_sum = math.fsum(
            value.acc_root_predicted_sum_m_per_frame2 for value in matched
        )
        acc_reference_sum = math.fsum(
            value.acc_root_reference_sum_m_per_frame2 for value in matched
        )
        expected_metrics = (
            None if self.joint_sample_count == 0
            else 1000.0 * mpjpe_sum / self.joint_sample_count,
            None if self.joint_sample_count == 0
            else 1000.0 * t_mpjpe_sum / self.joint_sample_count,
            None if self.joint_sample_count == 0
            else 1000.0 * pa_mpjpe_sum / self.joint_sample_count,
            None if self.oks_vis_count == 0
            else oks_sum / self.oks_vis_count,
            None
            if self.acc_root_sample_count == 0 or acc_reference_sum <= 0.0
            else acc_predicted_sum / acc_reference_sum,
        )
        if (
                self.mpjpe_world_mm, self.t_mpjpe_mm, self.pa_mpjpe_mm,
                self.oks_vis, self.acc_root_ratio,
            ) != expected_metrics:
            raise ValueError('NAIVE metrics differ from raw scene statistics')


@dataclass(frozen=True, slots=True)
class VirtualCrowd_Direct_Target_Join:
    '''One immutable validated alignment of GT and direct-target predictions.'''

    sequence: Corrected_Crowd_Sequence
    gt_rows: NDArray[np.int64]
    prediction_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        sequence = validate_corrected_crowd_sequence(self.sequence)
        gt_rows = np.asarray(self.gt_rows)
        prediction_rows = np.asarray(self.prediction_rows)
        if not np.issubdtype(gt_rows.dtype, np.integer):
            raise TypeError('gt_rows must have integer dtype')
        if not np.issubdtype(prediction_rows.dtype, np.integer):
            raise TypeError('prediction_rows must have integer dtype')
        gt_rows = np.array(gt_rows, dtype=np.int64, copy=True)
        prediction_rows = np.array(
            prediction_rows,
            dtype=np.int64,
            copy=True,
        )
        if gt_rows.ndim != 1 or prediction_rows.ndim != 1:
            raise ValueError('direct-target join rows must be one-dimensional')
        if len(gt_rows) != len(prediction_rows):
            raise ValueError('direct-target join row counts must match')
        if len(np.unique(gt_rows)) != len(gt_rows):
            raise ValueError('direct-target GT rows must be unique')
        if len(np.unique(prediction_rows)) != len(prediction_rows):
            raise ValueError('direct-target prediction rows must be unique')
        if np.any((gt_rows < 0) | (gt_rows >= len(sequence.gt_frame_ids))):
            raise ValueError('direct-target GT row is out of range')
        if np.any(
            (prediction_rows < 0)
            | (prediction_rows >= len(sequence.prediction_frame_ids))
        ):
            raise ValueError('direct-target prediction row is out of range')
        targets = sequence.prediction_identity_target_gt_rows[prediction_rows]
        if not np.array_equal(targets, gt_rows):
            raise ValueError('direct-target rows do not satisfy identity targets')
        if not np.array_equal(
            sequence.prediction_frame_ids[prediction_rows],
            sequence.gt_frame_ids[gt_rows],
        ):
            raise ValueError('direct-target rows do not share exact frame IDs')
        gt_rows.setflags(write=False)
        prediction_rows.setflags(write=False)
        object.__setattr__(self, 'sequence', sequence)
        object.__setattr__(self, 'gt_rows', gt_rows)
        object.__setattr__(self, 'prediction_rows', prediction_rows)


def direct_target_join(
    sequence: Corrected_Crowd_Sequence,
    selected_gt_mask: NDArray[np.generic],
) -> VirtualCrowd_Direct_Target_Join:
    '''Join every selected GT row to exactly one direct-target prediction.'''
    selected_mask = bool_array(selected_gt_mask, 'selected_gt_mask')
    gt_count = len(sequence.gt_frame_ids)
    if selected_mask.shape != (gt_count,):
        raise ValueError('selected_gt_mask must have shape (%d,)' % gt_count)
    selected_gt_rows = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    targets = sequence.prediction_identity_target_gt_rows
    mapped_prediction_rows = np.flatnonzero(targets >= 0).astype(
        np.int64,
        copy=False,
    )
    target_counts = np.bincount(
        targets[mapped_prediction_rows],
        minlength=gt_count,
    )
    if np.any(target_counts[selected_gt_rows] != 1):
        raise ValueError(
            'every selected GT row must have exactly one direct-target prediction'
        )
    prediction_for_gt = np.full(gt_count, -1, dtype=np.int64)
    prediction_for_gt[targets[mapped_prediction_rows]] = mapped_prediction_rows
    return VirtualCrowd_Direct_Target_Join(
        sequence,
        selected_gt_rows,
        prediction_for_gt[selected_gt_rows],
    )


def compute_virtualcrowd_mpjpe_world_statistics(
    join: VirtualCrowd_Direct_Target_Join,
) -> tuple[float, int]:
    '''Return MPJPE-WORLD error sum in metres and joint support.'''
    if type(join) is not VirtualCrowd_Direct_Target_Join:
        raise TypeError('join must be a VirtualCrowd_Direct_Target_Join')
    sequence = join.sequence
    predicted = sequence.prediction_joints_world_m[join.prediction_rows]
    reference = sequence.gt_joints_world_m[join.gt_rows]
    absolute = compute_joint_position_errors(predicted, reference).reshape(-1)
    return math.fsum(float(value) for value in absolute), int(absolute.size)


def compute_virtualcrowd_t_mpjpe_statistics(
    join: VirtualCrowd_Direct_Target_Join,
) -> tuple[float, int]:
    '''Return pelvis-relative MPJPE error sum in metres and joint support.'''
    if type(join) is not VirtualCrowd_Direct_Target_Join:
        raise TypeError('join must be a VirtualCrowd_Direct_Target_Join')
    sequence = join.sequence
    predicted = sequence.prediction_joints_world_m[join.prediction_rows]
    reference = sequence.gt_joints_world_m[join.gt_rows]
    predicted_local = predicted - predicted[:, :1]
    reference_local = reference - reference[:, :1]
    pelvis_relative = compute_joint_position_errors(
        predicted_local,
        reference_local,
    ).reshape(-1)
    return math.fsum(float(value) for value in pelvis_relative), int(
        pelvis_relative.size
    )


def compute_virtualcrowd_pa_mpjpe_statistics(
    join: VirtualCrowd_Direct_Target_Join,
) -> tuple[float, int]:
    '''Return per-person-frame PA-MPJPE sum in metres and joint support.'''
    if type(join) is not VirtualCrowd_Direct_Target_Join:
        raise TypeError('join must be a VirtualCrowd_Direct_Target_Join')
    sequence = join.sequence
    errors = compute_pa_joint_position_errors(
        sequence.prediction_joints_world_m[join.prediction_rows],
        sequence.gt_joints_world_m[join.gt_rows],
    ).reshape(-1)
    return math.fsum(float(value) for value in errors), int(errors.size)


def compute_virtualcrowd_oks_vis_statistics(
    join: VirtualCrowd_Direct_Target_Join,
) -> tuple[float, int]:
    '''Return visibility-aware paired OKS sum and person-frame support.'''
    if type(join) is not VirtualCrowd_Direct_Target_Join:
        raise TypeError('join must be a VirtualCrowd_Direct_Target_Join')
    sequence = join.sequence
    gt_rows = join.gt_rows
    prediction_rows = join.prediction_rows
    values: list[float] = []
    for frame_id in np.unique(sequence.gt_frame_ids[gt_rows]):
        frame_selected = sequence.gt_frame_ids[gt_rows] == frame_id
        frame_gt_rows = gt_rows[frame_selected]
        frame_prediction_rows = prediction_rows[frame_selected]
        valid = make_positive_depth_joint_mask(
            sequence.gt_visibility_native[frame_gt_rows] > 0.0,
            sequence.prediction_coco17_camera_depth_m[frame_prediction_rows],
        )
        row_supported = np.any(valid, axis=1)
        if not np.any(row_supported):
            continue
        supported_gt_rows = frame_gt_rows[row_supported]
        supported_prediction_rows = frame_prediction_rows[row_supported]
        supported_valid = valid[row_supported]
        bbox = sequence.gt_bbox_xyxy_px[supported_gt_rows]
        area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        paired = compute_paired_keypoint_oks(
            sequence.gt_coco17_xy_px[supported_gt_rows],
            sequence.prediction_coco17_xy_px[supported_prediction_rows],
            area,
            COCO17_SIGMAS,
            supported_valid,
        )
        values.extend(float(value) for value in paired)
    return math.fsum(values), len(values)


def _exact_consecutive_segments(
    sequence: Corrected_Crowd_Sequence,
    gt_rows: NDArray[np.int64],
    prediction_rows: NDArray[np.int64],
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    '''Return maximal selected direct-match segments by native GT identity.'''
    segments: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    track_ids = sequence.gt_track_ids[gt_rows]
    for track_id in np.unique(track_ids):
        selected = np.flatnonzero(track_ids == track_id)
        order = np.argsort(
            sequence.gt_frame_ids[gt_rows[selected]],
            kind='stable',
        )
        selected = selected[order]
        frames = sequence.gt_frame_ids[gt_rows[selected]]
        starts = np.concatenate((
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.diff(frames) != 1).astype(np.int64) + 1,
        ))
        ends = np.concatenate((
            starts[1:],
            np.array([len(selected)], dtype=np.int64),
        ))
        for start, end in zip(starts, ends, strict=True):
            segment_rows = selected[int(start):int(end)]
            segments.append((gt_rows[segment_rows], prediction_rows[segment_rows]))
    return segments


def root_acceleration_magnitudes(
    root_world_m: NDArray[np.generic],
) -> NDArray[np.float64]:
    '''Return historical twice-central-differenced root magnitudes.'''
    root = np.asarray(root_world_m, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError('root_world_m must have shape (T, 3)')
    if not np.isfinite(root).all():
        raise ValueError('root_world_m must be finite')
    return compute_central_acceleration_magnitudes(root)


def compute_virtualcrowd_acc_root_ratio_statistics(
    join: VirtualCrowd_Direct_Target_Join,
) -> tuple[float, float, int]:
    '''Return additive predicted/reference acceleration magnitudes and support.'''
    if type(join) is not VirtualCrowd_Direct_Target_Join:
        raise TypeError('join must be a VirtualCrowd_Direct_Target_Join')
    sequence = join.sequence
    predicted_values: list[float] = []
    reference_values: list[float] = []
    for segment_gt_rows, segment_prediction_rows in _exact_consecutive_segments(
        sequence,
        join.gt_rows,
        join.prediction_rows,
    ):
        predicted = root_acceleration_magnitudes(
            sequence.prediction_joints_world_m[segment_prediction_rows, 0],
        )
        reference = root_acceleration_magnitudes(
            sequence.gt_joints_world_m[segment_gt_rows, 0],
        )
        if predicted.shape != reference.shape:
            raise ValueError('predicted/reference acceleration supports differ')
        predicted_values.extend(float(value) for value in predicted)
        reference_values.extend(float(value) for value in reference)
    return (
        math.fsum(predicted_values),
        math.fsum(reference_values),
        len(predicted_values),
    )


def evaluate_virtualcrowd_naive_comparison(
    sequence: Corrected_Crowd_Sequence,
    filtering_id: str,
    split_id: str,
    selected_gt_mask: NDArray[np.generic],
) -> VirtualCrowd_Naive_Comparison_Sequence_Summary:
    '''Evaluate one scene on one exact caller-selected population.'''
    validated_filtering_id = require_identity(filtering_id, 'filtering_id')
    validated_split_id = require_identity(split_id, 'split_id')
    join = direct_target_join(sequence, selected_gt_mask)
    validated = join.sequence
    selected_count = len(join.gt_rows)
    mpjpe_sum, joint_count = compute_virtualcrowd_mpjpe_world_statistics(join)
    t_mpjpe_sum, t_joint_count = compute_virtualcrowd_t_mpjpe_statistics(join)
    pa_mpjpe_sum, pa_joint_count = compute_virtualcrowd_pa_mpjpe_statistics(join)
    if t_joint_count != joint_count or pa_joint_count != joint_count:
        raise ValueError('MPJPE leaf supports differ')
    oks_sum, oks_count = compute_virtualcrowd_oks_vis_statistics(join)
    acc_predicted_sum, acc_reference_sum, acc_count = (
        compute_virtualcrowd_acc_root_ratio_statistics(join)
    )
    return VirtualCrowd_Naive_Comparison_Sequence_Summary(
        profile_id=VC_NAIVE_COMPARISON_PROFILE_ID,
        filtering_id=validated_filtering_id,
        split_id=validated_split_id,
        scene_id=validated.scene_id,
        selected_gt_count=selected_count,
        matched_selected_count=len(join.prediction_rows),
        mpjpe_world_sum_m=mpjpe_sum,
        mpjpe_world_count=joint_count,
        t_mpjpe_sum_m=t_mpjpe_sum,
        t_mpjpe_count=joint_count,
        oks_vis_sum=oks_sum,
        oks_vis_count=oks_count,
        acc_root_predicted_sum_m_per_frame2=acc_predicted_sum,
        acc_root_reference_sum_m_per_frame2=acc_reference_sum,
        acc_root_sample_count=acc_count,
        pa_mpjpe_sum_m=pa_mpjpe_sum,
        pa_mpjpe_count=pa_joint_count,
    )


def evaluate_virtualcrowd_naive_matched(
        sequence: Corrected_Crowd_Sequence,
        filtering_id: str,
        split_id: str,
        selected_gt_mask: NDArray[np.generic],
    ) -> VirtualCrowd_Naive_Matched_Sequence_Summary:
    '''Evaluate one closed association partition on selected GT rows.'''
    validated = validate_corrected_crowd_sequence(sequence)
    selected = bool_array(selected_gt_mask, 'selected_gt_mask')
    gt_count = len(validated.gt_frame_ids)
    prediction_count = len(validated.prediction_frame_ids)
    if selected.shape != (gt_count,):
        raise ValueError('selected_gt_mask must have shape (%d,)' % gt_count)
    matched_gt = validated.matched_gt_rows
    matched_prediction = validated.matched_prediction_rows
    if np.any(~selected[matched_gt]):
        raise ValueError('matched GT rows must belong to selected population')
    mapped_prediction = np.flatnonzero(
        validated.prediction_identity_target_gt_rows >= 0,
    ).astype(np.int64, copy=False)
    if not np.array_equal(np.sort(matched_prediction), mapped_prediction):
        raise ValueError('mapped predictions must equal the matched partition')
    if not np.array_equal(
            validated.prediction_identity_target_gt_rows[matched_prediction],
            matched_gt,
        ):
        raise ValueError('matched row pairs must equal identity targets')
    matched_mask = np.zeros(gt_count, dtype=np.bool_)
    matched_mask[matched_gt] = True
    tp = len(matched_gt)
    selected_count = int(np.count_nonzero(selected))
    matched = evaluate_virtualcrowd_naive_comparison(
        validated, filtering_id, split_id, matched_mask,
    )
    return VirtualCrowd_Naive_Matched_Sequence_Summary(
        profile_id=VC_NAIVE_MATCHED_PROFILE_ID,
        gt_count=selected_count,
        prediction_count=prediction_count,
        tp=tp,
        fn=selected_count - tp,
        fp=prediction_count - tp,
        matched=matched,
    )


def reduce_virtualcrowd_naive_matched_summaries(
        summaries: Sequence[VirtualCrowd_Naive_Matched_Sequence_Summary],
    ) -> VirtualCrowd_Naive_Matched_Result:
    '''Micro-reduce association counts and matched-only NAIVE statistics.'''
    values = tuple(summaries)
    if not values:
        raise ValueError('NAIVE matched summary collection is empty')
    if any(
            type(value) is not VirtualCrowd_Naive_Matched_Sequence_Summary
            for value in values
        ):
        raise TypeError('summaries must contain NAIVE matched summaries')
    ordered = tuple(sorted(values, key=lambda value: value.matched.scene_id))
    scene_ids = tuple(value.matched.scene_id for value in ordered)
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError('scene_id values must be unique')
    filtering_id = ordered[0].matched.filtering_id
    split_id = ordered[0].matched.split_id
    if any(value.matched.filtering_id != filtering_id for value in ordered):
        raise ValueError('all summaries must have the same filtering_id')
    if any(value.matched.split_id != split_id for value in ordered):
        raise ValueError('all summaries must have the same split_id')
    matched = tuple(value.matched for value in ordered)
    joint_count = sum(value.mpjpe_world_count for value in matched)
    t_joint_count = sum(value.t_mpjpe_count for value in matched)
    if any(value.pa_mpjpe_count is None for value in matched):
        raise ValueError('V2 PA-MPJPE sufficient statistics are missing')
    pa_joint_count = sum(cast(int, value.pa_mpjpe_count) for value in matched)
    if joint_count != t_joint_count or joint_count != pa_joint_count:
        raise ValueError('MPJPE sufficient-statistic counts differ')
    gt_count = sum(value.gt_count for value in ordered)
    prediction_count = sum(value.prediction_count for value in ordered)
    tp = sum(value.tp for value in ordered)
    fn = sum(value.fn for value in ordered)
    fp = sum(value.fp for value in ordered)
    oks_count = sum(value.oks_vis_count for value in matched)
    acc_count = sum(value.acc_root_sample_count for value in matched)
    mpjpe_sum = math.fsum(value.mpjpe_world_sum_m for value in matched)
    t_mpjpe_sum = math.fsum(value.t_mpjpe_sum_m for value in matched)
    pa_mpjpe_sum = math.fsum(
        cast(float, value.pa_mpjpe_sum_m) for value in matched
    )
    oks_sum = math.fsum(value.oks_vis_sum for value in matched)
    acc_predicted_sum = math.fsum(
        value.acc_root_predicted_sum_m_per_frame2 for value in matched
    )
    acc_reference_sum = math.fsum(
        value.acc_root_reference_sum_m_per_frame2 for value in matched
    )
    precision = tp / prediction_count if prediction_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return VirtualCrowd_Naive_Matched_Result(
        profile_id=VC_NAIVE_MATCHED_PROFILE_ID,
        filtering_id=filtering_id,
        split_id=split_id,
        scene_count=len(ordered),
        gt_count=gt_count,
        prediction_count=prediction_count,
        tp=tp,
        fn=fn,
        fp=fp,
        precision=precision,
        recall=recall,
        f1=f1,
        joint_sample_count=joint_count,
        oks_vis_count=oks_count,
        acc_root_sample_count=acc_count,
        mpjpe_world_mm=None if joint_count == 0 else 1000.0 * mpjpe_sum / joint_count,
        t_mpjpe_mm=None if joint_count == 0 else 1000.0 * t_mpjpe_sum / joint_count,
        pa_mpjpe_mm=None if joint_count == 0 else 1000.0 * pa_mpjpe_sum / joint_count,
        oks_vis=None if oks_count == 0 else oks_sum / oks_count,
        acc_root_ratio=(
            None
            if acc_count == 0 or acc_reference_sum <= 0.0
            else acc_predicted_sum / acc_reference_sum
        ),
        scene_summaries=ordered,
    )


def reduce_virtualcrowd_naive_comparison_summaries(
    summaries: Sequence[VirtualCrowd_Naive_Comparison_Sequence_Summary],
) -> VirtualCrowd_Naive_Comparison_Result:
    '''Reduce scene sufficient statistics into the provisional NAIVE metrics.'''
    summary_tuple = tuple(summaries)
    if not summary_tuple:
        raise ValueError('naive comparison summary collection is empty')
    if any(
        type(summary) is not VirtualCrowd_Naive_Comparison_Sequence_Summary
        for summary in summary_tuple
    ):
        raise TypeError('summaries must contain naive comparison summaries')
    ordered = tuple(sorted(summary_tuple, key=lambda summary: summary.scene_id))
    scene_ids = tuple(summary.scene_id for summary in ordered)
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError('scene_id values must be unique')
    filtering_id = ordered[0].filtering_id
    split_id = ordered[0].split_id
    if any(summary.filtering_id != filtering_id for summary in ordered):
        raise ValueError('all summaries must have the same filtering_id')
    if any(summary.split_id != split_id for summary in ordered):
        raise ValueError('all summaries must have the same split_id')
    if any(
            summary.profile_id != VC_NAIVE_COMPARISON_PROFILE_ID
            for summary in ordered
        ):
        raise ValueError('all summaries must use the current naive profile')
    selected_count = sum(summary.selected_gt_count for summary in ordered)
    matched_count = sum(summary.matched_selected_count for summary in ordered)
    joint_count = sum(summary.mpjpe_world_count for summary in ordered)
    t_joint_count = sum(summary.t_mpjpe_count for summary in ordered)
    if any(summary.pa_mpjpe_count is None for summary in ordered):
        raise ValueError('V2 PA-MPJPE sufficient statistics are missing')
    pa_joint_count = sum(cast(int, summary.pa_mpjpe_count) for summary in ordered)
    if t_joint_count != joint_count or pa_joint_count != joint_count:
        raise ValueError('MPJPE sufficient-statistic counts differ')
    oks_count = sum(summary.oks_vis_count for summary in ordered)
    acc_count = sum(summary.acc_root_sample_count for summary in ordered)
    if selected_count <= 0 or joint_count <= 0:
        raise ValueError('selected joint population is empty')
    if oks_count <= 0:
        raise ValueError('OKS-VIS support is empty')
    if acc_count <= 0:
        raise ValueError('ACC-ROOT-RATIO support is empty')
    mpjpe_sum = math.fsum(summary.mpjpe_world_sum_m for summary in ordered)
    t_mpjpe_sum = math.fsum(summary.t_mpjpe_sum_m for summary in ordered)
    pa_mpjpe_sum = math.fsum(
        cast(float, summary.pa_mpjpe_sum_m) for summary in ordered)
    oks_sum = math.fsum(summary.oks_vis_sum for summary in ordered)
    acc_predicted_sum = math.fsum(
        summary.acc_root_predicted_sum_m_per_frame2 for summary in ordered
    )
    acc_reference_sum = math.fsum(
        summary.acc_root_reference_sum_m_per_frame2 for summary in ordered
    )
    if acc_reference_sum <= 0.0:
        raise ValueError('ACC-ROOT-RATIO reference denominator must be positive')
    return VirtualCrowd_Naive_Comparison_Result(
        profile_id=VC_NAIVE_COMPARISON_PROFILE_ID,
        filtering_id=filtering_id,
        split_id=split_id,
        scene_count=len(ordered),
        selected_gt_count=selected_count,
        matched_selected_count=matched_count,
        joint_sample_count=joint_count,
        oks_vis_count=oks_count,
        acc_root_sample_count=acc_count,
        mpjpe_world_mm=1000.0 * mpjpe_sum / joint_count,
        t_mpjpe_mm=1000.0 * t_mpjpe_sum / joint_count,
        pa_mpjpe_mm=1000.0 * pa_mpjpe_sum / joint_count,
        oks_vis=oks_sum / oks_count,
        acc_root_ratio=acc_predicted_sum / acc_reference_sum,
    )
