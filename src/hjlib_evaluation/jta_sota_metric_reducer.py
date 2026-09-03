'''Canonical six-metric reduction for paired JTA fitted-SMPL occurrences.'''
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from hjlib_geometry import (
    apply_rigid_registration,
    apply_similarity_registration,
    fit_rigid_registration,
    fit_similarity_registration,
)

from hjlib_evaluation.jta_person_detection_data import (
    JTA_ENDPOINT_NAMES,
    JTA_ENDPOINT_OKS_SIGMAS,
    SMPL54_ENDPOINT_INDICES,
)
from hjlib_evaluation.keypoint_oks import compute_paired_keypoint_oks
from hjlib_evaluation.smpl_joint_occurrence_reducer import (
    Joint_Coordinate_Frame,
    Joint_Unit,
    SMPL_Joint_Layout,
    validate_occurrence_inputs,
)


JTA_SOTA_METRIC_PROFILE = 'jta12_fitted_all_valid_v1'
JTA_SOTA_ROOT_INDICES = (1, 2)
JTA_SOTA_JOINT_COUNT = len(JTA_ENDPOINT_NAMES)


@dataclass(frozen=True, slots=True)
class JTA_SOTA_Metric_Sums:
    '''Additive sufficient statistics for canonical JTA SOTA metrics.'''

    profile: str
    joint_coordinate_frame: Joint_Coordinate_Frame
    image_coordinate_frame: str
    num_occurrence: int
    num_joint: int
    root_error_sum_mm: float
    mpjpe_sum_mm: float
    t_mpjpe_sum_mm: float
    rt_mpjpe_sum_mm: float
    pa_mpjpe_sum_mm: float
    oks_sum: float
    root_denominator: int
    joint_denominator: int
    oks_denominator: int

    def __post_init__(self) -> None:
        if self.profile != JTA_SOTA_METRIC_PROFILE:
            raise ValueError('unexpected JTA SOTA metric profile')
        if self.joint_coordinate_frame not in ('camera', 'world'):
            raise ValueError('joint coordinate frame differs')
        if type(self.image_coordinate_frame) is not str or not self.image_coordinate_frame:
            raise ValueError('image_coordinate_frame must be nonempty str')
        if type(self.num_occurrence) is not int or self.num_occurrence <= 0:
            raise ValueError('num_occurrence must be a positive exact int')
        if type(self.num_joint) is not int or self.num_joint != JTA_SOTA_JOINT_COUNT:
            raise ValueError('num_joint must equal the canonical JTA joint count')
        for name in (
                'root_error_sum_mm',
                'mpjpe_sum_mm',
                't_mpjpe_sum_mm',
                'rt_mpjpe_sum_mm',
                'pa_mpjpe_sum_mm',
                'oks_sum',
            ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError('%s must be finite and nonnegative' % name)
        expected = (
            self.num_occurrence,
            self.num_occurrence * self.num_joint,
            self.num_occurrence,
        )
        observed = (
            self.root_denominator,
            self.joint_denominator,
            self.oks_denominator,
        )
        if any(type(value) is not int for value in observed) or observed != expected:
            raise ValueError('metric denominators differ from occurrence count')

    def plus(self, other: 'JTA_SOTA_Metric_Sums') -> 'JTA_SOTA_Metric_Sums':
        '''Add statistics with identical semantic metadata.'''
        metadata = (
            self.profile,
            self.joint_coordinate_frame,
            self.image_coordinate_frame,
            self.num_joint,
        )
        other_metadata = (
            other.profile,
            other.joint_coordinate_frame,
            other.image_coordinate_frame,
            other.num_joint,
        )
        if metadata != other_metadata:
            raise ValueError('JTA SOTA metric metadata differs')
        return JTA_SOTA_Metric_Sums(
            profile=self.profile,
            joint_coordinate_frame=self.joint_coordinate_frame,
            image_coordinate_frame=self.image_coordinate_frame,
            num_occurrence=self.num_occurrence + other.num_occurrence,
            num_joint=self.num_joint,
            root_error_sum_mm=self.root_error_sum_mm + other.root_error_sum_mm,
            mpjpe_sum_mm=self.mpjpe_sum_mm + other.mpjpe_sum_mm,
            t_mpjpe_sum_mm=self.t_mpjpe_sum_mm + other.t_mpjpe_sum_mm,
            rt_mpjpe_sum_mm=self.rt_mpjpe_sum_mm + other.rt_mpjpe_sum_mm,
            pa_mpjpe_sum_mm=self.pa_mpjpe_sum_mm + other.pa_mpjpe_sum_mm,
            oks_sum=self.oks_sum + other.oks_sum,
            root_denominator=self.root_denominator + other.root_denominator,
            joint_denominator=self.joint_denominator + other.joint_denominator,
            oks_denominator=self.oks_denominator + other.oks_denominator,
        )


@dataclass(frozen=True, slots=True)
class JTA_SOTA_Metric_Result:
    '''Final occurrence-weighted canonical JTA SOTA metrics.'''

    profile: str
    joint_coordinate_frame: Joint_Coordinate_Frame
    image_coordinate_frame: str
    num_occurrence: int
    num_joint: int
    root_error_mm: float
    mpjpe_mm: float
    t_mpjpe_mm: float
    rt_mpjpe_mm: float
    pa_mpjpe_mm: float
    oks: float

    def __post_init__(self) -> None:
        if self.profile != JTA_SOTA_METRIC_PROFILE:
            raise ValueError('unexpected JTA SOTA metric profile')
        if self.joint_coordinate_frame not in ('camera', 'world'):
            raise ValueError('joint coordinate frame differs')
        if type(self.image_coordinate_frame) is not str or not self.image_coordinate_frame:
            raise ValueError('image_coordinate_frame must be nonempty str')
        if type(self.num_occurrence) is not int or self.num_occurrence <= 0:
            raise ValueError('num_occurrence must be positive')
        if self.num_joint != JTA_SOTA_JOINT_COUNT:
            raise ValueError('num_joint differs')
        for name in (
                'root_error_mm',
                'mpjpe_mm',
                't_mpjpe_mm',
                'rt_mpjpe_mm',
                'pa_mpjpe_mm',
                'oks',
            ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError('%s must be finite and nonnegative' % name)
        if self.oks > 1.0:
            raise ValueError('oks must not exceed one')


def validate_image_metric_inputs(
        pred_endpoint_xy: NDArray[np.generic],
        gt_endpoint_xy: NDArray[np.generic],
        gt_smpl24_xy: NDArray[np.generic],
        count: int,
        image_coordinate_frame: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Validate paired endpoint XY and fitted-target SMPL24 XY arrays.'''
    if type(image_coordinate_frame) is not str or not image_coordinate_frame:
        raise ValueError('image_coordinate_frame must be nonempty str')
    allowed = (np.dtype(np.float32), np.dtype(np.float64))
    normalized: list[NDArray[np.float64]] = []
    for name, raw, shape in (
            ('pred_endpoint_xy', pred_endpoint_xy, (count, JTA_SOTA_JOINT_COUNT, 2)),
            ('gt_endpoint_xy', gt_endpoint_xy, (count, JTA_SOTA_JOINT_COUNT, 2)),
            ('gt_smpl24_xy', gt_smpl24_xy, (count, 24, 2)),
        ):
        value = np.asarray(raw)
        if value.dtype not in allowed:
            raise TypeError('%s must use float32 or float64' % name)
        if value.shape != shape:
            raise ValueError('%s must have shape %r' % (name, shape))
        if not bool(np.isfinite(value).all()):
            raise ValueError('%s must be finite' % name)
        normalized.append(np.asarray(value, dtype=np.float64))
    return normalized[0], normalized[1], normalized[2]


def require_nonzero_endpoint_spread(
        pred_endpoint_xyz: NDArray[np.float64],
        gt_endpoint_xyz: NDArray[np.float64],
    ) -> None:
    '''Reject occurrences with an exactly degenerate selected endpoint set.'''
    for name, value in (
            ('prediction', pred_endpoint_xyz),
            ('GT', gt_endpoint_xyz),
        ):
        centered = value - value.mean(axis=1, keepdims=True, dtype=np.float64)
        spread = np.sum(centered * centered, axis=(1, 2), dtype=np.float64)
        if bool((spread <= 0.0).any()):
            raise ValueError('%s endpoint spread must be positive' % name)


def compute_jta_sota_metric_sums(
        pred_joints: NDArray[np.generic],
        gt_joints: NDArray[np.generic],
        pred_occurrence_ids: NDArray[np.generic],
        gt_occurrence_ids: NDArray[np.generic],
        pred_endpoint_xy: NDArray[np.generic],
        gt_endpoint_xy: NDArray[np.generic],
        gt_smpl24_xy: NDArray[np.generic],
        *,
        joint_layout: SMPL_Joint_Layout,
        unit_world: Joint_Unit,
        pred_coordinate_frame: Joint_Coordinate_Frame,
        gt_coordinate_frame: Joint_Coordinate_Frame,
        image_coordinate_frame: str,
    ) -> JTA_SOTA_Metric_Sums:
    '''Compute additive six-metric statistics for one paired batch.'''
    pred, gt, _ = validate_occurrence_inputs(
        pred_joints,
        gt_joints,
        pred_occurrence_ids,
        gt_occurrence_ids,
        joint_layout=joint_layout,
        unit_world=unit_world,
        pred_coordinate_frame=pred_coordinate_frame,
        gt_coordinate_frame=gt_coordinate_frame,
    )
    pred_xy, gt_xy, gt_full_xy = validate_image_metric_inputs(
        pred_endpoint_xy,
        gt_endpoint_xy,
        gt_smpl24_xy,
        pred.shape[0],
        image_coordinate_frame,
    )
    pred_endpoint = pred[:, SMPL54_ENDPOINT_INDICES, :]
    gt_endpoint = gt[:, SMPL54_ENDPOINT_INDICES, :]
    pred_root = pred[:, JTA_SOTA_ROOT_INDICES, :].mean(
        axis=1,
        dtype=np.float64,
    )
    gt_root = gt[:, JTA_SOTA_ROOT_INDICES, :].mean(
        axis=1,
        dtype=np.float64,
    )
    selected = (pred_endpoint, gt_endpoint, pred_root, gt_root)
    if not all(bool(np.isfinite(value).all()) for value in selected):
        raise ValueError('selected prediction/GT joints must be finite')
    require_nonzero_endpoint_spread(pred_endpoint, gt_endpoint)

    pair_valid = np.ones((JTA_SOTA_JOINT_COUNT,), dtype=np.bool_)
    rigid_rows: list[NDArray[np.float64]] = []
    similarity_rows: list[NDArray[np.float64]] = []
    for index in range(pred.shape[0]):
        rigid_fit = fit_rigid_registration(
            pred_endpoint[index],
            gt_endpoint[index],
            pair_valid,
        )
        similarity_fit = fit_similarity_registration(
            pred_endpoint[index],
            gt_endpoint[index],
            pair_valid,
        )
        rigid_rows.append(apply_rigid_registration(pred_endpoint[index], rigid_fit))
        similarity_rows.append(
            apply_similarity_registration(pred_endpoint[index], similarity_fit),
        )
    rigid = np.stack(rigid_rows, axis=0)
    similarity = np.stack(similarity_rows, axis=0)

    gt_span = gt_full_xy.max(axis=1) - gt_full_xy.min(axis=1)
    gt_area = gt_span[:, 0] * gt_span[:, 1]
    oks = compute_paired_keypoint_oks(
        gt_xy,
        pred_xy,
        gt_area,
        JTA_ENDPOINT_OKS_SIGMAS,
    )
    scale_mm = 1000.0 if unit_world == 'm' else 1.0
    root_sum = np.sum(np.linalg.norm(pred_root - gt_root, axis=1), dtype=np.float64)
    mpjpe_sum = np.sum(
        np.linalg.norm(pred_endpoint - gt_endpoint, axis=2),
        dtype=np.float64,
    )
    t_sum = np.sum(
        np.linalg.norm(
            (pred_endpoint - pred_root[:, None, :])
            - (gt_endpoint - gt_root[:, None, :]),
            axis=2,
        ),
        dtype=np.float64,
    )
    rt_sum = np.sum(np.linalg.norm(rigid - gt_endpoint, axis=2), dtype=np.float64)
    pa_sum = np.sum(
        np.linalg.norm(similarity - gt_endpoint, axis=2),
        dtype=np.float64,
    )
    count = int(pred.shape[0])
    return JTA_SOTA_Metric_Sums(
        profile=JTA_SOTA_METRIC_PROFILE,
        joint_coordinate_frame=pred_coordinate_frame,
        image_coordinate_frame=image_coordinate_frame,
        num_occurrence=count,
        num_joint=JTA_SOTA_JOINT_COUNT,
        root_error_sum_mm=float(root_sum) * scale_mm,
        mpjpe_sum_mm=float(mpjpe_sum) * scale_mm,
        t_mpjpe_sum_mm=float(t_sum) * scale_mm,
        rt_mpjpe_sum_mm=float(rt_sum) * scale_mm,
        pa_mpjpe_sum_mm=float(pa_sum) * scale_mm,
        oks_sum=float(np.sum(oks, dtype=np.float64)),
        root_denominator=count,
        joint_denominator=count * JTA_SOTA_JOINT_COUNT,
        oks_denominator=count,
    )


def finalize_jta_sota_metric_sums(
        sums: JTA_SOTA_Metric_Sums,
    ) -> JTA_SOTA_Metric_Result:
    '''Finalize one nonempty additive sums value.'''
    return JTA_SOTA_Metric_Result(
        profile=sums.profile,
        joint_coordinate_frame=sums.joint_coordinate_frame,
        image_coordinate_frame=sums.image_coordinate_frame,
        num_occurrence=sums.num_occurrence,
        num_joint=sums.num_joint,
        root_error_mm=sums.root_error_sum_mm / sums.root_denominator,
        mpjpe_mm=sums.mpjpe_sum_mm / sums.joint_denominator,
        t_mpjpe_mm=sums.t_mpjpe_sum_mm / sums.joint_denominator,
        rt_mpjpe_mm=sums.rt_mpjpe_sum_mm / sums.joint_denominator,
        pa_mpjpe_mm=sums.pa_mpjpe_sum_mm / sums.joint_denominator,
        oks=sums.oks_sum / sums.oks_denominator,
    )


def validate_jta_sota_occurrence_partition(
        expected_ids: NDArray[np.generic],
        batch_occurrence_ids: Sequence[NDArray[np.generic]],
    ) -> None:
    '''Require an ordered batch-ID sequence to equal the expected population.'''
    expected = np.asarray(expected_ids)
    if expected.dtype != np.int64 or expected.ndim != 1 or expected.size <= 0:
        raise ValueError('expected_ids must be nonempty one-dimensional int64')
    if np.unique(expected).size != expected.size:
        raise ValueError('expected_ids must be unique')
    if not batch_occurrence_ids:
        raise ValueError('batch_occurrence_ids must be nonempty')
    batches: list[NDArray[np.int64]] = []
    for index, raw in enumerate(batch_occurrence_ids):
        value = np.asarray(raw)
        if value.dtype != np.int64 or value.ndim != 1 or value.size <= 0:
            raise ValueError(
                'batch_occurrence_ids[%d] must be nonempty one-dimensional int64'
                % index,
            )
        batches.append(np.asarray(value, dtype=np.int64))
    observed = np.concatenate(batches)
    if not np.array_equal(observed, expected):
        raise ValueError('ordered occurrence partition differs from expected IDs')


__all__ = [
    'JTA_SOTA_JOINT_COUNT',
    'JTA_SOTA_METRIC_PROFILE',
    'JTA_SOTA_Metric_Result',
    'JTA_SOTA_Metric_Sums',
    'compute_jta_sota_metric_sums',
    'finalize_jta_sota_metric_sums',
    'validate_jta_sota_occurrence_partition',
]
