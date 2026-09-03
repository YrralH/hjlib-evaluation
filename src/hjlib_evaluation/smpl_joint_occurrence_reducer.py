'''Stateless MPJPE/T-MPJPE reduction over sparse SMPL joint occurrences.'''
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.eval_meta import Metric_Spec_3D


type SMPL_Joint_Layout = Literal['smpl_24', 'smpl_all_54']
type Joint_Coordinate_Frame = Literal['camera', 'world']
type Joint_Unit = Literal['m', 'mm']


@dataclass(frozen=True, slots=True)
class SMPL_Joint_Occurrence_Result:
    '''One occurrence-weighted SMPL joint metric result in millimetres.'''

    metric_name: str
    num_occurrence: int
    num_joint: int
    mpjpe_mm: float
    t_mpjpe_mm: float

    def __post_init__(self) -> None:
        if not self.metric_name:
            raise ValueError('metric_name must be nonempty')
        if type(self.num_occurrence) is not int or self.num_occurrence <= 0:
            raise ValueError('num_occurrence must be a positive exact int')
        if type(self.num_joint) is not int or self.num_joint <= 0:
            raise ValueError('num_joint must be a positive exact int')
        if not math.isfinite(self.mpjpe_mm) or self.mpjpe_mm < 0.0:
            raise ValueError('mpjpe_mm must be finite and nonnegative')
        if not math.isfinite(self.t_mpjpe_mm) or self.t_mpjpe_mm < 0.0:
            raise ValueError('t_mpjpe_mm must be finite and nonnegative')


def require_metric_indices(
    metric: Metric_Spec_3D,
    joint_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    '''Validate and normalize one metric's subject and root joint indices.'''
    normalized: list[tuple[int, ...]] = []
    for name, values in (
        ('joint_indices_smpl_54', metric.joint_indices_smpl_54),
        ('root_indices_smpl_54_for_alignment', metric.root_indices_smpl_54_for_alignment),
    ):
        if not values:
            raise ValueError('%s must be nonempty' % name)
        indices: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError('%s must contain non-bool integers' % name)
            index = int(value)
            if index < 0 or index >= joint_count:
                raise ValueError('%s index is outside the joint layout' % name)
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise ValueError('%s must not contain duplicates' % name)
        normalized.append(tuple(indices))
    return normalized[0], normalized[1]


def validate_occurrence_inputs(
    pred_joints: NDArray[np.generic],
    gt_joints: NDArray[np.generic],
    pred_occurrence_ids: NDArray[np.generic],
    gt_occurrence_ids: NDArray[np.generic],
    *,
    joint_layout: SMPL_Joint_Layout,
    unit_world: Joint_Unit,
    pred_coordinate_frame: Joint_Coordinate_Frame,
    gt_coordinate_frame: Joint_Coordinate_Frame,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    '''Validate pairing/layout declarations and return float64 joint views.'''
    if joint_layout not in ('smpl_24', 'smpl_all_54'):
        raise ValueError('unsupported SMPL joint layout')
    if unit_world not in ('m', 'mm'):
        raise ValueError('unit_world must be m or mm')
    if pred_coordinate_frame not in ('camera', 'world') or gt_coordinate_frame not in (
        'camera', 'world'
    ):
        raise ValueError('coordinate frame must be camera or world')
    if pred_coordinate_frame != gt_coordinate_frame:
        raise ValueError('prediction and GT coordinate frames differ')

    pred = np.asarray(pred_joints)
    gt = np.asarray(gt_joints)
    allowed_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
    if pred.dtype not in allowed_dtypes or gt.dtype not in allowed_dtypes:
        raise TypeError('joint arrays must use float32 or float64')
    expected_joint_count = 24 if joint_layout == 'smpl_24' else 54
    if pred.ndim != 3 or pred.shape[1:] != (expected_joint_count, 3):
        raise ValueError('prediction joints have the wrong layout shape')
    if gt.shape != pred.shape:
        raise ValueError('prediction and GT joint shapes differ')
    if pred.shape[0] <= 0:
        raise ValueError('joint occurrence population is empty')

    pred_ids = np.asarray(pred_occurrence_ids)
    gt_ids = np.asarray(gt_occurrence_ids)
    if pred_ids.dtype != np.int64 or gt_ids.dtype != np.int64:
        raise TypeError('occurrence IDs must use int64')
    expected_id_shape = (pred.shape[0],)
    if pred_ids.shape != expected_id_shape or gt_ids.shape != expected_id_shape:
        raise ValueError('occurrence ID shapes differ from joint rows')
    if np.unique(pred_ids).size != pred_ids.size or np.unique(gt_ids).size != gt_ids.size:
        raise ValueError('occurrence IDs must be unique')
    if not np.array_equal(pred_ids, gt_ids):
        raise ValueError('prediction and GT occurrence IDs differ')
    return (
        np.asarray(pred, dtype=np.float64),
        np.asarray(gt, dtype=np.float64),
        cast(NDArray[np.int64], pred_ids),
    )


def compute_smpl_joint_occurrence_metric(
    pred_joints: NDArray[np.generic],
    gt_joints: NDArray[np.generic],
    pred_occurrence_ids: NDArray[np.generic],
    gt_occurrence_ids: NDArray[np.generic],
    metric: Metric_Spec_3D,
    *,
    joint_layout: SMPL_Joint_Layout,
    unit_world: Joint_Unit,
    pred_coordinate_frame: Joint_Coordinate_Frame,
    gt_coordinate_frame: Joint_Coordinate_Frame,
) -> SMPL_Joint_Occurrence_Result:
    '''Compute one occurrence-weighted MPJPE/T-MPJPE metric.'''
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
    subject_indices, root_indices = require_metric_indices(metric, pred.shape[1])
    pred_subject = pred[:, subject_indices, :]
    gt_subject = gt[:, subject_indices, :]
    pred_root = pred[:, root_indices, :].mean(axis=1, keepdims=True)
    gt_root = gt[:, root_indices, :].mean(axis=1, keepdims=True)
    selected = (pred_subject, gt_subject, pred_root, gt_root)
    if not all(bool(np.isfinite(value).all()) for value in selected):
        raise ValueError('selected prediction/GT joint values must be finite')
    scale_mm = 1000.0 if unit_world == 'm' else 1.0
    mpjpe = np.linalg.norm(pred_subject - gt_subject, axis=2)
    t_mpjpe = np.linalg.norm(
        (pred_subject - pred_root) - (gt_subject - gt_root),
        axis=2,
    )
    mpjpe_mm = float(mpjpe.mean(dtype=np.float64)) * scale_mm
    t_mpjpe_mm = float(t_mpjpe.mean(dtype=np.float64)) * scale_mm
    return SMPL_Joint_Occurrence_Result(
        metric_name=metric.name,
        num_occurrence=int(pred.shape[0]),
        num_joint=len(subject_indices),
        mpjpe_mm=mpjpe_mm,
        t_mpjpe_mm=t_mpjpe_mm,
    )


def reduce_smpl_joint_occurrences(
    pred_joints: NDArray[np.generic],
    gt_joints: NDArray[np.generic],
    pred_occurrence_ids: NDArray[np.generic],
    gt_occurrence_ids: NDArray[np.generic],
    metric_specs: Sequence[Metric_Spec_3D],
    *,
    joint_layout: SMPL_Joint_Layout,
    unit_world: Joint_Unit,
    pred_coordinate_frame: Joint_Coordinate_Frame,
    gt_coordinate_frame: Joint_Coordinate_Frame,
) -> tuple[SMPL_Joint_Occurrence_Result, ...]:
    '''Reduce all metric specs over one paired sparse occurrence population.'''
    if not metric_specs:
        raise ValueError('metric_specs must be nonempty')
    names = [metric.name for metric in metric_specs]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError('metric names must be nonempty and unique')
    return tuple(
        compute_smpl_joint_occurrence_metric(
            pred_joints,
            gt_joints,
            pred_occurrence_ids,
            gt_occurrence_ids,
            metric,
            joint_layout=joint_layout,
            unit_world=unit_world,
            pred_coordinate_frame=pred_coordinate_frame,
            gt_coordinate_frame=gt_coordinate_frame,
        )
        for metric in metric_specs
    )


__all__ = [
    'Joint_Coordinate_Frame',
    'Joint_Unit',
    'SMPL_Joint_Layout',
    'SMPL_Joint_Occurrence_Result',
    'compute_smpl_joint_occurrence_metric',
    'reduce_smpl_joint_occurrences',
    'require_metric_indices',
    'validate_occurrence_inputs',
]
