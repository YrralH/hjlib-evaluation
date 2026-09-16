'''Pure metric leaves for the LSV-HR standard evaluation profile.'''
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.crowd_layout import compute_ppds_scores
from hjlib_evaluation.joint_acceleration import (
    compute_central_acceleration_magnitudes,
    compute_joint_acceleration_errors,
)
from hjlib_evaluation.joint_error import (
    compute_joint_position_errors,
    validate_joint_points,
)
from hjlib_evaluation.keypoint_oks import (
    COCO17_OKS_SIGMAS,
    compute_paired_keypoint_oks,
    make_positive_depth_joint_mask,
)
from hjlib_geometry import (
    apply_mean_translation,
    apply_rigid_registration,
    fit_mean_translation,
    fit_rigid_registration,
    fit_similarity_registration,
)


def readonly_metric_values(
        value: NDArray[np.float64],
    ) -> NDArray[np.float64]:
    '''Mark one newly computed metric population as read-only.'''
    output = np.asarray(value, dtype=np.float64)
    output.setflags(write=False)
    return output


def validate_paired_smpl24(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Normalize one non-empty pair of finite `(N,24,3)` tensors.'''
    predicted = validate_joint_points(predicted_joints, 'predicted_joints')
    reference = validate_joint_points(reference_joints, 'reference_joints')
    if predicted.ndim != 3 or predicted.shape[1:] != (24, 3):
        raise ValueError('predicted_joints must have shape (N, 24, 3)')
    if reference.shape != predicted.shape:
        raise ValueError('reference_joints must match predicted_joints shape')
    if len(predicted) == 0:
        raise ValueError('paired SMPL24 population must be non-empty')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError('paired SMPL24 population must be finite')
    return predicted, reference


def compute_standard_mpjpe_world_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return per-occurrence, per-joint world-position errors in metres.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    return readonly_metric_values(
        compute_joint_position_errors(predicted, reference),
    )


def compute_standard_t_mpjpe_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return pelvis-translation-aligned errors in metres.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    predicted_local = predicted - predicted[:, :1]
    reference_local = reference - reference[:, :1]
    return readonly_metric_values(
        compute_joint_position_errors(predicted_local, reference_local),
    )


def compute_standard_rt_mpjpe_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return one rigid-fit error population per person occurrence.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    aligned = np.empty_like(predicted)
    mask = np.ones(24, dtype=np.bool_)
    for row in range(len(predicted)):
        fit = fit_rigid_registration(predicted[row], reference[row], mask)
        aligned[row] = apply_rigid_registration(predicted[row], fit)
    return readonly_metric_values(
        compute_joint_position_errors(aligned, reference),
    )


def compute_standard_sequence_t_mpjpe_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return errors after one translation fit over the complete input scope.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    predicted_points = predicted.reshape(-1, 3)
    reference_points = reference.reshape(-1, 3)
    fit = fit_mean_translation(
        predicted_points,
        reference_points,
        np.ones(len(predicted_points), dtype=np.bool_),
    )
    aligned = apply_mean_translation(predicted, fit)
    return readonly_metric_values(
        compute_joint_position_errors(aligned, reference),
    )


def compute_standard_sequence_rt_mpjpe_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return errors after one rigid fit over the complete input scope.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    predicted_points = predicted.reshape(-1, 3)
    reference_points = reference.reshape(-1, 3)
    fit = fit_rigid_registration(
        predicted_points,
        reference_points,
        np.ones(len(predicted_points), dtype=np.bool_),
    )
    aligned = apply_rigid_registration(predicted, fit)
    return readonly_metric_values(
        compute_joint_position_errors(aligned, reference),
    )


def compute_standard_rte_world_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return per-occurrence pelvis world-position error in metres.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    return readonly_metric_values(
        np.linalg.norm(predicted[:, 0] - reference[:, 0], axis=1),
    )


def compute_standard_acc_root_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return pelvis acceleration-vector residuals in metres per frame squared.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    return readonly_metric_values(
        compute_joint_acceleration_errors(
            predicted[:, :1],
            reference[:, :1],
        ).reshape(-1),
    )


def central_acceleration_magnitudes(
        joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return twice-central-difference magnitudes after three-sample trimming.'''
    values = validate_joint_points(joints, 'joints')
    if values.ndim != 3 or values.shape[1:] != (24, 3):
        raise ValueError('joints must have shape (T, 24, 3)')
    return readonly_metric_values(compute_central_acceleration_magnitudes(values))


def compute_standard_acc_root_ratio_magnitudes(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Return independent root-ratio numerator/denominator populations.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    return (
        central_acceleration_magnitudes(predicted)[:, 0],
        central_acceleration_magnitudes(reference)[:, 0],
    )


def compute_standard_acc_joint_ratio_magnitudes(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Return independent SMPL24 ratio numerator/denominator populations.'''
    predicted, reference = validate_paired_smpl24(
        predicted_joints,
        reference_joints,
    )
    return (
        central_acceleration_magnitudes(predicted),
        central_acceleration_magnitudes(reference),
    )


def compute_standard_oks_vis_values(
        predicted_xy_px: NDArray[np.generic],
        reference_xy_px: NDArray[np.generic],
        reference_visibility: NDArray[np.generic],
        predicted_camera_depth_m: NDArray[np.generic],
        reference_bbox_xyxy_px: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return supported visibility- and depth-aware paired COCO17 OKS values.'''
    predicted = np.asarray(predicted_xy_px, dtype=np.float64)
    reference = np.asarray(reference_xy_px, dtype=np.float64)
    visibility = np.asarray(reference_visibility, dtype=np.float64)
    depth = np.asarray(predicted_camera_depth_m, dtype=np.float64)
    bbox = np.asarray(reference_bbox_xyxy_px, dtype=np.float64)
    count = len(predicted)
    if predicted.shape != (count, 17, 2) or reference.shape != predicted.shape:
        raise ValueError('paired COCO17 points must have shape (N, 17, 2)')
    if visibility.shape != (count, 17):
        raise ValueError('reference_visibility must have shape (N, 17)')
    if depth.shape != (count, 17):
        raise ValueError('predicted_camera_depth_m must have shape (N, 17)')
    if bbox.shape != (count, 4):
        raise ValueError('reference_bbox_xyxy_px must have shape (N, 4)')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all() \
            or not np.isfinite(visibility).all() \
            or not np.isfinite(depth).all() or not np.isfinite(bbox).all():
        raise ValueError('OKS inputs must be finite')
    valid = make_positive_depth_joint_mask(visibility > 0.0, depth)
    supported = np.any(valid, axis=1)
    if not np.any(supported):
        return readonly_metric_values(np.empty((0,), dtype=np.float64))
    supported_bbox = bbox[supported]
    area = (
        (supported_bbox[:, 2] - supported_bbox[:, 0])
        * (supported_bbox[:, 3] - supported_bbox[:, 1])
    )
    return readonly_metric_values(
        compute_paired_keypoint_oks(
            reference[supported],
            predicted[supported],
            area,
            COCO17_OKS_SIGMAS,
            valid[supported],
        ),
    )


def compute_standard_ppds_values(
        predicted_pelvis: NDArray[np.generic],
        reference_pelvis: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return unordered-pair PPDS values for one frame.'''
    return readonly_metric_values(
        compute_ppds_scores(predicted_pelvis, reference_pelvis),
    )


def compute_standard_pa_ppds_values(
        predicted_pelvis: NDArray[np.generic],
        reference_pelvis: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return PPDS after applying one frame-level similarity-fit scale.'''
    predicted = np.asarray(predicted_pelvis, dtype=np.float64)
    reference = np.asarray(reference_pelvis, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1:] != (3,):
        raise ValueError('predicted_pelvis must have shape (N, 3)')
    if reference.shape != predicted.shape:
        raise ValueError('reference_pelvis must match predicted_pelvis shape')
    if len(predicted) < 2:
        return readonly_metric_values(np.empty((0,), dtype=np.float64))
    fit = fit_similarity_registration(
        predicted,
        reference,
        np.ones(len(predicted), dtype=np.bool_),
    )
    return readonly_metric_values(
        compute_ppds_scores(
            predicted,
            reference,
            fit.scale_target_to_reference,
        ),
    )


__all__ = [
    'compute_standard_acc_root_values',
    'compute_standard_acc_joint_ratio_magnitudes',
    'compute_standard_acc_root_ratio_magnitudes',
    'compute_standard_mpjpe_world_values',
    'compute_standard_oks_vis_values',
    'compute_standard_pa_ppds_values',
    'compute_standard_ppds_values',
    'compute_standard_rt_mpjpe_values',
    'compute_standard_rte_world_values',
    'compute_standard_sequence_rt_mpjpe_values',
    'compute_standard_sequence_t_mpjpe_values',
    'compute_standard_t_mpjpe_values',
]
