'''Unreduced pelvis-centered and proper-rigid local SMPL22 joint errors.'''

import numpy as np
from numpy.typing import NDArray

from hjlib_geometry import apply_rigid_registration, fit_rigid_registration
from hjlib_evaluation.joint_error import (
    compute_joint_position_errors,
    validate_joint_points,
)


SMPL22_JOINT_COUNT = 22
RIGID_RANK_RELATIVE_TOLERANCE = 1e-10
RIGID_POINT_SINGULAR_VALUE_FLOOR_M = 1e-12
RIGID_COVARIANCE_SINGULAR_VALUE_FLOOR_M2 = 1e-24


def validate_local_smpl22_joint_pair(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Select SMPL joints0..21 before finite checks and require equal frames.'''
    selected: list[NDArray[np.float64]] = []
    for value, name in (
            (predicted_joints, 'predicted_joints'),
            (reference_joints, 'reference_joints'),
        ):
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] != 3 \
                or array.shape[1] < SMPL22_JOINT_COUNT:
            raise ValueError('%s must have shape (N, J>=22, 3)' % name)
        if array.shape[0] <= 0:
            raise ValueError('%s must have a positive frame count' % name)
        with np.errstate(over='raise', invalid='raise'):
            try:
                points = validate_joint_points(
                    array[:, :SMPL22_JOINT_COUNT], name)
            except FloatingPointError as err:
                raise ValueError('%s float64 normalization overflowed' % name) from err
        if not np.isfinite(points).all():
            raise ValueError('%s selected SMPL22 joints must be finite' % name)
        selected.append(points)
    predicted, reference = selected
    if predicted.shape[0] != reference.shape[0]:
        raise ValueError('predicted_joints and reference_joints must have equal frame counts')
    return predicted, reference


def validate_local_smpl22_rigid_support(
        predicted: NDArray[np.float64],
        reference: NDArray[np.float64],
        frame_index: int,
    ) -> None:
    '''Require rank>=2 point sets and cross-covariance for one rigid fit.'''
    try:
        with np.errstate(over='raise', invalid='raise'):
            predicted_centered = predicted - predicted.mean(axis=0)
            reference_centered = reference - reference.mean(axis=0)
            covariance = predicted_centered.T @ reference_centered
            supports = (
                (predicted_centered, 'predicted points', RIGID_POINT_SINGULAR_VALUE_FLOOR_M),
                (reference_centered, 'reference points', RIGID_POINT_SINGULAR_VALUE_FLOOR_M),
                (covariance, 'cross-covariance', RIGID_COVARIANCE_SINGULAR_VALUE_FLOOR_M2),
            )
            for support, name, absolute_floor in supports:
                singular_values = np.linalg.svd(support, compute_uv=False)
                threshold = max(
                    float(singular_values[0]) * RIGID_RANK_RELATIVE_TOLERANCE,
                    absolute_floor,
                )
                if not np.isfinite(singular_values).all() \
                        or singular_values[1] <= threshold:
                    raise ValueError(
                        'RT-MPJPE frame %s %s must have rank >= 2 at the stated tolerance'
                        % (frame_index, name))
    except (FloatingPointError, np.linalg.LinAlgError) as err:
        raise ValueError('RT-MPJPE frame %s support validation failed' % frame_index) from err


def immutable_local_smpl22_error_values(
        errors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
    '''Freeze finite per-joint metre errors without changing their population.'''
    if not np.isfinite(errors).all():
        raise ValueError('local SMPL22 errors must be finite')
    return np.frombuffer(errors.tobytes(), dtype=np.float64).reshape(errors.shape)


def compute_local_smpl22_t_mpjpe_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return read-only `(N,22)` metre errors after independent pelvis centering.

    Both inputs are world-metre SMPL-ordered `(N,J>=22,3)` arrays. Joint0
    is the pelvis and remains in the returned population. Extra joints may be
    nonfinite and never affect selection, alignment or reduction.
    '''
    predicted, reference = validate_local_smpl22_joint_pair(
        predicted_joints, reference_joints)
    try:
        with np.errstate(over='raise', invalid='raise'):
            errors = compute_joint_position_errors(
                predicted - predicted[:, :1],
                reference - reference[:, :1],
            )
    except FloatingPointError as err:
        raise ValueError('T-MPJPE computation overflowed') from err
    return immutable_local_smpl22_error_values(errors)


def compute_local_smpl22_rt_mpjpe_values(
        predicted_joints: NDArray[np.generic],
        reference_joints: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return read-only `(N,22)` metre errors after per-frame proper rigid fits.

    Fit all22 selected joints with rotation and translation only. No scale,
    reflection or point dropping is allowed. Planar support is valid; rank<2
    point sets or cross-covariance fail at the documented numerical tolerance.
    '''
    predicted, reference = validate_local_smpl22_joint_pair(
        predicted_joints, reference_joints)
    aligned = np.empty_like(predicted)
    mask = np.ones(SMPL22_JOINT_COUNT, dtype=np.bool_)
    for frame_index, (predicted_frame, reference_frame) in enumerate(
            zip(predicted, reference, strict=True),
        ):
        validate_local_smpl22_rigid_support(
            predicted_frame, reference_frame, frame_index)
        fit = fit_rigid_registration(predicted_frame, reference_frame, mask)
        aligned[frame_index] = apply_rigid_registration(predicted_frame, fit)
    try:
        with np.errstate(over='raise', invalid='raise'):
            errors = compute_joint_position_errors(aligned, reference)
    except FloatingPointError as err:
        raise ValueError('RT-MPJPE computation overflowed') from err
    return immutable_local_smpl22_error_values(errors)
