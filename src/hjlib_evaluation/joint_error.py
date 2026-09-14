'''Method-neutral unreduced per-joint position and height errors.'''

import numpy as np
from numpy.typing import NDArray

from hjlib_geometry import (
    apply_similarity_registration,
    fit_similarity_registration,
)


GROUND_NORMAL_MIN_NORM = 1e-12


def compute_joint_position_errors(
        target_points: NDArray[np.generic],
        reference_points: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return unreduced Euclidean error for equal `(..., J, 3)` arrays.'''
    target = validate_joint_points(target_points, 'target_points')
    reference = validate_joint_points(reference_points, 'reference_points')
    if target.shape != reference.shape:
        raise ValueError(
            'target_points and reference_points must have equal shape')
    return np.linalg.norm(target - reference, axis=-1)


def compute_pa_joint_position_errors(
        target_points: NDArray[np.generic],
        reference_points: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return per-occurrence similarity-aligned joint errors for `(N,J,3)`.'''
    target = validate_joint_points(target_points, 'target_points')
    reference = validate_joint_points(reference_points, 'reference_points')
    if target.shape != reference.shape:
        raise ValueError(
            'target_points and reference_points must have equal shape')
    if target.ndim != 3:
        raise ValueError('PA joint points must have shape (N, J, 3)')
    if target.shape[1] <= 0:
        raise ValueError('PA joint points must have a positive joint count')
    if not np.isfinite(target).all() or not np.isfinite(reference).all():
        raise ValueError('PA joint points must be finite')
    if target.shape[0] == 0:
        return np.empty(target.shape[:2], dtype=np.float64)
    mask = np.ones(target.shape[1], dtype=np.bool_)
    aligned: list[NDArray[np.float64]] = []
    for target_occurrence, reference_occurrence in zip(
            target, reference, strict=True,
        ):
        fit = fit_similarity_registration(
            target_occurrence,
            reference_occurrence,
            mask,
        )
        aligned.append(apply_similarity_registration(target_occurrence, fit))
    return compute_joint_position_errors(
        np.stack(aligned, axis=0),
        reference,
    )


def compute_joint_height_errors(
        target_points: NDArray[np.generic],
        reference_points: NDArray[np.generic],
        ground_normal: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return absolute per-joint error along normalized ground normals.

    `ground_normal` is either one `(3,)` normal or one normal per leading
    point-array position, shaped `(..., 3)` for points `(..., J, 3)`.
    '''
    target = validate_joint_points(target_points, 'target_points')
    reference = validate_joint_points(reference_points, 'reference_points')
    if target.shape != reference.shape:
        raise ValueError('target_points and reference_points must have equal shape')
    normal = validate_ground_normal(ground_normal, target.shape[:-2])
    if normal.ndim > 1:
        normal = np.expand_dims(normal, axis=-2)
    return np.abs(np.sum((target - reference) * normal, axis=-1))


def validate_ground_normal(
        value: NDArray[np.generic],
        leading_shape: tuple[int, ...],
    ) -> NDArray[np.float64]:
    '''Validate and normalize one shared or leading-position ground normal.'''
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise TypeError('ground_normal must be a real numeric array')
    if array.shape not in ((3,), (*leading_shape, 3)):
        raise ValueError('ground_normal shape must be (3,) or (..., 3)')
    normal = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(normal, axis=-1, keepdims=True)
    if not bool(np.isfinite(normal).all()) \
            or not bool(np.isfinite(norms).all()) \
            or bool(np.any(norms <= GROUND_NORMAL_MIN_NORM)):
        raise ValueError('ground_normal must be finite and nondegenerate')
    return normal / norms


def validate_joint_points(
        value: NDArray[np.generic],
        name: str,
    ) -> NDArray[np.float64]:
    '''Validate one real numeric joint array and normalize it to float64.'''
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise TypeError('%s must be a real numeric array' % name)
    if array.ndim < 2 or array.shape[-1] != 3:
        raise ValueError('%s must have shape (..., J, 3)' % name)
    return np.asarray(array, dtype=np.float64)
