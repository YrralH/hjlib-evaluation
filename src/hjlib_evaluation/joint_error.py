'''Method-neutral per-joint Euclidean position errors.'''

import numpy as np
from numpy.typing import NDArray


def compute_joint_position_errors(
        target_points: NDArray[np.generic],
        reference_points: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Return unreduced Euclidean error for equal `(..., J, 3)` arrays.'''
    target = validate_joint_points(target_points, 'target_points')
    reference = validate_joint_points(reference_points, 'reference_points')
    if target.shape != reference.shape:
        raise ValueError('target_points and reference_points must have equal shape')
    return np.linalg.norm(target - reference, axis=-1)


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
