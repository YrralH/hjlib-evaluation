'''Method-neutral paired joint acceleration residuals.'''

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.joint_error import validate_joint_points


def compute_joint_acceleration_errors(
    predicted_joints: NDArray[np.generic],
    reference_joints: NDArray[np.generic],
) -> NDArray[np.float64]:
    '''Return unreduced vector acceleration errors for `(T,J,3)` arrays.'''
    predicted = validate_joint_points(predicted_joints, 'predicted_joints')
    reference = validate_joint_points(reference_joints, 'reference_joints')
    if predicted.ndim != 3:
        raise ValueError('predicted_joints must have shape (T, J, 3)')
    if reference.shape != predicted.shape:
        raise ValueError('reference_joints must match predicted_joints shape')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError('joint acceleration inputs must be finite')
    if predicted.shape[0] < 3:
        return np.empty((0, predicted.shape[1]), dtype=np.float64)
    predicted_acceleration = predicted[2:] - 2.0 * predicted[1:-1] + predicted[:-2]
    reference_acceleration = reference[2:] - 2.0 * reference[1:-1] + reference[:-2]
    return np.linalg.norm(predicted_acceleration - reference_acceleration, axis=-1)
