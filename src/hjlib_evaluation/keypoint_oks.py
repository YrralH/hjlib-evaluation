'''Method-neutral Object Keypoint Similarity matrix computation.'''

import numpy as np
from numpy.typing import NDArray


def make_positive_depth_joint_mask(
        reference_joint_valid: NDArray[np.bool_],
        target_camera_depth: NDArray[np.generic],
    ) -> NDArray[np.bool_]:
    '''Keep reference-valid joints with finite positive paired target depth.'''
    valid = np.asarray(reference_joint_valid)
    if valid.dtype != np.bool_ or valid.ndim != 2:
        raise TypeError('reference_joint_valid must be a two-dimensional bool array')
    depth = validate_real_numeric_array(
        target_camera_depth,
        'target_camera_depth',
        ndim=2,
    )
    if depth.shape != valid.shape:
        raise ValueError('target_camera_depth must match reference_joint_valid shape')
    if not np.isfinite(depth).all():
        raise ValueError('target_camera_depth must be finite')
    return np.asarray(valid & (depth > 0.0), dtype=np.bool_)


def compute_paired_keypoint_oks(
        reference_points_xy: NDArray[np.generic],
        target_points_xy: NDArray[np.generic],
        reference_areas: NDArray[np.generic],
        sigmas: NDArray[np.generic],
        reference_joint_valid: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
    '''Return identity-paired OKS in linear time.'''
    reference = validate_real_numeric_array(
        reference_points_xy,
        'reference_points_xy',
        ndim=3,
    )
    target = validate_real_numeric_array(
        target_points_xy,
        'target_points_xy',
        ndim=3,
    )
    areas = validate_real_numeric_array(reference_areas, 'reference_areas', ndim=1)
    sigma_values = validate_real_numeric_array(sigmas, 'sigmas', ndim=1)
    if reference.shape != target.shape or reference.shape[-1] != 2:
        raise ValueError('paired keypoints must share shape (N, J, 2)')
    count, joint_count = reference.shape[:2]
    if count <= 0 or joint_count <= 0:
        raise ValueError('paired keypoint population must be nonempty')
    if areas.shape != (count,):
        raise ValueError('reference_areas must have shape (%d,)' % count)
    if sigma_values.shape != (joint_count,):
        raise ValueError('sigmas must have shape (%d,)' % joint_count)
    if not np.isfinite(reference).all() or not np.isfinite(target).all():
        raise ValueError('paired keypoints must be finite')
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError('reference_areas must be finite and strictly positive')
    if not np.isfinite(sigma_values).all() or np.any(sigma_values <= 0.0):
        raise ValueError('sigmas must be finite and strictly positive')
    squared_distance = np.sum((reference - target) ** 2, axis=2)
    denominator = (
        (2.0 * sigma_values[None, :]) ** 2
        * areas[:, None]
        * 2.0
    )
    similarity = np.exp(-squared_distance / denominator)
    if reference_joint_valid is None:
        return np.asarray(similarity.mean(axis=1), dtype=np.float64)
    valid = np.asarray(reference_joint_valid)
    if valid.dtype != np.bool_:
        raise TypeError('reference_joint_valid must have bool dtype')
    if valid.shape != (count, joint_count):
        raise ValueError('reference_joint_valid must have shape (N, J)')
    support = valid.sum(axis=1)
    if np.any(support == 0):
        raise ValueError('paired OKS requires at least one valid joint per row')
    return np.asarray(np.where(valid, similarity, 0.0).sum(axis=1) / support,
        dtype=np.float64)


def compute_keypoint_oks_matrix(
        reference_points_xy: NDArray[np.generic],
        target_points_xy: NDArray[np.generic],
        reference_areas: NDArray[np.generic],
        sigmas: NDArray[np.generic],
        reference_joint_valid: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
    '''Return pairwise OKS for reference rows and target columns.

    Bounding-box construction, epsilon policy, association, and aggregation are
    intentionally owned by the caller's evaluation protocol.
    '''
    reference = validate_real_numeric_array(
        reference_points_xy,
        'reference_points_xy',
        ndim=3,
    )
    target = validate_real_numeric_array(
        target_points_xy,
        'target_points_xy',
        ndim=3,
    )
    areas = validate_real_numeric_array(reference_areas, 'reference_areas', ndim=1)
    sigma_values = validate_real_numeric_array(sigmas, 'sigmas', ndim=1)
    valid = np.asarray(reference_joint_valid)

    if reference.shape[-1] != 2 or target.shape[-1] != 2:
        raise ValueError('keypoint coordinates must end in dimension 2')
    if reference.shape[1] != target.shape[1]:
        raise ValueError('reference and target joint counts must match')
    gt_count, joint_count = reference.shape[:2]
    pred_count = target.shape[0]
    if areas.shape != (gt_count,):
        raise ValueError('reference_areas must have shape (%d,)' % gt_count)
    if sigma_values.shape != (joint_count,):
        raise ValueError('sigmas must have shape (%d,)' % joint_count)
    if valid.dtype != np.bool_:
        raise TypeError('reference_joint_valid must have bool dtype')
    if valid.shape != (gt_count, joint_count):
        raise ValueError(
            'reference_joint_valid must have shape (%d, %d)'
            % (gt_count, joint_count)
        )
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('reference_areas must be finite and strictly positive')
    if not np.isfinite(sigma_values).all() or np.any(sigma_values <= 0):
        raise ValueError('sigmas must be finite and strictly positive')

    output = np.zeros((gt_count, pred_count), dtype=np.float64)
    if gt_count == 0 or pred_count == 0:
        return output

    squared_distance = np.sum(
        (
            reference[:, None, :, :]
            - target[None, :, :, :]
        ) ** 2,
        axis=-1,
    )
    denominator = (
        (2.0 * sigma_values[None, None, :]) ** 2
        * areas[:, None, None]
        * 2.0
    )
    similarity = np.exp(-squared_distance / denominator)
    for index_reference in range(gt_count):
        mask = valid[index_reference]
        if mask.any():
            output[index_reference] = similarity[index_reference][:, mask].mean(
                axis=1,
            )
    return output


def validate_real_numeric_array(
        value: NDArray[np.generic],
        name: str,
        *,
        ndim: int,
    ) -> NDArray[np.float64]:
    '''Validate and normalize a real numeric array to float64.'''
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise TypeError('%s must be a real numeric array' % name)
    if array.ndim != ndim:
        raise ValueError('%s must have %d dimensions' % (name, ndim))
    return np.asarray(array, dtype=np.float64)
