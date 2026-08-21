'''Named population selections for corrected VirtualCrowd evaluation.'''
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import Bool_Array, bool_array


C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9 = (
    'C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9'
)


def make_coco17_visible_ge9_common_mask(
    visibility_native: NDArray[np.generic],
    old_common_mask: NDArray[np.generic],
) -> Bool_Array:
    '''Select old-common GT rows with at least nine source-visible COCO joints.'''
    visibility = np.asarray(visibility_native)
    if not np.issubdtype(visibility.dtype, np.number):
        raise TypeError('visibility_native must have numeric dtype')
    if np.issubdtype(visibility.dtype, np.complexfloating):
        raise TypeError('visibility_native must have real numeric dtype')
    visibility = np.asarray(visibility, dtype=np.float64)
    if visibility.ndim != 2 or visibility.shape[1] != 17:
        raise ValueError('visibility_native must have shape [G,17]')
    if not np.isfinite(visibility).all():
        raise ValueError('visibility_native must be finite')
    if not np.isin(visibility, (0.0, 0.5, 1.0)).all():
        raise ValueError('visibility_native values must be 0, 0.5, or 1')
    common = bool_array(old_common_mask, 'old_common_mask')
    if common.shape != (len(visibility),):
        raise ValueError('old_common_mask must have shape [G]')
    base_visible = np.any(visibility > 0.0, axis=1)
    if np.any(common & ~base_visible):
        raise ValueError('old_common_mask must be a subset of GT-visible rows')
    return bool_array(
        common & (np.count_nonzero(visibility > 0.0, axis=1) >= 9),
        'selected_common_mask',
    )
