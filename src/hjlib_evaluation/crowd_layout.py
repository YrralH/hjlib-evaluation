'''Method-neutral crowd-layout metric leaves.'''

import numpy as np
from numpy.typing import NDArray


def normalize_finite_array(
    value: NDArray[np.generic],
    name: str,
) -> NDArray[np.float64]:
    '''Normalize one real numeric finite array to float64.'''
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError('%s must have numeric dtype' % name)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError('%s must have real numeric dtype' % name)
    output = np.asarray(array, dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError('%s must be finite' % name)
    return output


def unordered_pair_indices(count: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    '''Return canonical `i<j` pair indices.'''
    if type(count) is not int or count < 0:
        raise ValueError('count must be a non-negative exact int')
    left, right = np.triu_indices(count, k=1)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


def compute_ppds_scores(
    predicted_anchors: NDArray[np.generic],
    reference_anchors: NDArray[np.generic],
    scale: float = 1.0,
) -> NDArray[np.float64]:
    '''Return clipped relative pair-distance scores in canonical pair order.'''
    predicted = normalize_finite_array(predicted_anchors, 'predicted_anchors')
    reference = normalize_finite_array(reference_anchors, 'reference_anchors')
    if predicted.ndim != 2 or predicted.shape[1:] != (3,):
        raise ValueError('predicted_anchors must have shape (N, 3)')
    if reference.shape != predicted.shape:
        raise ValueError('reference_anchors must match predicted_anchors shape')
    scale_value = float(scale)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError('scale must be finite and strictly positive')
    left, right = unordered_pair_indices(len(predicted))
    if left.size == 0:
        return np.empty((0,), dtype=np.float64)
    predicted_distance = np.linalg.norm(
        predicted[left] - predicted[right],
        axis=1,
    ) * scale_value
    reference_distance = np.linalg.norm(
        reference[left] - reference[right],
        axis=1,
    )
    if np.any(reference_distance <= 0.0):
        raise ValueError('consumed reference pair distance must be positive')
    return np.maximum(
        0.0,
        1.0 - np.abs(predicted_distance - reference_distance) / reference_distance,
    )


def compute_pcod_3class_matches(
    predicted_depths: NDArray[np.generic],
    reference_depths: NDArray[np.generic],
    tolerance_m: float,
) -> NDArray[np.bool_]:
    '''Return three-class pair-depth matches in canonical pair order.'''
    predicted = normalize_finite_array(predicted_depths, 'predicted_depths')
    reference = normalize_finite_array(reference_depths, 'reference_depths')
    if predicted.ndim != 1:
        raise ValueError('predicted_depths must have shape (N,)')
    if reference.shape != predicted.shape:
        raise ValueError('reference_depths must match predicted_depths shape')
    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('tolerance_m must be finite and non-negative')
    left, right = unordered_pair_indices(len(predicted))
    predicted_delta = predicted[left] - predicted[right]
    reference_delta = reference[left] - reference[right]

    def classify(delta: NDArray[np.float64]) -> NDArray[np.int8]:
        output = np.zeros(delta.shape, dtype=np.int8)
        output[delta < -tolerance] = -1
        output[delta > tolerance] = 1
        return output

    return classify(predicted_delta) == classify(reference_delta)
