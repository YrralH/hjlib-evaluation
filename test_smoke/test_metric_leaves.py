'''Data-free contracts for method-neutral OKS and joint-error leaves.'''

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import hjlib_evaluation
from hjlib_evaluation import (
    compute_joint_position_errors,
    compute_keypoint_oks_matrix,
)


def test_joint_error_known_values_dtype_and_non_finite_policy() -> None:
    target = np.array([[[3, 4, 0], [0, 0, 0]]], dtype=np.int64)
    reference = np.zeros((1, 2, 3), dtype=np.float32)
    errors = compute_joint_position_errors(target, reference)
    assert errors.dtype == np.float64
    assert np.array_equal(errors, np.array([[5.0, 0.0]]))

    non_finite = compute_joint_position_errors(
        np.array([[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]]),
        np.zeros((2, 3)),
    )
    assert np.isnan(non_finite[0])
    assert np.isinf(non_finite[1])


def test_joint_error_invalid_inputs_fail() -> None:
    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: compute_joint_position_errors(
            np.zeros((2, 3)),
            np.zeros((3, 3)),
        ),
        lambda: compute_joint_position_errors(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        ),
        lambda: compute_joint_position_errors(
            np.array([['x', 'y', 'z']]),
            np.zeros((1, 3)),
        ),
        lambda: compute_joint_position_errors(
            np.array([[1.0 + 1.0j, 0.0, 0.0]]),
            np.zeros((1, 3)),
        ),
    )
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_oks_known_value_mask_and_empty_axes() -> None:
    reference = np.array([[[0.0, 0.0], [10.0, 10.0]]])
    target = np.array(
        [
            [[0.0, 0.0], [100.0, 100.0]],
            [[2.0, 0.0], [10.0, 10.0]],
        ]
    )
    oks = compute_keypoint_oks_matrix(
        reference,
        target,
        np.array([2.0]),
        np.array([0.5, 0.5]),
        np.array([[True, False]]),
    )
    assert oks.dtype == np.float64
    assert oks.shape == (1, 2)
    assert oks[0, 0] == 1.0
    assert oks[0, 1] == np.exp(-1.0)

    no_valid = compute_keypoint_oks_matrix(
        reference,
        target,
        np.array([2.0]),
        np.array([0.5, 0.5]),
        np.array([[False, False]]),
    )
    assert np.array_equal(no_valid, np.zeros((1, 2)))

    empty_reference = compute_keypoint_oks_matrix(
        np.zeros((0, 2, 2)),
        target,
        np.zeros((0,)),
        np.array([0.5, 0.5]),
        np.zeros((0, 2), dtype=np.bool_),
    )
    empty_target = compute_keypoint_oks_matrix(
        reference,
        np.zeros((0, 2, 2)),
        np.array([2.0]),
        np.array([0.5, 0.5]),
        np.array([[True, True]]),
    )
    assert empty_reference.shape == (0, 2)
    assert empty_target.shape == (1, 0)


@pytest.mark.parametrize('bad_positive', [0.0, -1.0, np.nan, np.inf])
def test_oks_positive_finite_contract(bad_positive: float) -> None:
    points = np.zeros((1, 1, 2))
    valid = np.ones((1, 1), dtype=np.bool_)
    with pytest.raises(ValueError):
        compute_keypoint_oks_matrix(
            points,
            points,
            np.array([bad_positive]),
            np.array([1.0]),
            valid,
        )
    with pytest.raises(ValueError):
        compute_keypoint_oks_matrix(
            points,
            points,
            np.array([1.0]),
            np.array([bad_positive]),
            valid,
        )


def test_oks_invalid_shapes_and_mask_dtype_fail() -> None:
    points = np.zeros((1, 2, 2))
    calls: tuple[Callable[[], object], ...] = (
        lambda: compute_keypoint_oks_matrix(
            points,
            np.zeros((1, 3, 2)),
            np.ones(1),
            np.ones(2),
            np.ones((1, 2), dtype=np.bool_),
        ),
        lambda: compute_keypoint_oks_matrix(
            points,
            points,
            np.ones(2),
            np.ones(2),
            np.ones((1, 2), dtype=np.bool_),
        ),
        lambda: compute_keypoint_oks_matrix(
            points,
            points,
            np.ones(1),
            np.ones(2),
            cast(Any, np.ones((1, 2), dtype=np.int64)),
        ),
    )
    for call in calls:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_top_level_exports() -> None:
    assert hjlib_evaluation.compute_joint_position_errors is compute_joint_position_errors
    assert hjlib_evaluation.compute_keypoint_oks_matrix is compute_keypoint_oks_matrix


def smoke_test_metric_leaves() -> None:
    test_joint_error_known_values_dtype_and_non_finite_policy()
    test_joint_error_invalid_inputs_fail()
    test_oks_known_value_mask_and_empty_axes()
    for bad_positive in (0.0, -1.0, np.nan, np.inf):
        test_oks_positive_finite_contract(bad_positive)
    test_oks_invalid_shapes_and_mask_dtype_fail()
    test_top_level_exports()
