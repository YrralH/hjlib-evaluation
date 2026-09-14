'''Data-free contracts for method-neutral OKS and joint-error leaves.'''

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import hjlib_evaluation
from hjlib_evaluation import (
    compute_joint_height_errors,
    compute_pa_joint_position_errors,
    compute_joint_position_errors,
    compute_keypoint_oks_matrix,
)
from hjlib_evaluation.keypoint_oks import make_positive_depth_joint_mask


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


def test_pa_joint_error_similarity_reflection_and_input_contract() -> None:
    target = np.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ]])
    angle = np.deg2rad(37.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    reference = 2.5 * (target @ rotation.T) + np.array([4.0, -3.0, 2.0])
    errors = compute_pa_joint_position_errors(target, reference)
    world_errors = compute_joint_position_errors(target, reference)
    translation_aligned_errors = compute_joint_position_errors(
        target - target[:, :1],
        reference - reference[:, :1],
    )
    assert errors.shape == (1, 4)
    assert float(np.mean(world_errors)) > 0.0
    assert float(np.mean(translation_aligned_errors)) > 0.0
    assert float(np.max(errors)) <= 1e-10 * float(
        max(1.0, np.max(np.abs(reference))))

    reflected = np.array(target, copy=True)
    reflected[..., 0] *= -1.0
    reflected_errors = compute_pa_joint_position_errors(reflected, target)
    assert float(np.max(reflected_errors)) > 1e-6

    empty = compute_pa_joint_position_errors(
        np.empty((0, 24, 3)),
        np.empty((0, 24, 3)),
    )
    assert empty.shape == (0, 24) and empty.dtype == np.float64
    with pytest.raises(ValueError, match='finite'):
        compute_pa_joint_position_errors(
            np.full((1, 24, 3), np.nan),
            np.zeros((1, 24, 3)),
        )
    with pytest.raises(ValueError, match='spread'):
        compute_pa_joint_position_errors(
            np.zeros((1, 24, 3)),
            np.zeros((1, 24, 3)),
        )


def test_joint_height_error_normalization_sign_and_leading_normals() -> None:
    target = np.array([[[0.0, 0.0, 2.0], [3.0, 4.0, 0.0]]])
    reference = np.zeros_like(target)
    unit = compute_joint_height_errors(
        target, reference, np.array([0.0, 0.0, 1.0]))
    scaled = compute_joint_height_errors(
        target, reference, np.array([0.0, 0.0, 0.1607]))
    flipped = compute_joint_height_errors(
        target, reference, np.array([0.0, 0.0, -1.0]))
    assert unit.dtype == np.float64
    assert np.array_equal(unit, np.array([[2.0, 0.0]]))
    assert np.allclose(scaled, unit)
    assert np.array_equal(flipped, unit)

    two_frames = np.concatenate((target, target), axis=0)
    leading = compute_joint_height_errors(
        two_frames,
        np.zeros_like(two_frames),
        np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
    )
    assert np.array_equal(leading, np.array([[2.0, 0.0], [0.0, 4.0]]))


def test_joint_height_error_invalid_normal_fails() -> None:
    points = np.zeros((2, 12, 3))
    invalid_normals = (
        np.zeros(3),
        np.array([0.0, 0.0, 1e-12]),
        np.array([0.0, 0.0, 1e-13]),
        np.array([0.0, np.nan, 1.0]),
        np.zeros((3, 3)),
        np.array(['x', 'y', 'z']),
        np.array([1.0 + 1.0j, 0.0, 0.0]),
    )
    for normal in invalid_normals:
        with pytest.raises((TypeError, ValueError)):
            compute_joint_height_errors(points, points, normal)


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


def test_positive_depth_joint_mask_contract() -> None:
    visible = np.array([[True, True, False], [True, False, True]])
    depth = np.array([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]])
    assert np.array_equal(
        make_positive_depth_joint_mask(visible, depth),
        np.array([[True, False, False], [True, False, True]]),
    )

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: make_positive_depth_joint_mask(
            cast(Any, visible.astype(np.int64)), depth),
        lambda: make_positive_depth_joint_mask(visible, depth[:, :2]),
        lambda: make_positive_depth_joint_mask(
            visible, np.full_like(depth, np.nan)),
    )
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_top_level_exports() -> None:
    assert hjlib_evaluation.compute_joint_height_errors is compute_joint_height_errors
    assert hjlib_evaluation.compute_pa_joint_position_errors is compute_pa_joint_position_errors
    assert hjlib_evaluation.compute_joint_position_errors is compute_joint_position_errors
    assert hjlib_evaluation.compute_keypoint_oks_matrix is compute_keypoint_oks_matrix


def smoke_test_metric_leaves() -> None:
    test_joint_error_known_values_dtype_and_non_finite_policy()
    test_joint_error_invalid_inputs_fail()
    test_pa_joint_error_similarity_reflection_and_input_contract()
    test_joint_height_error_normalization_sign_and_leading_normals()
    test_joint_height_error_invalid_normal_fails()
    test_oks_known_value_mask_and_empty_axes()
    for bad_positive in (0.0, -1.0, np.nan, np.inf):
        test_oks_positive_finite_contract(bad_positive)
    test_oks_invalid_shapes_and_mask_dtype_fail()
    test_positive_depth_joint_mask_contract()
    test_top_level_exports()
