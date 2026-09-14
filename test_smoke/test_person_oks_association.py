'''Executable protocol cases for per-frame OKS association.'''
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from hjlib_evaluation.person_oks_association import (
    COCO17_OKS_SIGMAS,
    CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    Person_OKS_Association,
    Person_OKS_Matching_Profile,
    STANDARD_MATCHING_PROFILE,
    match_coco17_people,
    solve_cardinality_quality_assignment,
)


EXPECTED_COCO17_OKS_SIGMAS = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
    0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
], dtype=np.float64) / 10.0


def test_standard_threshold_is_inclusive() -> None:
    gt, pred, boxes, camera = one_person_inputs()
    with patch(
            'hjlib_evaluation.person_oks_association.'
            'compute_keypoint_oks_matrix',
            return_value=np.array([[0.5]], dtype=np.float64)):
        result = match_coco17_people(gt, pred, boxes, camera)
    assert result.match_gt_indices.tolist() == [0]
    assert result.match_prediction_indices.tolist() == [0]


def test_standard_maximizes_cardinality_before_quality() -> None:
    oks = np.array([[0.9, 0.8], [0.8, 0.0]], dtype=np.float64)
    result = match_with_oks(oks)
    assert result.match_gt_indices.tolist() == [0, 1]
    assert result.match_prediction_indices.tolist() == [1, 0]


def test_standard_maximizes_quality_after_cardinality() -> None:
    oks = np.array([[0.9, 0.8], [0.8, 0.6]], dtype=np.float64)
    result = match_with_oks(oks)
    assert result.match_prediction_indices.tolist() == [1, 0]
    np.testing.assert_allclose(result.matched_oks, [0.8, 0.8])


def test_standard_tie_follows_fixed_ordered_hungarian_result() -> None:
    oks = np.full((2, 2), 0.7, dtype=np.float64)
    result = match_with_oks(oks)
    assert result.match_prediction_indices.tolist() == [0, 1]


def test_standard_partial_match_preserves_canonical_gt_order() -> None:
    oks = np.array([
        [0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9],
    ], dtype=np.float64)
    result = match_with_oks(oks)
    assert result.match_gt_indices.tolist() == [0, 2]
    assert result.match_prediction_indices.tolist() == [0, 2]
    assert result.matched_oks.tolist() == [0.8, 0.9]
    assert result.false_negative_gt_indices.tolist() == [1]
    assert result.false_positive_prediction_indices.tolist() == [1]


def test_author_greedy_drops_collision_without_second_choice() -> None:
    gt = np.zeros((2, 17, 3), dtype=np.float64)
    gt[:, :, 2] = 1.0
    pred = np.zeros((2, 17, 3), dtype=np.float64)
    pred[:, :, 2] = 1.0
    boxes = np.tile(np.array([0.0, 0.0, 10.0, 10.0]), (2, 1))
    oks = np.array([[0.9, 0.8], [0.85, 0.1]], dtype=np.float64)
    with patch(
            'hjlib_evaluation.person_oks_association.'
            'compute_keypoint_oks_matrix', return_value=oks):
        result = match_coco17_people(
            gt, pred, boxes, np.eye(3),
            CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
        )
    assert result.match_gt_indices.tolist() == [0]
    assert result.false_negative_gt_indices.tolist() == [1]
    assert result.false_positive_prediction_indices.tolist() == [1]


def test_author_threshold_is_strict() -> None:
    gt, pred, boxes, camera = one_person_inputs()
    with patch(
            'hjlib_evaluation.person_oks_association.'
            'compute_keypoint_oks_matrix',
            return_value=np.array([[1e-6]], dtype=np.float64)):
        result = match_coco17_people(
            gt, pred, boxes, camera,
            CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
        )
    assert result.match_gt_indices.size == 0
    assert result.false_negative_gt_indices.tolist() == [0]
    assert result.false_positive_prediction_indices.tolist() == [0]


def test_author_exact_best_prediction_tie_uses_first_column() -> None:
    result = match_with_oks(
        np.array([[0.9, 0.9]], dtype=np.float64),
        CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    )
    assert result.match_prediction_indices.tolist() == [0]
    assert result.false_positive_prediction_indices.tolist() == [1]


def test_author_projection_invalid_prediction_remains_false_positive() -> None:
    gt, prediction, boxes, camera = one_person_inputs()
    prediction[:, :, 2] = 0.0
    result = match_coco17_people(
        gt, prediction, boxes, camera,
        CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    )
    assert result.match_gt_indices.size == 0
    assert result.false_negative_gt_indices.tolist() == [0]
    assert result.false_positive_prediction_indices.tolist() == [0]
    assert not result.projection_valid_predictions[0]


def test_author_oks_uses_all_joints_without_visibility_filtering() -> None:
    gt, prediction, boxes, camera = one_person_inputs()
    gt[:, 1:, 2] = 0.0
    prediction[:, 1:, :2] = 1.0
    standard = match_coco17_people(gt, prediction, boxes, camera)
    author = match_coco17_people(
        gt, prediction, boxes, camera,
        CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    )
    squared_distance = np.zeros(17, dtype=np.float64)
    squared_distance[1:] = 2.0
    expected = np.exp(
        -squared_distance
        / ((2.0 * EXPECTED_COCO17_OKS_SIGMAS) ** 2 * 100.0 * 2.0),
    ).mean()
    np.testing.assert_array_equal(
        COCO17_OKS_SIGMAS, EXPECTED_COCO17_OKS_SIGMAS,
    )
    np.testing.assert_allclose(standard.matched_oks, [1.0])
    np.testing.assert_allclose(author.matched_oks, [expected])
    assert author.matched_oks[0] < standard.matched_oks[0]


def test_author_oks_clamps_degenerate_bbox_area() -> None:
    gt, prediction, boxes, camera = one_person_inputs()
    boxes[:] = 0.0
    prediction[0, :, 0] = (
        2.0 * EXPECTED_COCO17_OKS_SIGMAS * np.sqrt(2.0 * np.spacing(1))
    )
    author = match_coco17_people(
        gt, prediction, boxes, camera,
        CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    )
    np.testing.assert_allclose(author.matched_oks, [np.exp(-1.0)])
    try:
        match_coco17_people(gt, prediction, boxes, camera)
    except ValueError as error:
        assert 'positive GT bbox area' in str(error)
    else:
        raise AssertionError('standard profile accepted a degenerate bbox')


def test_author_preserves_release_best_distance_process_order() -> None:
    oks = np.array([
        [0.8, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.9],
    ], dtype=np.float64)
    result = match_with_oks(
        oks, CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    )
    assert result.match_gt_indices.tolist() == [2, 0]
    assert result.match_prediction_indices.tolist() == [2, 0]
    assert result.matched_oks.tolist() == [0.9, 0.8]
    assert result.false_negative_gt_indices.tolist() == [1]
    assert result.false_positive_prediction_indices.tolist() == [1]


def test_empty_and_invalid_projection_form_full_partitions() -> None:
    empty_gt = np.empty((0, 17, 3), dtype=np.float64)
    prediction = np.zeros((2, 17, 3), dtype=np.float64)
    prediction[:, :, 2] = -1.0
    result = match_coco17_people(
        empty_gt, prediction, np.empty((0, 4)), np.eye(3),
    )
    assert result.false_negative_gt_indices.size == 0
    assert result.false_positive_prediction_indices.tolist() == [0, 1]
    assert not result.projection_valid_predictions.any()


def test_nonempty_gt_cannot_match_projection_invalid_prediction() -> None:
    gt, pred, boxes, camera = one_person_inputs()
    pred[:, :, 2] = 0.0
    result = match_coco17_people(gt, pred, boxes, camera)
    assert result.match_gt_indices.size == 0
    assert result.false_negative_gt_indices.tolist() == [0]
    assert result.false_positive_prediction_indices.tolist() == [0]
    assert result.solver_call_count == 1


def test_standard_makes_zero_visible_gt_a_false_negative() -> None:
    gt, pred, boxes, camera = one_person_inputs()
    gt[:, :, 2] = 0.0
    result = match_coco17_people(gt, pred, boxes, camera)
    assert result.match_gt_indices.size == 0
    assert result.false_negative_gt_indices.tolist() == [0]
    assert result.false_positive_prediction_indices.tolist() == [0]
    assert result.solver_call_count == 0


def test_standard_zero_visible_rows_do_not_change_supported_tie() -> None:
    gt = np.zeros((4, 17, 3), dtype=np.float64)
    gt[[1, 3], :, 2] = 1.0
    pred = np.zeros((2, 17, 3), dtype=np.float64)
    pred[:, :, 2] = 1.0
    boxes = np.tile(np.array([0.0, 0.0, 10.0, 10.0]), (4, 1))
    with patch(
            'hjlib_evaluation.person_oks_association.'
            'compute_keypoint_oks_matrix',
            return_value=np.full((2, 2), 0.7, dtype=np.float64),
        ) as compute:
        result = match_coco17_people(gt, pred, boxes, np.eye(3))
    assert compute.call_args.args[0].shape == (2, 17, 2)
    assert result.match_gt_indices.tolist() == [1, 3]
    assert result.match_prediction_indices.tolist() == [0, 1]
    assert result.false_negative_gt_indices.tolist() == [0, 2]
    assert result.false_positive_prediction_indices.size == 0


def test_empty_profiles_report_zero_solver_calls() -> None:
    for profile in (
            STANDARD_MATCHING_PROFILE,
            CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
        ):
        empty_gt = np.empty((0, 17, 3), dtype=np.float64)
        prediction = np.ones((2, 17, 3), dtype=np.float64)
        result = match_coco17_people(
            empty_gt, prediction, np.empty((0, 4)), np.eye(3), profile,
        )
        assert result.solver_call_count == 0
        gt, _, boxes, camera = one_person_inputs()
        result = match_coco17_people(
            gt, np.empty((0, 17, 3)), boxes, camera, profile,
        )
        assert result.solver_call_count == 0


def test_public_dispatch_rejects_unsupported_profile() -> None:
    gt, pred, boxes, camera = one_person_inputs()
    try:
        match_coco17_people(
            gt, pred, boxes, camera, cast(Any, 'unknown-profile'),
        )
    except ValueError as error:
        assert 'unsupported person OKS matching profile' in str(error)
    else:
        raise AssertionError('unsupported matching profile was accepted')


def test_association_rejects_fractional_indices_before_normalization() -> None:
    try:
        Person_OKS_Association(
            match_gt_indices=cast(Any, np.array([0.5])),
            match_prediction_indices=np.array([0], dtype=np.int64),
            matched_oks=np.array([1.0]),
            false_negative_gt_indices=np.empty(0, dtype=np.int64),
            false_positive_prediction_indices=np.empty(0, dtype=np.int64),
            projection_valid_predictions=np.ones(1, dtype=np.bool_),
        )
    except TypeError as error:
        assert 'match_gt_indices must have integer dtype' in str(error)
    else:
        raise AssertionError('fractional association index was normalized')


def test_association_rejects_nonboolean_projection_mask() -> None:
    try:
        Person_OKS_Association(
            match_gt_indices=np.array([0], dtype=np.int64),
            match_prediction_indices=np.array([0], dtype=np.int64),
            matched_oks=np.array([1.0]),
            false_negative_gt_indices=np.empty(0, dtype=np.int64),
            false_positive_prediction_indices=np.empty(0, dtype=np.int64),
            projection_valid_predictions=cast(Any, np.array([np.nan])),
        )
    except TypeError as error:
        assert 'projection_valid_predictions must have bool dtype' in str(error)
    else:
        raise AssertionError('nonboolean projection mask was normalized')


def test_assignment_primitive_rejects_undefined_domain() -> None:
    invalid_inputs = (
        (np.ones((1, 1), dtype=np.int64), np.ones((1, 1), dtype=np.int64)),
        (np.ones((1, 2), dtype=np.bool_), np.ones((1, 1), dtype=np.int64)),
        (np.ones((1, 1), dtype=np.bool_), np.array([[-1]], dtype=np.int64)),
    )
    for admissible, quality in invalid_inputs:
        try:
            solve_cardinality_quality_assignment(
                cast(Any, admissible), quality,
            )
        except ValueError as error:
            assert 'objective arrays' in str(error)
        else:
            raise AssertionError('undefined assignment domain was accepted')


def test_dense_association_returns_disjoint_complete_partition() -> None:
    generator = np.random.default_rng(7)
    oks = generator.random((64, 80))
    result = match_with_oks(oks)
    assert len(np.unique(result.match_gt_indices)) == len(
        result.match_gt_indices)
    assert len(np.unique(result.match_prediction_indices)) == len(
        result.match_prediction_indices)
    assert sorted(
        result.match_gt_indices.tolist()
        + result.false_negative_gt_indices.tolist()
    ) == list(range(64))
    assert sorted(
        result.match_prediction_indices.tolist()
        + result.false_positive_prediction_indices.tolist()
    ) == list(range(80))


def match_with_oks(
        oks: np.ndarray,
        matching_profile: Person_OKS_Matching_Profile = \
            STANDARD_MATCHING_PROFILE,
    ) -> Person_OKS_Association:
    gt_count, prediction_count = oks.shape
    gt = np.zeros((gt_count, 17, 3), dtype=np.float64)
    gt[:, :, 2] = 1.0
    prediction = np.ones((prediction_count, 17, 3), dtype=np.float64)
    boxes = np.tile(
        np.array([0.0, 0.0, 10.0, 10.0]), (gt_count, 1),
    )
    with patch(
            'hjlib_evaluation.person_oks_association.'
            'compute_keypoint_oks_matrix', return_value=oks):
        return match_coco17_people(
            gt, prediction, boxes, np.eye(3), matching_profile,
        )


def one_person_inputs() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    ]:
    gt = np.zeros((1, 17, 3), dtype=np.float64)
    gt[:, :, 2] = 1.0
    pred = np.zeros((1, 17, 3), dtype=np.float64)
    pred[:, :, 2] = 1.0
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float64)
    return gt, pred, boxes, np.eye(3)
