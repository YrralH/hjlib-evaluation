'''Portable coverage for tracked-scene ground-estimation evaluation.'''

import numpy as np
import pytest

from hjlib_detection import Tracked_Person, Tracked_Scene, frame_indices_to_ranges
from hjlib_evaluation import (
    Ground_Effect_Support,
    Ground_Observation_Set,
    collect_ground_observations,
    compute_ground_effect_decomposition,
    compute_ground_plane_diagnostics,
    compute_same_ray_ground_errors,
    estimate_ground_from_observations,
    lower_weighted_median,
    sample_ground_observations,
    select_ground_observations_at_frame,
    summarize_ground_errors,
)


def make_keypoint_row(offset: float, score: float = 4.0) -> np.ndarray:
    row = np.zeros((133, 3), dtype=np.float32)
    row[:, 2] = score
    row[5, :2] = [10.0 + offset, 20.0]
    row[6, :2] = [14.0 + offset, 20.0]
    row[15, :2] = [11.0 + offset, 60.0]
    row[16, :2] = [13.0 + offset, 60.0]
    return row


def make_person(
        person_id: int,
        frame_indices: list[int],
        keypoints: np.ndarray,
        present: list[bool],
        num_frame: int = 3,
    ) -> Tracked_Person:
    mask = np.asarray(present, dtype=np.bool_)
    keypoints_stored = keypoints.copy()
    keypoints_stored[~mask] = np.nan
    bboxes = np.tile(
        np.array([[0.0, 100.0, 0.0, 100.0, 1.0]], dtype=np.float32),
        (len(frame_indices), 1),
    )
    return Tracked_Person(
        person_id=person_id,
        list_ranges=frame_indices_to_ranges(frame_indices),
        bboxes=bboxes,
        keypoints=keypoints_stored,
        keypoints_mask=mask,
        num_frame_scene=num_frame,
    )


def make_scene() -> Tracked_Scene:
    person0 = make_person(
        0,
        [0, 1, 2],
        np.stack([
            make_keypoint_row(0.0),
            make_keypoint_row(1.0),
            make_keypoint_row(2.0),
        ]),
        [True, False, True],
    )
    person1_rows = np.stack([
        make_keypoint_row(10.0),
        make_keypoint_row(12.0),
    ])
    person1_rows[0, 5, 2] = 3.0
    person1 = make_person(1, [0, 2], person1_rows, [True, True])
    return Tracked_Scene(
        num_frame=3,
        source_person_axis_size=2,
        has_bboxes=True,
        keypoint_shape=(133, 3),
        persons=(person0, person1),
    )


def test_collect_and_first_frame_selection() -> None:
    observations = collect_ground_observations(
        make_scene(),
        (5, 6),
        (15, 16),
        3.0,
    )
    assert observations.count == 3
    assert list(zip(
        observations.frame_indices.tolist(),
        observations.person_ids.tolist(),
        strict=True,
    )) == [(0, 0), (2, 0), (2, 1)]
    np.testing.assert_array_equal(observations.quality, [4.0, 4.0, 4.0])
    np.testing.assert_array_equal(observations.top_xy_px[0], [12.0, 20.0])
    np.testing.assert_array_equal(observations.bottom_xy_px[0], [12.0, 60.0])
    assert not observations.top_xy_px.flags.writeable

    first = select_ground_observations_at_frame(observations, 0)
    assert first.count == 1
    assert first.person_ids.tolist() == [0]


def test_collect_rejects_degenerate_high_confidence_row() -> None:
    row = make_keypoint_row(0.0)
    row[[15, 16], :2] = row[[5, 6], :2]
    person = make_person(0, [0], row[None], [True], num_frame=1)
    scene = Tracked_Scene(1, 1, True, (133, 3), (person,))
    with pytest.raises(ValueError, match='invalid'):
        collect_ground_observations(scene, (5, 6), (15, 16), 3.0)

    low_confidence = make_keypoint_row(0.0, score=1.0)
    low_confidence[[15, 16], :2] = low_confidence[[5, 6], :2]
    low_person = make_person(0, [0], low_confidence[None], [True], num_frame=1)
    with pytest.raises(ValueError, match='present'):
        collect_ground_observations(
            Tracked_Scene(1, 1, True, (133, 3), (low_person,)),
            (5, 6),
            (15, 16),
            4.0,
            maximum_bottom_pair_bbox_width_ratio=0.20,
        )


def test_strict_bottom_pair_bbox_ratio_filter_and_propagation() -> None:
    rows = np.stack([
        make_keypoint_row(0.0, 4.5),
        make_keypoint_row(1.0, 4.5),
        make_keypoint_row(2.0, 4.0),
    ])
    rows[1, 15, 0] = 0.0
    rows[1, 16, 0] = 30.0
    person = make_person(0, [0, 1, 2], rows, [True, True, True])
    scene = Tracked_Scene(3, 1, True, (133, 3), (person,))
    observations = collect_ground_observations(
        scene,
        (5, 6),
        (15, 16),
        4.0,
        maximum_bottom_pair_bbox_width_ratio=0.20,
    )
    assert observations.count == 1
    assert observations.frame_indices.tolist() == [0]
    assert observations.bottom_pair_bbox_width_ratio is not None
    np.testing.assert_allclose(observations.bottom_pair_bbox_width_ratio, [0.02])
    selected = select_ground_observations_at_frame(observations, 0)
    assert selected.bottom_pair_bbox_width_ratio is not None
    np.testing.assert_array_equal(
        selected.bottom_pair_bbox_width_ratio,
        observations.bottom_pair_bbox_width_ratio,
    )

    invalid_bbox = person.bboxes.copy() if person.bboxes is not None else None
    assert invalid_bbox is not None
    invalid_bbox[0, 3] = invalid_bbox[0, 2]
    invalid_person = Tracked_Person(
        person.person_id,
        person.list_ranges,
        invalid_bbox,
        person.keypoints,
        person.keypoints_mask,
        person.num_frame_scene,
    )
    with pytest.raises(ValueError, match='bbox-ratio'):
        collect_ground_observations(
            Tracked_Scene(3, 1, True, (133, 3), (invalid_person,)),
            (5, 6),
            (15, 16),
            4.0,
            maximum_bottom_pair_bbox_width_ratio=0.20,
        )


def make_large_observations(count: int = 6_000) -> Ground_Observation_Set:
    frames = np.arange(count, dtype=np.int64) // 30
    persons = np.arange(count, dtype=np.int64) % 30
    top = np.column_stack([
        np.arange(count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
    ])
    bottom = top + np.array([0.0, 10.0])
    return Ground_Observation_Set(
        frames,
        persons,
        top,
        bottom,
        np.full(count, 4.0, dtype=np.float64),
    )


def test_capped_sampling_is_deterministic_person_frame_sampling() -> None:
    observations = make_large_observations()
    selected = sample_ground_observations(observations, 5_000, 17)
    repeated = sample_ground_observations(observations, 5_000, 17)
    changed = sample_ground_observations(observations, 5_000, 18)
    assert selected.count == 5_000
    np.testing.assert_array_equal(selected.frame_indices, repeated.frame_indices)
    np.testing.assert_array_equal(selected.person_ids, repeated.person_ids)
    assert not np.array_equal(selected.top_xy_px, changed.top_xy_px)
    assert len(set(selected.person_ids.tolist())) == 30

    all_rows = sample_ground_observations(observations, 7_000, 19)
    assert all_rows.count == observations.count
    np.testing.assert_array_equal(all_rows.top_xy_px, observations.top_xy_px)


def test_estimator_seam_and_result_contract() -> None:
    observations = make_large_observations(6)
    captured: list[tuple[int, tuple[int, int]]] = []

    def estimator(
            top: np.ndarray,
            bottom: np.ndarray,
            K: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        captured.append((top.shape[0], K.shape))
        assert bottom.shape == top.shape
        return (
            np.array([0.0, 0.0, 1.0, -2.0], dtype=np.float64),
            np.array(0.25, dtype=np.float64),
        )

    result = estimate_ground_from_observations(
        observations,
        np.eye(3, dtype=np.float64),
        estimator,
    )
    assert captured == [(6, (3, 3))]
    assert result.objective == 0.25
    assert not result.plane_camera_abcd.flags.writeable


def make_support() -> Ground_Effect_Support:
    return Ground_Effect_Support(
        np.array([0, 0], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]], dtype=np.float64),
    )


def test_same_ray_errors_and_summary_in_metres() -> None:
    support = make_support()
    errors = compute_same_ray_ground_errors(
        support,
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 1.0, -3.0], dtype=np.float64),
    )
    np.testing.assert_allclose(errors, [1.0, np.sqrt(2.0)])
    summary = summarize_ground_errors(errors)
    assert summary['count'] == 2
    assert summary['mean_m'] == pytest.approx((1.0 + np.sqrt(2.0)) / 2.0)
    assert summary['std_m'] == pytest.approx((np.sqrt(2.0) - 1.0) / 2.0)


def test_plane_and_ground_effect_decomposition() -> None:
    diagnostics = compute_ground_plane_diagnostics(
        np.array([0.0, 0.0, -2.0, 6.0], dtype=np.float64),
        np.array([0.0, 0.0, 2.0, -4.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        diagnostics.normalized_pred_plane_camera_abcd,
        [0.0, 0.0, 1.0, -3.0],
    )
    np.testing.assert_allclose(
        diagnostics.normalized_gt_plane_camera_abcd,
        [0.0, 0.0, 1.0, -2.0],
    )
    assert diagnostics.normal_angle_deg == 0.0
    assert diagnostics.distance_ratio == 1.5
    assert not diagnostics.normalized_pred_plane_camera_abcd.flags.writeable

    decomposition = compute_ground_effect_decomposition(
        make_support(),
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, -2.0, 6.0], dtype=np.float64),
        np.array([0.0, 0.0, 2.0, -4.0], dtype=np.float64),
    )
    assert decomposition.oracle_distance_m == -2.0
    np.testing.assert_allclose(decomposition.normal_oracle_error_m, [0.0, 0.0])
    np.testing.assert_allclose(
        decomposition.distance_only_error_m,
        [1.0, np.sqrt(2.0)],
    )
    assert not decomposition.normal_oracle_error_m.flags.writeable

    tilted = compute_ground_effect_decomposition(
        make_support(),
        np.eye(3, dtype=np.float64),
        np.array([1.0, 0.0, 1.0, -3.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0, -2.0], dtype=np.float64),
    )
    np.testing.assert_allclose(tilted.oracle_distance_m, -np.sqrt(2.0))
    np.testing.assert_allclose(
        tilted.normal_oracle_error_m,
        [0.0, np.sqrt(2.0)],
    )
    distance_offset = 3.0 / np.sqrt(2.0) - 2.0
    np.testing.assert_allclose(
        tilted.distance_only_error_m,
        [distance_offset, np.sqrt(2.0) * distance_offset],
    )


def test_lower_weighted_median_is_canonical_and_fail_closed() -> None:
    values = np.array([2.0, 1.0, 2.0, 3.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 1.0, 4.0], dtype=np.float64)
    assert lower_weighted_median(values, weights) == 2.0
    with pytest.raises(ValueError, match='finite'):
        lower_weighted_median(
            np.array([0.0, np.nan], dtype=np.float64),
            np.ones(2, dtype=np.float64),
        )


def test_support_current_K_and_prior_method_mismatch_fail() -> None:
    support = make_support()
    K_wrong = np.eye(3, dtype=np.float64)
    K_wrong[0, 0] = 2.0
    with pytest.raises(ValueError, match='current K'):
        compute_same_ray_ground_errors(
            support,
            K_wrong,
            np.array([0.0, 0.0, 1.0, -3.0], dtype=np.float64),
        )


def smoke_test_ground_estimation_protocol() -> None:
    test_collect_and_first_frame_selection()
    test_collect_rejects_degenerate_high_confidence_row()
    test_strict_bottom_pair_bbox_ratio_filter_and_propagation()
    test_capped_sampling_is_deterministic_person_frame_sampling()
    test_estimator_seam_and_result_contract()
    test_same_ray_errors_and_summary_in_metres()
    test_plane_and_ground_effect_decomposition()
    test_lower_weighted_median_is_canonical_and_fail_closed()
    test_support_current_K_and_prior_method_mismatch_fail()


if __name__ == '__main__':
    smoke_test_ground_estimation_protocol()
