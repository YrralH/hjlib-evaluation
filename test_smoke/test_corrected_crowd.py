'''Portable smoke tests for corrected crowd metric facilities.'''
from dataclasses import replace

import numpy as np
import pytest

from hjlib_evaluation import (
    C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
    CORRECTED_CROWD_METRICS,
    CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
    CORRECTED_CROWD_SCHEMA_VERSION,
    Corrected_Crowd_Selected_View_Result,
    Corrected_Crowd_Selected_View_Sequence_Summary,
    Corrected_Crowd_Sequence,
    compute_joint_acceleration_errors,
    compute_joint_jerk_errors,
    compute_pcod_3class_matches,
    compute_ppds_scores,
    corrected_crowd_selected_view_result_from_json,
    corrected_crowd_selected_view_result_to_json,
    corrected_crowd_selected_view_summary_from_json,
    corrected_crowd_selected_view_summary_to_json,
    corrected_crowd_summary_from_json,
    corrected_crowd_summary_to_json,
    corrected_crowd_world_dynamics_result_from_json,
    corrected_crowd_world_dynamics_result_to_json,
    corrected_crowd_world_dynamics_summary_from_json,
    corrected_crowd_world_dynamics_summary_to_json,
    evaluate_corrected_crowd_selected_view_and_world_dynamics,
    evaluate_corrected_crowd_world_dynamics,
    evaluate_corrected_crowd_selected_view,
    evaluate_corrected_crowd_sequence,
    make_coco17_visible_ge9_common_mask,
    reduce_corrected_crowd_selected_view_summaries,
    reduce_corrected_crowd_summaries,
    reduce_corrected_crowd_world_dynamics_summaries,
)


def make_sequence(scene_id: str = 'scene_a') -> Corrected_Crowd_Sequence:
    '''Build a two-person temporal scene plus one visibility-excluded case.'''
    gt_frame_ids = np.array([0, 1, 2, 4, 0, 1, 2, 0], dtype=np.int64)
    gt_track_ids = np.array([1, 1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
    joint_template = np.stack([
        np.array([index % 3, index // 3, (index % 5) * 0.2], dtype=np.float64)
        for index in range(24)
    ]) * 0.05
    gt_joints = np.empty((8, 24, 3), dtype=np.float64)
    for row, (frame_id, track_id) in enumerate(
        zip(gt_frame_ids, gt_track_ids, strict=True),
    ):
        pelvis = np.array([
            float(track_id * 2),
            float(frame_id * frame_id) * 0.1,
            5.0 + float(track_id),
        ])
        gt_joints[row] = joint_template + pelvis
    rotation = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    matched_prediction = (
        1.2 * (gt_joints[:7] @ rotation.T)
        + np.array([1.0, 2.0, 0.5])
    )
    pred_joints = np.concatenate([
        matched_prediction,
        np.stack([gt_joints[7], gt_joints[0]]),
    ])
    gt_xy = np.zeros((8, 17, 2), dtype=np.float64)
    gt_xy[..., 0] = np.arange(17, dtype=np.float64)
    gt_xy[..., 1] = np.arange(17, dtype=np.float64) * 0.5
    pred_xy = np.concatenate([gt_xy[:7], gt_xy[7:8], gt_xy[:1]])
    visibility = np.ones((8, 17), dtype=np.float64)
    visibility[:, ::2] = 0.5
    visibility[7] = 0.0
    bboxes = np.tile(np.array([0.0, 0.0, 100.0, 200.0]), (8, 1))
    return Corrected_Crowd_Sequence(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id=scene_id,
        frame_domain=np.arange(5, dtype=np.int64),
        gt_frame_ids=gt_frame_ids,
        gt_track_ids=gt_track_ids,
        gt_joints_world_m=gt_joints,
        gt_coco17_xy_px=gt_xy,
        gt_visibility_native=visibility,
        gt_bbox_xyxy_px=bboxes,
        gt_pelvis_camera_depth_m=gt_joints[:, 0, 2],
        prediction_frame_ids=np.array([0, 1, 2, 4, 0, 1, 2, 0, 0], dtype=np.int64),
        prediction_local_track_ids=np.array(
            [10, 10, 10, 10, 20, 20, 20, 30, 99],
            dtype=np.int64,
        ),
        prediction_joints_world_m=pred_joints,
        prediction_coco17_xy_px=pred_xy,
        prediction_coco17_camera_depth_m=np.full((9, 17), 5.0),
        prediction_pelvis_camera_depth_m=pred_joints[:, 0, 2],
        prediction_identity_target_gt_rows=np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, -1],
            dtype=np.int64,
        ),
        matched_gt_rows=np.arange(7, dtype=np.int64),
        matched_prediction_rows=np.arange(7, dtype=np.int64),
        common_gt_mask=np.array(
            [True, True, True, True, True, False, True, False],
        ),
    )


def test_full_protocol_partition_views_and_round_trip() -> None:
    sequence = make_sequence()
    summary = evaluate_corrected_crowd_sequence(sequence)
    result = reduce_corrected_crowd_summaries([summary])

    assert (result.tp, result.fn, result.fp) == (7, 0, 1)
    assert result.accel_exact_consecutive_triple_count == (2, 1)
    assert all(value is not None for value in result.metric_values[0])
    pa_index = CORRECTED_CROWD_METRICS.index('PA-MPJPE')
    oks_index = CORRECTED_CROWD_METRICS.index('OKS-VIS')
    assert result.metric_values[0][pa_index] == pytest.approx(0.0, abs=1.0e-10)
    assert result.metric_values[0][oks_index] == pytest.approx(1.0)
    restored = corrected_crowd_summary_from_json(
        corrected_crowd_summary_to_json(summary),
    )
    np.testing.assert_array_equal(
        restored.metric_sample_counts,
        summary.metric_sample_counts,
    )
    np.testing.assert_allclose(
        restored.metric_sample_sums,
        summary.metric_sample_sums,
    )
    assert not summary.metric_sample_sums.flags.writeable


def test_visibility_exclusion_and_invalid_projection_contract() -> None:
    sequence = make_sequence()
    assert evaluate_corrected_crowd_sequence(sequence).fp == 1
    invalid_depth = np.array(sequence.prediction_coco17_camera_depth_m, copy=True)
    invalid_depth[0, 0] = 0.0
    with pytest.raises(ValueError, match='projection depth'):
        replace(sequence, prediction_coco17_camera_depth_m=invalid_depth)


def test_pair_and_pcod_boundaries() -> None:
    assert compute_ppds_scores(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )[0] == 0.0
    matches = compute_pcod_3class_matches(
        np.array([0.0, 0.3, 0.600001]),
        np.array([0.0, 0.3, 0.6]),
        0.3,
    )
    np.testing.assert_array_equal(matches, [True, True, False])
    assert compute_ppds_scores(np.empty((1, 3)), np.empty((1, 3))).shape == (0,)


def test_acceleration_uses_vector_residual() -> None:
    reference = np.zeros((3, 1, 3), dtype=np.float64)
    predicted = np.zeros_like(reference)
    predicted[2, 0] = np.array([1.0, 1.0, 0.0])
    error = compute_joint_acceleration_errors(predicted, reference)
    assert error[0, 0] == pytest.approx(np.sqrt(2.0))


def test_jerk_uses_vector_residual_and_cubic_difference() -> None:
    reference = np.zeros((4, 1, 3), dtype=np.float64)
    predicted = np.zeros_like(reference)
    time = np.arange(4, dtype=np.float64)
    predicted[:, 0, 0] = time ** 3
    predicted[:, 0, 1] = time ** 3
    error = compute_joint_jerk_errors(predicted, reference)
    assert error.shape == (1, 1)
    assert error[0, 0] == pytest.approx(6.0 * np.sqrt(2.0))


def make_dynamics_sequence() -> Corrected_Crowd_Sequence:
    '''Build one seven-frame matched track for dynamics protocol tests.'''
    source = make_sequence('dynamics_scene')
    gt_frame_ids = np.array(source.gt_frame_ids, copy=True)
    gt_frame_ids[:7] = np.arange(7, dtype=np.int64)
    gt_track_ids = np.array(source.gt_track_ids, copy=True)
    gt_track_ids[:7] = 1
    prediction_frame_ids = np.array(source.prediction_frame_ids, copy=True)
    prediction_frame_ids[:7] = np.arange(7, dtype=np.int64)
    prediction_local_track_ids = np.array(
        source.prediction_local_track_ids,
        copy=True,
    )
    prediction_local_track_ids[:7] = 10
    common = np.zeros((8,), dtype=np.bool_)
    common[:7] = True
    return replace(
        source,
        frame_domain=np.arange(7, dtype=np.int64),
        gt_frame_ids=gt_frame_ids,
        gt_track_ids=gt_track_ids,
        prediction_frame_ids=prediction_frame_ids,
        prediction_local_track_ids=prediction_local_track_ids,
        common_gt_mask=common,
    )


def test_world_dynamics_parity_counts_gaps_and_round_trip() -> None:
    sequence = make_dynamics_sequence()
    selected_mask = np.array(sequence.common_gt_mask, copy=True)
    legacy, dynamics = evaluate_corrected_crowd_selected_view_and_world_dynamics(
        sequence,
        C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
        selected_mask,
    )
    accel_index = CORRECTED_CROWD_METRICS.index('ACCEL-WORLD')
    np.testing.assert_equal(
        dynamics.metric_sample_sums[0],
        legacy.metric_sample_sums[accel_index],
    )
    assert dynamics.metric_sample_counts.tolist() == [120, 5, 96, 4]
    assert dynamics.accel_exact_consecutive_triple_count == 5
    assert dynamics.jerk_exact_consecutive_quadruple_count == 4
    restored = corrected_crowd_world_dynamics_summary_from_json(
        corrected_crowd_world_dynamics_summary_to_json(dynamics),
    )
    np.testing.assert_array_equal(
        restored.metric_sample_counts,
        dynamics.metric_sample_counts,
    )
    result = reduce_corrected_crowd_world_dynamics_summaries([dynamics])
    assert all(value is not None for value in result.metric_values)
    assert corrected_crowd_world_dynamics_result_from_json(
        corrected_crowd_world_dynamics_result_to_json(result),
    ) == result

    selected_mask[3] = False
    split = evaluate_corrected_crowd_world_dynamics(
        sequence,
        C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
        selected_mask,
    )
    assert split.metric_sample_counts.tolist() == [48, 2, 0, 0]
    split_result = reduce_corrected_crowd_world_dynamics_summaries([split])
    assert split_result.metric_values[2:] == (None, None)


def test_world_dynamics_rejects_mask_and_count_drift() -> None:
    sequence = make_dynamics_sequence()
    with pytest.raises(TypeError, match='boolean'):
        evaluate_corrected_crowd_world_dynamics(
            sequence,
            C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
            np.ones((8,), dtype=np.int64),
        )
    summary = evaluate_corrected_crowd_world_dynamics(
        sequence,
        C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
        sequence.common_gt_mask,
    )
    raw = corrected_crowd_world_dynamics_summary_to_json(summary)
    raw['metric_sample_counts'][0] += 1
    with pytest.raises(ValueError, match='exact-window support'):
        corrected_crowd_world_dynamics_summary_from_json(raw)
    with pytest.raises(ValueError, match='matched exact-window support'):
        replace(summary, matched_selected_count=0)
    impossible_counts = np.array([24, 1, 24, 1], dtype=np.int64)
    with pytest.raises(ValueError, match='smaller than triple'):
        replace(
            summary,
            matched_selected_count=4,
            metric_sample_counts=impossible_counts,
            accel_exact_consecutive_triple_count=1,
            jerk_exact_consecutive_quadruple_count=1,
        )
    result = reduce_corrected_crowd_world_dynamics_summaries([summary])
    with pytest.raises(ValueError, match='matched exact-window support'):
        replace(result, matched_selected_count=0)


def test_cross_scene_reduction_is_micro_and_order_independent() -> None:
    first = evaluate_corrected_crowd_sequence(make_sequence('z_scene'))
    second = evaluate_corrected_crowd_sequence(make_sequence('a_scene'))
    result = reduce_corrected_crowd_summaries([first, second])
    assert result == reduce_corrected_crowd_summaries([second, first])
    assert (result.tp, result.fn, result.fp) == (14, 0, 2)


def test_selected_view_matches_legacy_common_and_round_trips() -> None:
    sequence = make_sequence()
    legacy = evaluate_corrected_crowd_sequence(sequence)
    selected = evaluate_corrected_crowd_selected_view(
        sequence,
        C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
        sequence.common_gt_mask,
    )
    np.testing.assert_array_equal(
        selected.metric_sample_sums,
        legacy.metric_sample_sums[1],
    )
    np.testing.assert_array_equal(
        selected.metric_sample_counts,
        legacy.metric_sample_counts[1],
    )
    assert selected.accel_exact_consecutive_triple_count == int(
        legacy.accel_exact_consecutive_triple_count[1],
    )
    restored_summary = corrected_crowd_selected_view_summary_from_json(
        corrected_crowd_selected_view_summary_to_json(selected),
    )
    assert restored_summary.scene_id == selected.scene_id
    assert restored_summary.view_name == selected.view_name
    np.testing.assert_array_equal(
        restored_summary.metric_sample_sums,
        selected.metric_sample_sums,
    )
    np.testing.assert_array_equal(
        restored_summary.metric_sample_counts,
        selected.metric_sample_counts,
    )
    wrong_counts = np.array(selected.metric_sample_counts, copy=True)
    wrong_counts[0] += 1
    with pytest.raises(ValueError, match='joint metric counts'):
        replace(selected, metric_sample_counts=wrong_counts)
    wrong_pair_counts = np.array(selected.metric_sample_counts, copy=True)
    wrong_pair_counts[10] += 1
    with pytest.raises(ValueError, match='pair metric counts'):
        replace(selected, metric_sample_counts=wrong_pair_counts)
    result = reduce_corrected_crowd_selected_view_summaries([selected])
    restored_result = corrected_crowd_selected_view_result_from_json(
        corrected_crowd_selected_view_result_to_json(result),
    )
    assert restored_result == result
    with pytest.raises(ValueError, match='matched exact-window support'):
        replace(result, matched_selected_count=0)
    wrong_pair_values = list(result.metric_values)
    wrong_pair_values[10] = None
    with pytest.raises(ValueError, match='pair metric availability'):
        replace(result, metric_values=tuple(wrong_pair_values))


def test_visible_ge9_counts_half_visibility_and_preserves_selection_holes() -> None:
    sequence = make_sequence()
    visibility = np.zeros((8, 17), dtype=np.float64)
    visibility[:, :8] = 1.0
    visibility[0, 8] = 0.5
    visibility[2, 8] = 1.0
    old_common = np.ones((8,), dtype=np.bool_)
    selected_mask = make_coco17_visible_ge9_common_mask(visibility, old_common)
    np.testing.assert_array_equal(
        np.flatnonzero(selected_mask),
        np.array([0, 2], dtype=np.int64),
    )
    selected = evaluate_corrected_crowd_selected_view(
        sequence,
        C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
        selected_mask,
    )
    assert selected.selected_gt_count == 2
    assert selected.matched_selected_count == 2
    assert selected.accel_exact_consecutive_triple_count == 0


@pytest.mark.parametrize('reserved_name', ['GT_VISIBLE', 'C4D_DYCROWD_COMMON'])
def test_selected_view_constructors_reject_legacy_names(reserved_name: str) -> None:
    with pytest.raises(ValueError, match='reserved'):
        Corrected_Crowd_Selected_View_Sequence_Summary(
            schema_version=CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
            scene_id='scene_a',
            view_name=reserved_name,
            selected_gt_count=0,
            matched_selected_count=0,
            metric_sample_sums=np.zeros((len(CORRECTED_CROWD_METRICS),)),
            metric_sample_counts=np.zeros(
                (len(CORRECTED_CROWD_METRICS),),
                dtype=np.int64,
            ),
            accel_exact_consecutive_triple_count=0,
        )
    with pytest.raises(ValueError, match='reserved'):
        Corrected_Crowd_Selected_View_Result(
            schema_version=CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
            view_name=reserved_name,
            selected_gt_count=0,
            matched_selected_count=0,
            metric_values=(None,) * len(CORRECTED_CROWD_METRICS),
            accel_exact_consecutive_triple_count=0,
        )


def smoke_test_corrected_crowd() -> None:
    '''Run the full portable corrected-crowd smoke surface.'''
    test_full_protocol_partition_views_and_round_trip()
    test_visibility_exclusion_and_invalid_projection_contract()
    test_pair_and_pcod_boundaries()
    test_acceleration_uses_vector_residual()
    test_jerk_uses_vector_residual_and_cubic_difference()
    test_world_dynamics_parity_counts_gaps_and_round_trip()
    test_world_dynamics_rejects_mask_and_count_drift()
    test_cross_scene_reduction_is_micro_and_order_independent()
    test_selected_view_matches_legacy_common_and_round_trips()
    test_visible_ge9_counts_half_visibility_and_preserves_selection_holes()
    test_selected_view_constructors_reject_legacy_names('GT_VISIBLE')
    test_selected_view_constructors_reject_legacy_names('C4D_DYCROWD_COMMON')


if __name__ == '__main__':
    smoke_test_corrected_crowd()
