'''Portable smoke tests for corrected crowd metric facilities.'''
from dataclasses import replace

import numpy as np
import pytest

from hjlib_evaluation import (
    CORRECTED_CROWD_METRICS,
    CORRECTED_CROWD_SCHEMA_VERSION,
    Corrected_Crowd_Sequence,
    compute_joint_acceleration_errors,
    compute_pcod_3class_matches,
    compute_ppds_scores,
    corrected_crowd_summary_from_json,
    corrected_crowd_summary_to_json,
    evaluate_corrected_crowd_sequence,
    reduce_corrected_crowd_summaries,
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


def test_cross_scene_reduction_is_micro_and_order_independent() -> None:
    first = evaluate_corrected_crowd_sequence(make_sequence('z_scene'))
    second = evaluate_corrected_crowd_sequence(make_sequence('a_scene'))
    result = reduce_corrected_crowd_summaries([first, second])
    assert result == reduce_corrected_crowd_summaries([second, first])
    assert (result.tp, result.fn, result.fp) == (14, 0, 2)


def smoke_test_corrected_crowd() -> None:
    '''Run the full portable corrected-crowd smoke surface.'''
    test_full_protocol_partition_views_and_round_trip()
    test_visibility_exclusion_and_invalid_projection_contract()
    test_pair_and_pcod_boundaries()
    test_acceleration_uses_vector_residual()
    test_cross_scene_reduction_is_micro_and_order_independent()


if __name__ == '__main__':
    smoke_test_corrected_crowd()
