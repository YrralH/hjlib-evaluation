'''Smoke tests for the three-scope standard evaluation profile.'''
from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from hjlib_evaluation.corrected_crowd_data import (
    CORRECTED_CROWD_SCHEMA_VERSION,
    Corrected_Crowd_Sequence,
)
from hjlib_evaluation.standard_evaluation import (
    STANDARD_EVALUATION_METRICS,
    STANDARD_EVALUATION_PROFILE_ID,
    Standard_Evaluation_Scene_Summary,
    evaluate_standard_evaluation_scene_indexed,
    reduce_standard_evaluation_summaries,
    standard_evaluation_result_from_json,
    standard_evaluation_result_to_json,
    standard_evaluation_scene_metrics_to_json,
    standard_evaluation_summary_from_json,
    standard_evaluation_summary_to_json,
)
from hjlib_evaluation.standard_evaluation_scope import (
    Standard_Evaluation_Direct_Target_Join,
    Standard_Evaluation_Scope_Index,
    build_standard_evaluation_scope_index,
    standard_evaluation_scope_rows,
)
import hjlib_evaluation.standard_evaluation_metrics as metric_leaves


def make_standard_sequence() -> tuple[
        Corrected_Crowd_Sequence,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
    '''Build two ten-frame tracks with unequal per-joint acceleration ratios.'''
    frame_domain = np.arange(10, dtype=np.int64)
    gt_frames = np.repeat(frame_domain, 2)
    gt_tracks = np.tile(np.array([1, 2], dtype=np.int64), 10)
    count = len(gt_frames)
    joint = np.arange(24, dtype=np.float64)
    joint_template = np.stack((
        0.01 * joint,
        0.005 * np.sin(joint),
        0.007 * np.cos(joint),
    ), axis=1)
    coefficients = 0.0002 * (joint + 1.0)
    ratios = 0.5 + joint / 23.0 * 1.5
    gt_joints = np.empty((count, 24, 3), dtype=np.float64)
    predicted_joints = np.empty_like(gt_joints)
    for row, (frame_id, track_id) in enumerate(
            zip(gt_frames, gt_tracks, strict=True),
        ):
        anchor = np.array([2.0 * (track_id - 1), 0.0, 5.0])
        motion = coefficients * float(frame_id ** 2)
        gt_joints[row] = anchor + joint_template
        gt_joints[row, :, 1] += motion
        predicted_joints[row] = anchor + joint_template
        predicted_joints[row, :, 1] += ratios * motion
    gt_xy = np.empty((count, 17, 2), dtype=np.float64)
    gt_xy[:, :, 0] = 20.0 + np.arange(17, dtype=np.float64)
    gt_xy[:, :, 1] = 40.0 + 2.0 * np.arange(17, dtype=np.float64)
    prediction_xy = np.array(gt_xy, copy=True)
    prediction_xy[:, :, 0] += 1.0
    selected = np.ones(count, dtype=np.bool_)
    selected[(gt_frames == 4) & (gt_tracks == 1)] = False
    labels = np.where(gt_tracks == 1, 0, 1).astype(np.int64)
    rows = np.arange(count, dtype=np.int64)
    sequence = Corrected_Crowd_Sequence(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id='synthetic_scene',
        frame_domain=frame_domain,
        gt_frame_ids=gt_frames,
        gt_track_ids=gt_tracks,
        gt_joints_world_m=gt_joints,
        gt_coco17_xy_px=gt_xy,
        gt_visibility_native=np.ones((count, 17), dtype=np.float64),
        gt_bbox_xyxy_px=np.tile(
            np.array([0.0, 0.0, 100.0, 200.0]),
            (count, 1),
        ),
        gt_pelvis_camera_depth_m=gt_joints[:, 0, 2],
        prediction_frame_ids=gt_frames,
        prediction_local_track_ids=gt_tracks,
        prediction_joints_world_m=predicted_joints,
        prediction_coco17_xy_px=prediction_xy,
        prediction_coco17_camera_depth_m=np.full((count, 17), 5.0),
        prediction_pelvis_camera_depth_m=predicted_joints[:, 0, 2],
        prediction_identity_target_gt_rows=rows,
        matched_gt_rows=rows,
        matched_prediction_rows=rows,
        common_gt_mask=np.ones(count, dtype=np.bool_),
    )
    return sequence, selected, labels, ratios


def test_scope_index_preserves_alignment_run_and_splits_acceleration() -> None:
    sequence, selected, labels, _ratios = make_standard_sequence()
    with pytest.raises(ValueError, match='one-to-one'):
        Standard_Evaluation_Direct_Target_Join(
            sequence,
            np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64),
        )
    index = build_standard_evaluation_scope_index(sequence, selected, labels)

    assert index.tracks.count_scope == 2
    assert index.visruns.count_scope == 2
    assert index.acceleration_runs.count_scope == 3
    assert index.frames.count_scope == 10
    assert index.visrun_acceleration_offsets.tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match='tracks does not match'):
        Standard_Evaluation_Scope_Index(
            index.join,
            index.joined_visrun_labels,
            index.frames,
            index.visruns,
            index.acceleration_runs,
            index.visrun_acceleration_offsets,
            index.frames,
        )
    first_visrun = standard_evaluation_scope_rows(index.visruns, 0)
    first_frames = sequence.gt_frame_ids[index.join.gt_rows[first_visrun]]
    assert first_frames.tolist() == [0, 1, 2, 3, 5, 6, 7, 8, 9]


def test_all_metrics_reduce_and_joint_ratio_divides_before_mean() -> None:
    sequence, selected, labels, ratios = make_standard_sequence()
    index = build_standard_evaluation_scope_index(sequence, selected, labels)
    summary = evaluate_standard_evaluation_scene_indexed(
        index,
        'vc.visible_common',
        'vc.test6',
    )
    result = reduce_standard_evaluation_summaries([summary])

    assert STANDARD_EVALUATION_METRICS == (
        'MPJPE-WORLD', 'T-MPJPE', 'RT-MPJPE',
        'SEQ-T-MPJPE-VISRUN', 'SEQ-RT-MPJPE-VISRUN',
        'SEQ-T-MPJPE-TRACK', 'SEQ-RT-MPJPE-TRACK', 'RTE-WORLD',
        'ACC-ROOT', 'ACC-ROOT-RATIO', 'ACC-JOINT-RATIO',
        'OKS-VIS', 'PPDS', 'PA-PPDS',
    )
    assert np.all(result.metric_sample_counts > 0)
    assert result.metric('ACC-ROOT-RATIO') == pytest.approx(ratios[0])
    assert result.metric('ACC-JOINT-RATIO') == pytest.approx(float(np.mean(ratios)))
    weighted_ratio = float(
        np.sum(summary.acc_joint_ratio_predicted_sums)
        / np.sum(summary.acc_joint_ratio_reference_sums)
    )
    assert result.metric('ACC-JOINT-RATIO') != pytest.approx(weighted_ratio)
    assert summary.acc_root_ratio_predicted_sum \
        == summary.acc_joint_ratio_predicted_sums[0]
    assert summary.acc_root_ratio_reference_sum \
        == summary.acc_joint_ratio_reference_sums[0]
    invalid_sums = np.array(summary.metric_sample_sums, copy=True)
    invalid_sums[9] = float(summary.metric_sample_counts[9]) + 1.0
    with pytest.raises(ValueError, match='OKS-VIS sum'):
        replace(summary, metric_sample_sums=invalid_sums)


def test_standard_json_roundtrips_raw_statistics() -> None:
    sequence, selected, labels, _ratios = make_standard_sequence()
    index = build_standard_evaluation_scope_index(sequence, selected, labels)
    summary = evaluate_standard_evaluation_scene_indexed(
        index,
        'vc.visible_common',
        'vc.test6',
    )
    result = reduce_standard_evaluation_summaries([summary])
    parsed_summary = standard_evaluation_summary_from_json(
        standard_evaluation_summary_to_json(summary),
    )
    parsed_result = standard_evaluation_result_from_json(
        standard_evaluation_result_to_json(result),
    )
    assert np.array_equal(parsed_summary.metric_sample_sums, summary.metric_sample_sums)
    assert np.array_equal(parsed_result.metric_values, result.metric_values)


def test_scene_projection_retains_local_zero_support_as_null() -> None:
    counts = np.array([
        24, 24, 24, 24, 24, 24, 24, 1, 0, 1, 0, 0,
    ], dtype=np.int64)
    summary = Standard_Evaluation_Scene_Summary(
        profile_id=STANDARD_EVALUATION_PROFILE_ID,
        filtering_id='vc.visible_common',
        split_id='vc.test6',
        scene_id='short_single_person_scene',
        selected_gt_count=1,
        metric_sample_sums=np.where(counts > 0, 1.0, 0.0),
        metric_sample_counts=counts,
        acc_root_ratio_predicted_sum=0.0,
        acc_root_ratio_reference_sum=0.0,
        acc_root_ratio_sample_count=0,
        acc_joint_ratio_predicted_sums=np.zeros(24),
        acc_joint_ratio_reference_sums=np.zeros(24),
        acc_joint_ratio_sample_count=0,
    )
    projection = standard_evaluation_scene_metrics_to_json(summary)
    metrics = projection['metrics']
    assert isinstance(metrics, dict)
    assert metrics['ACC-ROOT'] is None
    assert metrics['ACC-ROOT-RATIO'] is None
    assert metrics['ACC-JOINT-RATIO'] is None
    assert metrics['PPDS'] is None
    assert metrics['PA-PPDS'] is None
    assert metrics['MPJPE-WORLD'] is not None
    sequence, selected, labels, _ratios = make_standard_sequence()
    supported = evaluate_standard_evaluation_scene_indexed(
        build_standard_evaluation_scope_index(sequence, selected, labels),
        'vc.visible_common',
        'vc.test6',
    )
    combined = reduce_standard_evaluation_summaries([summary, supported])
    assert combined.scene_count == 2
    assert np.all(combined.metric_sample_counts > 0)


def test_metric_leaf_signatures_cannot_observe_traversal_identity() -> None:
    forbidden = {'sequence', 'index', 'frame_id', 'track_id', 'scope'}
    leaf_names = [
        name for name in metric_leaves.__all__ if name.startswith('compute_standard_')
    ]
    assert len(leaf_names) == 12
    for name in leaf_names:
        parameters = set(inspect.signature(getattr(metric_leaves, name)).parameters)
        assert not parameters & forbidden


def test_position_alignment_leaves_have_independent_expected_invariances() -> None:
    joint = np.arange(24, dtype=np.float64)
    skeleton = np.stack((
        0.02 * joint,
        np.sin(joint),
        np.cos(0.7 * joint),
    ), axis=1)
    reference = np.stack((skeleton, skeleton + np.array([0.0, 2.0, 0.0])))
    translated = reference + np.array([1.0, -2.0, 0.5])
    world = metric_leaves.compute_standard_mpjpe_world_values(
        translated,
        reference,
    )
    root = metric_leaves.compute_standard_rte_world_values(translated, reference)
    assert np.all(world > 0.0) and np.all(root > 0.0)
    assert np.max(metric_leaves.compute_standard_t_mpjpe_values(
        translated,
        reference,
    )) < 1.0e-12
    assert np.max(metric_leaves.compute_standard_rt_mpjpe_values(
        translated,
        reference,
    )) < 1.0e-12
    assert np.max(metric_leaves.compute_standard_sequence_t_mpjpe_values(
        translated,
        reference,
    )) < 1.0e-12
    assert np.max(metric_leaves.compute_standard_sequence_rt_mpjpe_values(
        translated,
        reference,
    )) < 1.0e-12

    occurrence_translated = reference + np.array([
        [[1.0, 0.0, 0.0]],
        [[-1.0, 0.0, 0.0]],
    ])
    assert np.max(metric_leaves.compute_standard_t_mpjpe_values(
        occurrence_translated,
        reference,
    )) < 1.0e-12
    assert np.mean(metric_leaves.compute_standard_sequence_t_mpjpe_values(
        occurrence_translated,
        reference,
    )) > 0.5

    angle = np.pi / 3.0
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = reference @ rotation.T
    assert np.mean(metric_leaves.compute_standard_t_mpjpe_values(
        rotated,
        reference,
    )) > 0.1
    assert np.max(metric_leaves.compute_standard_rt_mpjpe_values(
        rotated,
        reference,
    )) < 1.0e-12
    assert np.max(metric_leaves.compute_standard_sequence_rt_mpjpe_values(
        rotated,
        reference,
    )) < 1.0e-12
    assert not world.flags.writeable


def test_acceleration_ratio_leaves_are_independent_per_joint() -> None:
    frame = np.arange(9, dtype=np.float64)
    joint = np.arange(24, dtype=np.float64)
    reference = np.zeros((9, 24, 3), dtype=np.float64)
    reference[:, :, 1] = frame[:, None] ** 2 * (joint[None, :] + 1.0)
    ratios = 0.5 + joint / 23.0
    predicted = np.zeros_like(reference)
    predicted[:, :, 1] = reference[:, :, 1] * ratios[None, :]
    root_predicted, root_reference = (
        metric_leaves.compute_standard_acc_root_ratio_magnitudes(
            predicted,
            reference,
        )
    )
    joint_predicted, joint_reference = (
        metric_leaves.compute_standard_acc_joint_ratio_magnitudes(
            predicted,
            reference,
        )
    )
    assert np.sum(root_predicted) / np.sum(root_reference) \
        == pytest.approx(ratios[0])
    assert np.sum(joint_predicted, axis=0) / np.sum(joint_reference, axis=0) \
        == pytest.approx(ratios)
    assert np.all(metric_leaves.compute_standard_acc_root_values(
        predicted,
        reference,
    ) > 0.0)
    assert not joint_predicted.flags.writeable


def test_oks_and_pair_distance_leaves_apply_visibility_depth_and_pa_scale() -> None:
    reference_xy = np.zeros((2, 17, 2), dtype=np.float64)
    predicted_xy = np.array(reference_xy, copy=True)
    visibility = np.ones((2, 17), dtype=np.float64)
    depth = np.ones((2, 17), dtype=np.float64)
    bbox = np.tile(np.array([0.0, 0.0, 100.0, 100.0]), (2, 1))
    perfect = metric_leaves.compute_standard_oks_vis_values(
        predicted_xy,
        reference_xy,
        visibility,
        depth,
        bbox,
    )
    assert perfect == pytest.approx(np.ones(2))
    predicted_xy[:, 0, 0] = 100.0
    visibility[:, 1:] = 0.0
    perturbed = metric_leaves.compute_standard_oks_vis_values(
        predicted_xy,
        reference_xy,
        visibility,
        depth,
        bbox,
    )
    assert np.all(perturbed < perfect)
    depth[:, 0] = -1.0
    assert metric_leaves.compute_standard_oks_vis_values(
        predicted_xy,
        reference_xy,
        visibility,
        depth,
        bbox,
    ).size == 0

    reference_pelvis = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    predicted_pelvis = 2.0 * reference_pelvis
    assert metric_leaves.compute_standard_ppds_values(
        predicted_pelvis,
        reference_pelvis,
    ) == pytest.approx(np.array([0.0]))
    assert metric_leaves.compute_standard_pa_ppds_values(
        predicted_pelvis,
        reference_pelvis,
    ) == pytest.approx(np.array([1.0]))


def smoke_test_standard_evaluation() -> None:
    '''Run the complete standard-evaluation smoke without pytest recursion.'''
    test_scope_index_preserves_alignment_run_and_splits_acceleration()
    test_all_metrics_reduce_and_joint_ratio_divides_before_mean()
    test_standard_json_roundtrips_raw_statistics()
    test_scene_projection_retains_local_zero_support_as_null()
    test_metric_leaf_signatures_cannot_observe_traversal_identity()
    test_position_alignment_leaves_have_independent_expected_invariances()
    test_acceleration_ratio_leaves_are_independent_per_joint()
    test_oks_and_pair_distance_leaves_apply_visibility_depth_and_pa_scale()


if __name__ == '__main__':
    smoke_test_standard_evaluation()
    print('test_standard_evaluation: smoke tests passed')
