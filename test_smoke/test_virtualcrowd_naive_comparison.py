'''Smoke tests for the provisional VirtualCrowd four-metric evaluator.'''
from dataclasses import replace

import numpy as np
import pytest

from hjlib_evaluation import (
    CORRECTED_CROWD_SCHEMA_VERSION,
    VC_NAIVE_COMPARISON_METRICS,
    VC_NAIVE_COMPARISON_PROFILE_ID,
    Corrected_Crowd_Sequence,
    VirtualCrowd_Direct_Target_Join,
    VirtualCrowd_Naive_Comparison_Result,
    VirtualCrowd_Naive_Comparison_Sequence_Summary,
    compute_virtualcrowd_acc_root_ratio_statistics,
    compute_virtualcrowd_mpjpe_world_statistics,
    compute_virtualcrowd_oks_vis_statistics,
    compute_virtualcrowd_t_mpjpe_statistics,
    direct_target_join,
    evaluate_virtualcrowd_naive_comparison,
    reduce_virtualcrowd_naive_comparison_summaries,
)


FILTERING_ID = 'vc.visible_common'
SPLIT_ID = 'vc.test6'


def make_sequence(
    scene_id: str = 'scene_a',
    *,
    reference_motion_scale: float = 1.0,
    prediction_motion_scale: float = 1.0,
    motion_power: int = 2,
) -> Corrected_Crowd_Sequence:
    '''Build one eight-frame track with an internal zero-visible occurrence.'''
    frames = np.arange(8, dtype=np.int64)
    phase = frames.astype(np.float64) ** motion_power
    reference_root = np.stack((
        np.zeros(8, dtype=np.float64),
        0.01 * reference_motion_scale * phase,
        np.full(8, 5.0, dtype=np.float64),
    ), axis=1)
    predicted_root = np.stack((
        np.zeros(8, dtype=np.float64),
        0.01 * reference_motion_scale * prediction_motion_scale * phase,
        np.full(8, 5.0, dtype=np.float64),
    ), axis=1)
    joint_template = np.stack((
        np.arange(24, dtype=np.float64) * 0.001,
        np.arange(24, dtype=np.float64) * 0.002,
        np.arange(24, dtype=np.float64) * 0.003,
    ), axis=1)
    gt_joints = reference_root[:, None, :] + joint_template[None, :, :]
    prediction_joints = predicted_root[:, None, :] + joint_template[None, :, :]
    gt_xy = np.zeros((8, 17, 2), dtype=np.float64)
    gt_xy[..., 0] = np.arange(17, dtype=np.float64)[None, :]
    gt_xy[..., 1] = 2.0 * np.arange(17, dtype=np.float64)[None, :]
    visibility = np.ones((8, 17), dtype=np.float64)
    visibility[3] = 0.0
    matched_rows = np.array([0, 1, 4, 5, 6, 7], dtype=np.int64)
    common_mask = np.any(visibility > 0.0, axis=1)
    return Corrected_Crowd_Sequence(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id=scene_id,
        frame_domain=frames,
        gt_frame_ids=frames,
        gt_track_ids=np.ones(8, dtype=np.int64),
        gt_joints_world_m=gt_joints,
        gt_coco17_xy_px=gt_xy,
        gt_visibility_native=visibility,
        gt_bbox_xyxy_px=np.tile(
            np.array([0.0, 0.0, 100.0, 200.0], dtype=np.float64),
            (8, 1),
        ),
        gt_pelvis_camera_depth_m=gt_joints[:, 0, 2],
        prediction_frame_ids=frames,
        prediction_local_track_ids=np.ones(8, dtype=np.int64),
        prediction_joints_world_m=prediction_joints,
        prediction_coco17_xy_px=np.array(gt_xy, copy=True),
        prediction_coco17_camera_depth_m=np.full((8, 17), 5.0),
        prediction_pelvis_camera_depth_m=prediction_joints[:, 0, 2],
        prediction_identity_target_gt_rows=frames,
        matched_gt_rows=matched_rows,
        matched_prediction_rows=matched_rows,
        common_gt_mask=common_mask,
    )


def selected_all(sequence: Corrected_Crowd_Sequence) -> np.ndarray:
    '''Return the full GT occurrence mask.'''
    return np.ones(len(sequence.gt_frame_ids), dtype=np.bool_)


def evaluate_result(sequence: Corrected_Crowd_Sequence):
    '''Evaluate and reduce one full synthetic scene.'''
    summary = evaluate_virtualcrowd_naive_comparison(
        sequence,
        FILTERING_ID,
        SPLIT_ID,
        selected_all(sequence),
    )
    return summary, reduce_virtualcrowd_naive_comparison_summaries([summary])


def test_exact_geometry_profile_and_zero_visible_support() -> None:
    sequence = make_sequence()
    summary, result = evaluate_result(sequence)

    assert VC_NAIVE_COMPARISON_PROFILE_ID == 'VC_NAIVE_COMPARISON_METRICS_V1'
    assert VC_NAIVE_COMPARISON_METRICS == (
        'MPJPE-WORLD', 'T-MPJPE', 'OKS-VIS', 'ACC-ROOT-RATIO',
    )
    assert isinstance(summary, VirtualCrowd_Naive_Comparison_Sequence_Summary)
    assert isinstance(result, VirtualCrowd_Naive_Comparison_Result)
    assert summary.selected_gt_count == 8
    assert summary.matched_selected_count == 8
    assert summary.mpjpe_world_count == 8 * 24
    assert summary.oks_vis_count == 7
    assert summary.acc_root_sample_count == 2
    assert result.mpjpe_world_mm == pytest.approx(0.0)
    assert result.t_mpjpe_mm == pytest.approx(0.0)
    assert result.oks_vis == pytest.approx(1.0)
    assert result.acc_root_ratio == pytest.approx(1.0)


def test_validated_join_and_independent_metric_leaves() -> None:
    sequence = make_sequence()
    join = direct_target_join(sequence, selected_all(sequence))
    assert isinstance(join, VirtualCrowd_Direct_Target_Join)
    assert join.gt_rows.tolist() == list(range(8))
    assert join.prediction_rows.tolist() == list(range(8))
    assert not join.gt_rows.flags.writeable
    assert not join.prediction_rows.flags.writeable

    assert compute_virtualcrowd_mpjpe_world_statistics(join) == (
        pytest.approx(0.0),
        8 * 24,
    )
    assert compute_virtualcrowd_t_mpjpe_statistics(join) == (
        pytest.approx(0.0),
        8 * 24,
    )
    assert compute_virtualcrowd_oks_vis_statistics(join) == (
        pytest.approx(7.0),
        7,
    )
    predicted_acc, reference_acc, acc_count = (
        compute_virtualcrowd_acc_root_ratio_statistics(join)
    )
    assert predicted_acc == pytest.approx(reference_acc)
    assert acc_count == 2

    with pytest.raises(ValueError, match='row counts must match'):
        VirtualCrowd_Direct_Target_Join(
            sequence,
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError, match='do not satisfy identity targets'):
        VirtualCrowd_Direct_Target_Join(
            sequence,
            np.array([0, 1], dtype=np.int64),
            np.array([1, 0], dtype=np.int64),
        )


def test_naive_sequence_construction_skips_layout_preflight() -> None:
    base = make_sequence()
    coincident = np.repeat(base.gt_joints_world_m[:1], 2, axis=0)
    sequence = Corrected_Crowd_Sequence(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id='degenerate_layout',
        frame_domain=np.array([0], dtype=np.int64),
        gt_frame_ids=np.array([0, 0], dtype=np.int64),
        gt_track_ids=np.array([1, 2], dtype=np.int64),
        gt_joints_world_m=coincident,
        gt_coco17_xy_px=np.repeat(base.gt_coco17_xy_px[:1], 2, axis=0),
        gt_visibility_native=np.ones((2, 17), dtype=np.float64),
        gt_bbox_xyxy_px=np.repeat(base.gt_bbox_xyxy_px[:1], 2, axis=0),
        gt_pelvis_camera_depth_m=np.full(2, 5.0),
        prediction_frame_ids=np.array([0, 0], dtype=np.int64),
        prediction_local_track_ids=np.array([1, 2], dtype=np.int64),
        prediction_joints_world_m=coincident,
        prediction_coco17_xy_px=np.repeat(
            base.prediction_coco17_xy_px[:1],
            2,
            axis=0,
        ),
        prediction_coco17_camera_depth_m=np.full((2, 17), 5.0),
        prediction_pelvis_camera_depth_m=np.full(2, 5.0),
        prediction_identity_target_gt_rows=np.array([0, 1], dtype=np.int64),
        matched_gt_rows=np.array([0, 1], dtype=np.int64),
        matched_prediction_rows=np.array([0, 1], dtype=np.int64),
        common_gt_mask=np.ones(2, dtype=np.bool_),
    )
    summary = evaluate_virtualcrowd_naive_comparison(
        sequence,
        FILTERING_ID,
        SPLIT_ID,
        np.ones(2, dtype=np.bool_),
    )
    assert summary.selected_gt_count == 2
    assert summary.mpjpe_world_sum_m == pytest.approx(0.0)


def test_translation_and_acceleration_scaling_semantics() -> None:
    sequence = make_sequence()
    translated_joints = np.array(sequence.prediction_joints_world_m, copy=True)
    translated_joints += np.array([1.0, 0.0, 0.0])
    translated = replace(sequence, prediction_joints_world_m=translated_joints)
    _, translated_result = evaluate_result(translated)
    assert translated_result.mpjpe_world_mm == pytest.approx(1000.0)
    assert translated_result.t_mpjpe_mm == pytest.approx(0.0)
    assert translated_result.oks_vis == pytest.approx(1.0)
    assert translated_result.acc_root_ratio == pytest.approx(1.0)

    _, scaled_result = evaluate_result(
        make_sequence(prediction_motion_scale=2.0),
    )
    assert scaled_result.t_mpjpe_mm == pytest.approx(0.0)
    assert scaled_result.acc_root_ratio == pytest.approx(2.0)


def test_oks_visibility_bbox_and_source_depth() -> None:
    sequence = make_sequence()
    visibility = np.array(sequence.gt_visibility_native, copy=True)
    visibility[0, 0] = 0.0
    hidden_error_xy = np.array(sequence.prediction_coco17_xy_px, copy=True)
    hidden_error_xy[0, 0] += 10000.0
    hidden_error = replace(
        sequence,
        gt_visibility_native=visibility,
        prediction_coco17_xy_px=hidden_error_xy,
    )
    assert evaluate_result(hidden_error)[1].oks_vis == pytest.approx(1.0)

    visible_error_xy = np.array(sequence.prediction_coco17_xy_px, copy=True)
    visible_error_xy[0, 1, 0] += 10.0
    visible_error = replace(sequence, prediction_coco17_xy_px=visible_error_xy)
    normal_oks = evaluate_result(visible_error)[1].oks_vis
    large_bbox = np.array(sequence.gt_bbox_xyxy_px, copy=True)
    large_bbox[:, 2:] *= 2.0
    large_bbox_oks = evaluate_result(
        replace(visible_error, gt_bbox_xyxy_px=large_bbox),
    )[1].oks_vis
    assert large_bbox_oks > normal_oks

    invalid_depth = np.array(sequence.prediction_coco17_camera_depth_m, copy=True)
    invalid_depth[2, 0] = 0.0
    unchecked_by_stable_match = replace(
        sequence,
        prediction_coco17_camera_depth_m=invalid_depth,
    )
    with pytest.raises(ValueError, match='positive prediction camera depth'):
        evaluate_virtualcrowd_naive_comparison(
            unchecked_by_stable_match,
            FILTERING_ID,
            SPLIT_ID,
            selected_all(sequence),
        )


def test_exact_direct_target_completeness() -> None:
    sequence = make_sequence()
    missing_targets = np.array(
        sequence.prediction_identity_target_gt_rows,
        copy=True,
    )
    missing_targets[3] = -1
    missing = replace(
        sequence,
        prediction_identity_target_gt_rows=missing_targets,
    )
    with pytest.raises(ValueError, match='exactly one direct-target prediction'):
        evaluate_virtualcrowd_naive_comparison(
            missing,
            FILTERING_ID,
            SPLIT_ID,
            selected_all(sequence),
        )

    duplicate = replace(
        sequence,
        prediction_frame_ids=np.append(sequence.prediction_frame_ids, 0),
        prediction_local_track_ids=np.append(
            sequence.prediction_local_track_ids,
            99,
        ),
        prediction_joints_world_m=np.concatenate((
            sequence.prediction_joints_world_m,
            sequence.prediction_joints_world_m[:1],
        )),
        prediction_coco17_xy_px=np.concatenate((
            sequence.prediction_coco17_xy_px,
            sequence.prediction_coco17_xy_px[:1],
        )),
        prediction_coco17_camera_depth_m=np.concatenate((
            sequence.prediction_coco17_camera_depth_m,
            sequence.prediction_coco17_camera_depth_m[:1],
        )),
        prediction_pelvis_camera_depth_m=np.append(
            sequence.prediction_pelvis_camera_depth_m,
            sequence.prediction_pelvis_camera_depth_m[0],
        ),
        prediction_identity_target_gt_rows=np.append(
            sequence.prediction_identity_target_gt_rows,
            0,
        ),
    )
    with pytest.raises(ValueError, match='exactly one direct-target prediction'):
        evaluate_virtualcrowd_naive_comparison(
            duplicate,
            FILTERING_ID,
            SPLIT_ID,
            selected_all(sequence),
        )


def test_global_ratio_reduction_and_order_invariance() -> None:
    first = evaluate_virtualcrowd_naive_comparison(
        make_sequence('scene_a', prediction_motion_scale=2.0),
        FILTERING_ID,
        SPLIT_ID,
        np.ones(8, dtype=np.bool_),
    )
    second = evaluate_virtualcrowd_naive_comparison(
        make_sequence(
            'scene_b',
            reference_motion_scale=4.0,
            prediction_motion_scale=0.5,
        ),
        FILTERING_ID,
        SPLIT_ID,
        np.ones(8, dtype=np.bool_),
    )
    result = reduce_virtualcrowd_naive_comparison_summaries([first, second])
    reverse = reduce_virtualcrowd_naive_comparison_summaries([second, first])
    expected = (
        first.acc_root_predicted_sum_m_per_frame2
        + second.acc_root_predicted_sum_m_per_frame2
    ) / (
        first.acc_root_reference_sum_m_per_frame2
        + second.acc_root_reference_sum_m_per_frame2
    )
    assert result == reverse
    assert result.acc_root_ratio == pytest.approx(expected)
    assert result.acc_root_ratio != pytest.approx((2.0 + 0.5) / 2.0)
    with pytest.raises(ValueError, match='scene_id values must be unique'):
        reduce_virtualcrowd_naive_comparison_summaries([first, first])


def test_temporal_support_and_failure_boundaries() -> None:
    sequence = make_sequence()
    split_mask = np.array(
        [True, True, True, False, False, True, True, True],
        dtype=np.bool_,
    )
    short = evaluate_virtualcrowd_naive_comparison(
        sequence,
        FILTERING_ID,
        SPLIT_ID,
        split_mask,
    )
    assert short.acc_root_sample_count == 0
    with pytest.raises(ValueError, match='ACC-ROOT-RATIO support is empty'):
        reduce_virtualcrowd_naive_comparison_summaries([short])

    static_root = evaluate_virtualcrowd_naive_comparison(
        make_sequence(motion_power=0),
        FILTERING_ID,
        SPLIT_ID,
        selected_all(sequence),
    )
    with pytest.raises(ValueError, match='reference denominator must be positive'):
        reduce_virtualcrowd_naive_comparison_summaries([static_root])
    with pytest.raises(ValueError, match='selected_gt_mask must have shape'):
        evaluate_virtualcrowd_naive_comparison(
            sequence,
            FILTERING_ID,
            SPLIT_ID,
            np.ones(7, dtype=np.bool_),
        )


def test_unequal_support_micro_reduction_and_partial_scene_support() -> None:
    base = make_sequence('scene_base')
    base_summary = evaluate_virtualcrowd_naive_comparison(
        base,
        FILTERING_ID,
        SPLIT_ID,
        selected_all(base),
    )

    unequal = make_sequence('scene_unequal')
    unequal_joints = np.array(unequal.prediction_joints_world_m, copy=True)
    unequal_joints[:, 1, 0] += 1.0
    unequal_xy = np.array(unequal.prediction_coco17_xy_px, copy=True)
    unequal_xy[:, 1, 0] += 10.0
    unequal = replace(
        unequal,
        prediction_joints_world_m=unequal_joints,
        prediction_coco17_xy_px=unequal_xy,
    )
    first_seven = np.array(
        [True, True, True, True, True, True, True, False],
        dtype=np.bool_,
    )
    unequal_summary = evaluate_virtualcrowd_naive_comparison(
        unequal,
        FILTERING_ID,
        SPLIT_ID,
        first_seven,
    )
    unequal_result = reduce_virtualcrowd_naive_comparison_summaries([
        unequal_summary,
        base_summary,
    ])
    assert unequal_result.mpjpe_world_mm == pytest.approx(
        1000.0
        * (
            base_summary.mpjpe_world_sum_m
            + unequal_summary.mpjpe_world_sum_m
        )
        / (
            base_summary.mpjpe_world_count
            + unequal_summary.mpjpe_world_count
        )
    )
    assert unequal_result.t_mpjpe_mm == pytest.approx(
        1000.0
        * (base_summary.t_mpjpe_sum_m + unequal_summary.t_mpjpe_sum_m)
        / (base_summary.t_mpjpe_count + unequal_summary.t_mpjpe_count)
    )
    assert unequal_result.oks_vis == pytest.approx(
        (base_summary.oks_vis_sum + unequal_summary.oks_vis_sum)
        / (base_summary.oks_vis_count + unequal_summary.oks_vis_count)
    )

    zero_visibility = make_sequence('scene_zero_oks')
    zero_oks_sequence = replace(
        zero_visibility,
        gt_visibility_native=np.zeros((8, 17), dtype=np.float64),
        matched_gt_rows=np.empty((0,), dtype=np.int64),
        matched_prediction_rows=np.empty((0,), dtype=np.int64),
        common_gt_mask=np.zeros(8, dtype=np.bool_),
    )
    zero_oks = evaluate_virtualcrowd_naive_comparison(
        zero_oks_sequence,
        FILTERING_ID,
        SPLIT_ID,
        selected_all(zero_oks_sequence),
    )
    assert zero_oks.oks_vis_count == 0
    combined_oks = reduce_virtualcrowd_naive_comparison_summaries([
        zero_oks,
        base_summary,
    ])
    assert combined_oks.oks_vis == pytest.approx(
        base_summary.oks_vis_sum / base_summary.oks_vis_count
    )

    zero_acc_sequence = make_sequence('scene_zero_acc')
    split_mask = np.array(
        [True, True, True, False, False, True, True, True],
        dtype=np.bool_,
    )
    zero_acc = evaluate_virtualcrowd_naive_comparison(
        zero_acc_sequence,
        FILTERING_ID,
        SPLIT_ID,
        split_mask,
    )
    assert zero_acc.acc_root_sample_count == 0
    combined_acc = reduce_virtualcrowd_naive_comparison_summaries([
        zero_acc,
        base_summary,
    ])
    assert combined_acc.acc_root_ratio == pytest.approx(
        base_summary.acc_root_predicted_sum_m_per_frame2
        / base_summary.acc_root_reference_sum_m_per_frame2
    )


def smoke_test_virtualcrowd_naive_comparison() -> None:
    '''Run the public provisional-evaluator smoke entry point.'''
    test_exact_geometry_profile_and_zero_visible_support()
    test_validated_join_and_independent_metric_leaves()
    test_naive_sequence_construction_skips_layout_preflight()
    test_translation_and_acceleration_scaling_semantics()
    test_oks_visibility_bbox_and_source_depth()
    test_exact_direct_target_completeness()
    test_global_ratio_reduction_and_order_invariance()
    test_temporal_support_and_failure_boundaries()
    test_unequal_support_micro_reduction_and_partial_scene_support()


if __name__ == '__main__':
    smoke_test_virtualcrowd_naive_comparison()
    print('test_virtualcrowd_naive_comparison: smoke tests passed')
