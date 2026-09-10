'''Track partition parity, support boundaries and identity failure tests.'''
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from hjlib_evaluation import (
    Naive_Track_Statistics,
    evaluate_naive_tracks,
    finalize_naive_statistics,
    merge_naive_statistics,
)
from hjlib_evaluation.corrected_crowd_data import Corrected_Crowd_Sequence
from hjlib_evaluation.keypoint_oks import compute_paired_keypoint_oks
from hjlib_evaluation.virtualcrowd_naive_comparison import (
    direct_target_join,
    evaluate_virtualcrowd_naive_comparison,
)
from test_virtualcrowd_naive_comparison import FILTERING_ID, SPLIT_ID, make_sequence


def make_two_tracks() -> Corrected_Crowd_Sequence:
    '''Build two GT tracks; prediction chunk IDs split each halfway through.'''
    base = make_sequence(prediction_motion_scale=2.0)
    return replace(
        base,
        gt_frame_ids=np.tile(base.gt_frame_ids, 2),
        gt_track_ids=np.repeat(np.array([1, 2], dtype=np.int64), 8),
        gt_joints_world_m=np.concatenate((base.gt_joints_world_m, 3 * base.gt_joints_world_m)),
        gt_coco17_xy_px=np.tile(base.gt_coco17_xy_px, (2, 1, 1)),
        gt_visibility_native=np.tile(base.gt_visibility_native, (2, 1)),
        gt_bbox_xyxy_px=np.tile(base.gt_bbox_xyxy_px, (2, 1)),
        gt_pelvis_camera_depth_m=np.tile(base.gt_pelvis_camera_depth_m, 2),
        prediction_frame_ids=np.tile(base.prediction_frame_ids, 2),
        prediction_local_track_ids=np.repeat(np.array([10, 11, 20, 21], dtype=np.int64), 4),
        prediction_joints_world_m=np.concatenate((
            base.prediction_joints_world_m, base.gt_joints_world_m,
        )),
        prediction_coco17_xy_px=np.tile(base.prediction_coco17_xy_px, (2, 1, 1)) + 1.5,
        prediction_coco17_camera_depth_m=np.tile(base.prediction_coco17_camera_depth_m, (2, 1)),
        prediction_pelvis_camera_depth_m=np.tile(base.prediction_pelvis_camera_depth_m, 2),
        prediction_identity_target_gt_rows=np.arange(16, dtype=np.int64),
        matched_gt_rows=np.concatenate((base.matched_gt_rows, base.matched_gt_rows + 8)),
        matched_prediction_rows=np.concatenate((base.matched_prediction_rows, base.matched_prediction_rows + 8)),
        common_gt_mask=np.tile(base.common_gt_mask, 2),
    )


def test_track_partition_parity_and_chunk_independence() -> None:
    sequence = make_two_tracks()
    selected: NDArray[np.bool_] = np.ones(16, dtype=np.bool_)
    selected[-1] = False
    with patch('hjlib_evaluation.naive_track_statistics.direct_target_join', wraps=direct_target_join) as join:
        tracks = evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, selected)
        assert join.call_count == 1
    assert tuple(item.gt_track_id for item in tracks) == (1, 2)
    assert tracks[0].frame_ranges == ((0, 8),)
    assert tracks[1].frame_ranges == ((0, 7),)
    assert tracks[0].statistics.acc_root_sample_count == 2
    assert tracks[1].statistics.acc_root_sample_count == 1
    merged = merge_naive_statistics([item.statistics for item in tracks], sequence.scene_id)
    baseline = evaluate_virtualcrowd_naive_comparison(sequence, FILTERING_ID, SPLIT_ID, selected)
    for name, expected in asdict(baseline).items():
        actual = getattr(merged, name)
        assert actual == pytest.approx(expected) if isinstance(expected, float) else actual == expected
    root_ratio = finalize_naive_statistics(merged)['acc_root_ratio']
    assert root_ratio == pytest.approx(
        merged.acc_root_predicted_sum_m_per_frame2 / merged.acc_root_reference_sum_m_per_frame2,
    )
    assert root_ratio != pytest.approx((2.0 + 1.0 / 3.0) / 2.0)


def test_gaps_row_zero_and_empty_selection() -> None:
    sequence = make_sequence()
    selected = np.array([True, True, True, False, False, True, True, True])
    tracks = evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, selected)
    assert len(tracks) == 1
    assert tracks[0].frame_ranges == ((0, 3), (5, 8))
    assert tracks[0].statistics.selected_gt_count == 6
    assert tracks[0].statistics.acc_root_sample_count == 0
    assert finalize_naive_statistics(tracks[0].statistics)['acc_root_ratio'] is None
    assert evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, np.zeros(8, dtype=np.bool_)) == ()
    only_first = np.array([True, False, False, False, False, False, False, False])
    assert evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, only_first)[0].frame_ranges == ((0, 1),)


def test_unsupported_metrics_and_stationary_reference() -> None:
    sequence = make_sequence(reference_motion_scale=0.0)
    sequence = replace(
        sequence, gt_visibility_native=np.zeros((8, 17)),
        matched_gt_rows=np.empty(0, dtype=np.int64),
        matched_prediction_rows=np.empty(0, dtype=np.int64),
        common_gt_mask=np.zeros(8, dtype=np.bool_),
    )
    track = evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, np.ones(8, dtype=np.bool_))[0]
    values = finalize_naive_statistics(track.statistics)
    assert values == {'mpjpe_world_mm': 0.0, 't_mpjpe_mm': 0.0, 'oks_vis': None, 'acc_root_ratio': None}
    assert track.statistics.acc_root_sample_count == 2
    empty = evaluate_virtualcrowd_naive_comparison(sequence, FILTERING_ID, SPLIT_ID, np.zeros(8, dtype=np.bool_))
    assert all(value is None for value in finalize_naive_statistics(empty).values())


def test_duplicate_direct_target_and_invalid_ranges() -> None:
    base = make_sequence()
    sequence = replace(
        base,
        prediction_frame_ids=np.append(base.prediction_frame_ids, 0),
        prediction_local_track_ids=np.append(base.prediction_local_track_ids, 99),
        prediction_joints_world_m=np.concatenate((base.prediction_joints_world_m, base.prediction_joints_world_m[:1])),
        prediction_coco17_xy_px=np.concatenate((base.prediction_coco17_xy_px, base.prediction_coco17_xy_px[:1])),
        prediction_coco17_camera_depth_m=np.concatenate((base.prediction_coco17_camera_depth_m, base.prediction_coco17_camera_depth_m[:1])),
        prediction_pelvis_camera_depth_m=np.append(base.prediction_pelvis_camera_depth_m, 5.0),
        prediction_identity_target_gt_rows=np.append(base.prediction_identity_target_gt_rows, 0),
    )
    with pytest.raises(ValueError, match='exactly one direct-target'):
        evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, np.ones(8, dtype=np.bool_))
    track = evaluate_naive_tracks(base, FILTERING_ID, SPLIT_ID, np.ones(8, dtype=np.bool_))[0]
    with pytest.raises(ValueError, match='disjoint and maximal'):
        Naive_Track_Statistics(1, ((0, 4), (4, 8)), track.statistics)
    with pytest.raises(ValueError, match='selected_gt_count'):
        Naive_Track_Statistics(1, ((0, 7),), track.statistics)
    with pytest.raises(ValueError, match='same profile, filtering and split'):
        merge_naive_statistics(
            [track.statistics, replace(track.statistics, split_id='other')],
            track.statistics.scene_id,
        )
    with pytest.raises(ValueError, match='requested scene ID'):
        merge_naive_statistics(
            [track.statistics, replace(track.statistics, scene_id='other')],
            track.statistics.scene_id,
        )
    with pytest.raises(ValueError, match='empty'):
        merge_naive_statistics([], 'scene')


def test_paired_visibility_parity_and_validation() -> None:
    base = make_sequence()
    visibility = np.array(base.gt_visibility_native, copy=True)
    visibility[:, 0] = 0.0
    target = np.array(base.prediction_coco17_xy_px, copy=True)
    target[:, 0] += 5000.0
    target[:, 1] += 10.0
    sequence = replace(base, gt_visibility_native=visibility, prediction_coco17_xy_px=target)
    selected = np.ones(8, dtype=np.bool_)
    track = evaluate_naive_tracks(sequence, FILTERING_ID, SPLIT_ID, selected)[0]
    baseline = evaluate_virtualcrowd_naive_comparison(sequence, FILTERING_ID, SPLIT_ID, selected)
    assert track.statistics.oks_vis_sum == pytest.approx(baseline.oks_vis_sum)
    points = np.zeros((1, 17, 2), dtype=np.float64)
    with pytest.raises(ValueError, match='at least one valid'):
        compute_paired_keypoint_oks(points, points, np.ones(1), np.ones(17), np.zeros((1, 17), dtype=np.bool_))
    bad_depth = np.array(sequence.prediction_coco17_camera_depth_m, copy=True)
    bad_depth[2, 1] = 0.0
    with pytest.raises(ValueError, match='positive prediction camera depth'):
        evaluate_naive_tracks(replace(sequence, prediction_coco17_camera_depth_m=bad_depth), FILTERING_ID, SPLIT_ID, selected)


def smoke_test_naive_track_statistics() -> None:
    '''Run the GT-MOT sufficient-statistics smoke entry point.'''
    test_track_partition_parity_and_chunk_independence()
    test_gaps_row_zero_and_empty_selection()
    test_unsupported_metrics_and_stationary_reference()
    test_duplicate_direct_target_and_invalid_ranges()
    test_paired_visibility_parity_and_validation()


if __name__ == '__main__':
    smoke_test_naive_track_statistics()
    print('test_naive_track_statistics: smoke tests passed')
