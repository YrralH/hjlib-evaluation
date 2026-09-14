'''GT-MOT sequence statistics, independent of prediction runner chunks.'''
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import Corrected_Crowd_Sequence
from hjlib_evaluation.corrected_crowd_protocol import COCO17_SIGMAS
from hjlib_evaluation.joint_error import (
    compute_joint_position_errors,
    compute_pa_joint_position_errors,
)
from hjlib_evaluation.keypoint_oks import (
    compute_paired_keypoint_oks,
    make_positive_depth_joint_mask,
)
from hjlib_evaluation.virtualcrowd_naive_comparison import (
    VC_NAIVE_COMPARISON_PROFILE_ID,
    VC_NAIVE_COMPARISON_PROFILE_V2_ID,
    VirtualCrowd_Naive_Comparison_Sequence_Summary,
    direct_target_join,
    require_exact_nonnegative_int,
    require_identity,
    root_acceleration_magnitudes,
)


@dataclass(frozen=True, slots=True)
class Naive_Track_Statistics:
    '''One scene/native GT track with maximal selected half-open frame ranges.'''

    gt_track_id: int
    frame_ranges: tuple[tuple[int, int], ...]
    statistics: VirtualCrowd_Naive_Comparison_Sequence_Summary

    def __post_init__(self) -> None:
        require_exact_nonnegative_int(self.gt_track_id, 'gt_track_id')
        if type(self.statistics) is not VirtualCrowd_Naive_Comparison_Sequence_Summary:
            raise TypeError('statistics must be a naive comparison summary')
        if type(self.frame_ranges) is not tuple or not self.frame_ranges:
            raise ValueError('frame_ranges must be a nonempty tuple')
        previous_end = -1
        count = 0
        for pair in self.frame_ranges:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('frame ranges must be pairs')
            start, end = pair
            require_exact_nonnegative_int(start, 'range start')
            require_exact_nonnegative_int(end, 'range end')
            if start >= end or start <= previous_end:
                raise ValueError('frame ranges must be positive, disjoint and maximal')
            count += end - start
            previous_end = end
        if count != self.statistics.selected_gt_count:
            raise ValueError('frame ranges must match selected_gt_count')


def track_oks_statistics(
    sequence: Corrected_Crowd_Sequence,
    gt_rows: NDArray[np.int64],
    prediction_rows: NDArray[np.int64],
) -> tuple[float, int]:
    '''Compute selected identity-paired visible OKS without a pairwise matrix.'''
    valid = make_positive_depth_joint_mask(
        sequence.gt_visibility_native[gt_rows] > 0.0,
        sequence.prediction_coco17_camera_depth_m[prediction_rows],
    )
    supported = np.any(valid, axis=1)
    if not np.any(supported):
        return 0.0, 0
    gt_rows = gt_rows[supported]
    prediction_rows = prediction_rows[supported]
    valid = valid[supported]
    bbox = sequence.gt_bbox_xyxy_px[gt_rows]
    values = compute_paired_keypoint_oks(
        sequence.gt_coco17_xy_px[gt_rows],
        sequence.prediction_coco17_xy_px[prediction_rows],
        (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]),
        COCO17_SIGMAS,
        valid,
    )
    return math.fsum(float(value) for value in values), len(values)


def summarize_naive_track_rows(
    sequence: Corrected_Crowd_Sequence,
    gt_rows: NDArray[np.int64],
    prediction_rows: NDArray[np.int64],
    filtering_id: str,
    split_id: str,
) -> Naive_Track_Statistics:
    '''Summarize one already validated track in native frame order.'''
    frames = sequence.gt_frame_ids[gt_rows]
    starts = np.concatenate((np.array([0]), np.flatnonzero(np.diff(frames) != 1) + 1))
    ends = np.concatenate((starts[1:], np.array([len(frames)])))
    predicted = sequence.prediction_joints_world_m[prediction_rows]
    reference = sequence.gt_joints_world_m[gt_rows]
    world = compute_joint_position_errors(predicted, reference).reshape(-1)
    local = compute_joint_position_errors(
        predicted - predicted[:, :1], reference - reference[:, :1],
    ).reshape(-1)
    pa = compute_pa_joint_position_errors(predicted, reference).reshape(-1)
    oks_sum, oks_count = track_oks_statistics(sequence, gt_rows, prediction_rows)
    predicted_acc: list[float] = []
    reference_acc: list[float] = []
    ranges: list[tuple[int, int]] = []
    for start, end in zip(starts, ends, strict=True):
        first, last = int(start), int(end)
        ranges.append((int(frames[first]), int(frames[last - 1]) + 1))
        predicted_acc.extend(float(value) for value in root_acceleration_magnitudes(
            predicted[first:last, 0],
        ))
        reference_acc.extend(float(value) for value in root_acceleration_magnitudes(
            reference[first:last, 0],
        ))
    summary = VirtualCrowd_Naive_Comparison_Sequence_Summary(
        profile_id=VC_NAIVE_COMPARISON_PROFILE_ID,
        filtering_id=filtering_id,
        split_id=split_id,
        scene_id=sequence.scene_id,
        selected_gt_count=len(gt_rows),
        matched_selected_count=len(prediction_rows),
        mpjpe_world_sum_m=math.fsum(float(value) for value in world),
        mpjpe_world_count=len(world),
        t_mpjpe_sum_m=math.fsum(float(value) for value in local),
        t_mpjpe_count=len(local),
        oks_vis_sum=oks_sum,
        oks_vis_count=oks_count,
        acc_root_predicted_sum_m_per_frame2=math.fsum(predicted_acc),
        acc_root_reference_sum_m_per_frame2=math.fsum(reference_acc),
        acc_root_sample_count=len(predicted_acc),
        pa_mpjpe_sum_m=math.fsum(float(value) for value in pa),
        pa_mpjpe_count=len(pa),
    )
    return Naive_Track_Statistics(int(sequence.gt_track_ids[gt_rows[0]]), tuple(ranges), summary)


def evaluate_naive_tracks(
    sequence: Corrected_Crowd_Sequence,
    filtering_id: str,
    split_id: str,
    selected_gt_mask: NDArray[np.generic],
) -> tuple[Naive_Track_Statistics, ...]:
    '''Validate/join once, then evaluate selected GT tracks, not runner chunks.

    Empty selection returns no records. Gaps split temporal support within each
    GT track; changing prediction local IDs never creates an evaluation boundary.
    '''
    require_identity(filtering_id, 'filtering_id')
    require_identity(split_id, 'split_id')
    join = direct_target_join(sequence, selected_gt_mask)
    sequence = join.sequence
    track_ids = sequence.gt_track_ids[join.gt_rows]
    order = np.lexsort((sequence.gt_frame_ids[join.gt_rows], track_ids))
    sorted_ids = track_ids[order]
    if not len(order):
        return ()
    boundaries = np.flatnonzero(np.diff(sorted_ids) != 0) + 1
    groups = np.split(order, boundaries)
    return tuple(summarize_naive_track_rows(
        sequence, join.gt_rows[group], join.prediction_rows[group], filtering_id, split_id,
    ) for group in groups)


def merge_naive_statistics(
    summaries: Sequence[VirtualCrowd_Naive_Comparison_Sequence_Summary],
    scene_id: str,
) -> VirtualCrowd_Naive_Comparison_Sequence_Summary:
    '''Merge compatible additive statistics; caller owns identity disjointness.

    Repeated scene IDs are intentional for track partitions. Statistics alone
    cannot prove disjoint GT identities; storage must validate those identities.
    '''
    require_identity(scene_id, 'scene_id')
    if not summaries:
        raise ValueError('naive statistics collection is empty')
    if any(type(item) is not VirtualCrowd_Naive_Comparison_Sequence_Summary for item in summaries):
        raise TypeError('summaries must contain naive comparison summaries')
    if any(item.scene_id != scene_id for item in summaries):
        raise ValueError('statistics require the requested scene ID')
    first = summaries[0]
    if any((item.profile_id, item.filtering_id, item.split_id) != (
        first.profile_id, first.filtering_id, first.split_id,
    ) for item in summaries):
        raise ValueError('statistics require the same profile, filtering and split')
    pa_sum = None
    pa_count = None
    if first.profile_id == VC_NAIVE_COMPARISON_PROFILE_V2_ID:
        pa_sum = math.fsum(
            cast(float, item.pa_mpjpe_sum_m) for item in summaries)
        pa_count = sum(
            cast(int, item.pa_mpjpe_count) for item in summaries)
    return VirtualCrowd_Naive_Comparison_Sequence_Summary(
        profile_id=first.profile_id, filtering_id=first.filtering_id,
        split_id=first.split_id, scene_id=scene_id,
        selected_gt_count=sum(item.selected_gt_count for item in summaries),
        matched_selected_count=sum(item.matched_selected_count for item in summaries),
        mpjpe_world_sum_m=math.fsum(item.mpjpe_world_sum_m for item in summaries),
        mpjpe_world_count=sum(item.mpjpe_world_count for item in summaries),
        t_mpjpe_sum_m=math.fsum(item.t_mpjpe_sum_m for item in summaries),
        t_mpjpe_count=sum(item.t_mpjpe_count for item in summaries),
        oks_vis_sum=math.fsum(item.oks_vis_sum for item in summaries),
        oks_vis_count=sum(item.oks_vis_count for item in summaries),
        acc_root_predicted_sum_m_per_frame2=math.fsum(
            item.acc_root_predicted_sum_m_per_frame2 for item in summaries),
        acc_root_reference_sum_m_per_frame2=math.fsum(
            item.acc_root_reference_sum_m_per_frame2 for item in summaries),
        acc_root_sample_count=sum(item.acc_root_sample_count for item in summaries),
        pa_mpjpe_sum_m=pa_sum,
        pa_mpjpe_count=pa_count,
    )


def finalize_naive_statistics(
    summary: VirtualCrowd_Naive_Comparison_Sequence_Summary,
) -> dict[str, float | None]:
    '''Return existing metric field names, with null for unsupported quantities.'''
    if type(summary) is not VirtualCrowd_Naive_Comparison_Sequence_Summary:
        raise TypeError('summary must be a naive comparison summary')
    return {
        'mpjpe_world_mm': 1000.0 * summary.mpjpe_world_sum_m / summary.mpjpe_world_count
            if summary.mpjpe_world_count else None,
        't_mpjpe_mm': 1000.0 * summary.t_mpjpe_sum_m / summary.t_mpjpe_count
            if summary.t_mpjpe_count else None,
        **({'pa_mpjpe_mm': 1000.0 * cast(float, summary.pa_mpjpe_sum_m)
                / summary.pa_mpjpe_count
            if summary.pa_mpjpe_count else None}
            if summary.profile_id == VC_NAIVE_COMPARISON_PROFILE_V2_ID else {}),
        'oks_vis': summary.oks_vis_sum / summary.oks_vis_count
            if summary.oks_vis_count else None,
        'acc_root_ratio': summary.acc_root_predicted_sum_m_per_frame2
            / summary.acc_root_reference_sum_m_per_frame2
            if summary.acc_root_sample_count and summary.acc_root_reference_sum_m_per_frame2 > 0.0
            else None,
    }
