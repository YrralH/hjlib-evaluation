'''Corrected two-view crowd evaluation and exact micro reduction.'''
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import (
    CORRECTED_CROWD_METRICS,
    CORRECTED_CROWD_SCHEMA_VERSION,
    CORRECTED_CROWD_VIEWS,
    Corrected_Crowd_Result,
    Corrected_Crowd_Sequence,
    Corrected_Crowd_Sequence_Summary,
    validate_corrected_crowd_sequence,
)
from hjlib_evaluation.crowd_layout import (
    compute_pcod_3class_matches,
    compute_ppds_scores,
)
from hjlib_evaluation.joint_acceleration import compute_joint_acceleration_errors
from hjlib_evaluation.joint_error import compute_joint_position_errors
from hjlib_evaluation.keypoint_oks import compute_keypoint_oks_matrix
from hjlib_geometry import (
    apply_mean_translation,
    apply_rigid_registration,
    apply_similarity_registration,
    fit_mean_translation,
    fit_rigid_registration,
    fit_similarity_registration,
)


COCO17_SIGMAS = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
    0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
], dtype=np.float64) / 10.0
METRIC_INDEX = {name: index for index, name in enumerate(CORRECTED_CROWD_METRICS)}


def add_metric_values(
    sums: NDArray[np.float64],
    counts: NDArray[np.int64],
    view_index: int,
    metric_name: str,
    values: NDArray[np.generic],
) -> None:
    '''Add one finite non-negative micro population to sufficient statistics.'''
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError('%s metric population must be finite and non-negative' % metric_name)
    metric_index = METRIC_INDEX[metric_name]
    sums[view_index, metric_index] += math.fsum(float(value) for value in array)
    counts[view_index, metric_index] += int(array.size)


def build_visrun_labels(sequence: Corrected_Crowd_Sequence) -> NDArray[np.int64]:
    '''Label GT-visible rows by maximal exact-consecutive track run.'''
    visible = np.any(sequence.gt_visibility_native > 0.0, axis=1)
    labels = np.full(len(visible), -1, dtype=np.int64)
    next_label = 0
    for track_id in np.unique(sequence.gt_track_ids[visible]):
        rows = np.flatnonzero(visible & (sequence.gt_track_ids == track_id))
        rows = rows[np.argsort(sequence.gt_frame_ids[rows], kind='stable')]
        previous_frame: int | None = None
        for row in rows:
            frame_id = int(sequence.gt_frame_ids[row])
            if previous_frame is None or frame_id != previous_frame + 1:
                next_label += 1
            labels[row] = next_label
            previous_frame = frame_id
    return labels


def aligned_joint_errors(
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    mode: str,
) -> NDArray[np.float64]:
    '''Fit one scope transform and return per-point Euclidean errors.'''
    flat_predicted = predicted.reshape(-1, 3)
    flat_reference = reference.reshape(-1, 3)
    mask = np.ones(len(flat_predicted), dtype=np.bool_)
    if mode == 'T':
        fit = fit_mean_translation(flat_predicted, flat_reference, mask)
        aligned = apply_mean_translation(flat_predicted, fit)
    elif mode == 'RT':
        fit = fit_rigid_registration(flat_predicted, flat_reference, mask)
        aligned = apply_rigid_registration(flat_predicted, fit)
    elif mode == 'PA':
        fit = fit_similarity_registration(flat_predicted, flat_reference, mask)
        aligned = apply_similarity_registration(flat_predicted, fit)
    else:
        raise ValueError('unknown alignment mode: %s' % mode)
    return compute_joint_position_errors(aligned, flat_reference)


def add_frame_joint_metrics(
    sequence: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    view_index: int,
    sums: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> None:
    '''Add frame-level absolute, pelvis, rigid, and similarity populations.'''
    predicted = sequence.prediction_joints_world_m[match_pred]
    reference = sequence.gt_joints_world_m[match_gt]
    add_metric_values(
        sums, counts, view_index, 'MPJPE-WORLD',
        compute_joint_position_errors(predicted, reference),
    )
    predicted_local = predicted - predicted[:, :1]
    reference_local = reference - reference[:, :1]
    add_metric_values(
        sums, counts, view_index, 'T-MPJPE',
        compute_joint_position_errors(predicted_local, reference_local),
    )
    for metric_name, mode in (('RT-MPJPE', 'RT'), ('PA-MPJPE', 'PA')):
        values = [
            aligned_joint_errors(predicted[index], reference[index], mode)
            for index in range(len(predicted))
        ]
        if values:
            add_metric_values(
                sums,
                counts,
                view_index,
                metric_name,
                np.concatenate(values),
            )


def add_sequence_joint_metrics(
    sequence: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    visrun_labels: NDArray[np.int64],
    view_index: int,
    sums: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> None:
    '''Add VISRUN and TRACK one-fit-per-scope joint populations.'''
    scope_specs = (
        ('VISRUN', visrun_labels[match_gt]),
        ('TRACK', sequence.gt_track_ids[match_gt]),
    )
    for scope_name, labels in scope_specs:
        for mode in ('T', 'RT', 'PA'):
            populations: list[NDArray[np.float64]] = []
            for label in np.unique(labels):
                selected = labels == label
                predicted = sequence.prediction_joints_world_m[match_pred[selected]]
                reference = sequence.gt_joints_world_m[match_gt[selected]]
                populations.append(aligned_joint_errors(predicted, reference, mode))
            if populations:
                add_metric_values(
                    sums,
                    counts,
                    view_index,
                    'SEQ-%s-MPJPE-%s' % (mode, scope_name),
                    np.concatenate(populations),
                )


def add_frame_layout_and_oks_metrics(
    sequence: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    view_index: int,
    sums: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> None:
    '''Add pair-layout and visibility-aware OKS populations by frame.'''
    for frame_id in np.unique(sequence.gt_frame_ids[match_gt]):
        selected = sequence.gt_frame_ids[match_gt] == frame_id
        gt_rows = match_gt[selected]
        pred_rows = match_pred[selected]
        predicted_anchor = sequence.prediction_joints_world_m[pred_rows, 0]
        reference_anchor = sequence.gt_joints_world_m[gt_rows, 0]
        ppds = compute_ppds_scores(predicted_anchor, reference_anchor)
        add_metric_values(sums, counts, view_index, 'PPDS', ppds)
        if len(gt_rows) >= 2:
            fit = fit_similarity_registration(
                predicted_anchor,
                reference_anchor,
                np.ones(len(gt_rows), dtype=np.bool_),
            )
            pa_ppds = compute_ppds_scores(
                predicted_anchor,
                reference_anchor,
                scale=fit.scale_target_to_reference,
            )
            add_metric_values(sums, counts, view_index, 'PA-PPDS', pa_ppds)
        pcod = compute_pcod_3class_matches(
            sequence.prediction_pelvis_camera_depth_m[pred_rows],
            sequence.gt_pelvis_camera_depth_m[gt_rows],
            0.3,
        )
        add_metric_values(sums, counts, view_index, 'PCOD-3C-0.3m', pcod)
        valid = sequence.gt_visibility_native[gt_rows] > 0.0
        bboxes = sequence.gt_bbox_xyxy_px[gt_rows]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        oks_matrix = compute_keypoint_oks_matrix(
            sequence.gt_coco17_xy_px[gt_rows],
            sequence.prediction_coco17_xy_px[pred_rows],
            areas,
            COCO17_SIGMAS,
            valid,
        )
        add_metric_values(
            sums,
            counts,
            view_index,
            'OKS-VIS',
            np.diag(oks_matrix),
        )


def add_acceleration_metrics(
    sequence: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    visrun_labels: NDArray[np.int64],
    view_index: int,
    sums: NDArray[np.float64],
    counts: NDArray[np.int64],
) -> int:
    '''Add exact-consecutive VISRUN acceleration and return triple count.'''
    populations: list[NDArray[np.float64]] = []
    triple_count = 0
    labels = visrun_labels[match_gt]
    for label in np.unique(labels):
        selected = np.flatnonzero(labels == label)
        order = np.argsort(sequence.gt_frame_ids[match_gt[selected]], kind='stable')
        selected = selected[order]
        frames = sequence.gt_frame_ids[match_gt[selected]]
        for center in range(1, len(selected) - 1):
            if frames[center] - frames[center - 1] != 1:
                continue
            if frames[center + 1] - frames[center] != 1:
                continue
            triple = selected[center - 1:center + 2]
            populations.append(compute_joint_acceleration_errors(
                sequence.prediction_joints_world_m[match_pred[triple]],
                sequence.gt_joints_world_m[match_gt[triple]],
            ))
            triple_count += 1
    if populations:
        add_metric_values(
            sums,
            counts,
            view_index,
            'ACCEL-WORLD',
            np.concatenate(populations),
        )
    return triple_count


def evaluate_corrected_crowd_sequence(
    sequence: Corrected_Crowd_Sequence,
) -> Corrected_Crowd_Sequence_Summary:
    '''Evaluate one normalized scene into exact sufficient statistics.'''
    data = validate_corrected_crowd_sequence(sequence)
    visible = np.any(data.gt_visibility_native > 0.0, axis=1)
    targets = data.prediction_identity_target_gt_rows
    mapped_invisible = (targets >= 0) & ~visible[np.maximum(targets, 0)]
    prediction_in_scope = ~mapped_invisible
    tp = len(data.matched_gt_rows)
    fn = int(np.count_nonzero(visible)) - tp
    fp = int(np.count_nonzero(prediction_in_scope)) - tp
    if fn < 0 or fp < 0:
        raise ValueError('association completeness counts are inconsistent')

    sums = np.zeros(
        (len(CORRECTED_CROWD_VIEWS), len(CORRECTED_CROWD_METRICS)),
        dtype=np.float64,
    )
    counts = np.zeros(sums.shape, dtype=np.int64)
    triples = np.zeros((len(CORRECTED_CROWD_VIEWS),), dtype=np.int64)
    visrun_labels = build_visrun_labels(data)
    for view_index in range(len(CORRECTED_CROWD_VIEWS)):
        selected = np.ones(tp, dtype=np.bool_)
        if view_index == 1:
            selected = data.common_gt_mask[data.matched_gt_rows]
        match_gt = data.matched_gt_rows[selected]
        match_pred = data.matched_prediction_rows[selected]
        add_frame_joint_metrics(data, match_gt, match_pred, view_index, sums, counts)
        add_sequence_joint_metrics(
            data, match_gt, match_pred, visrun_labels, view_index, sums, counts,
        )
        add_frame_layout_and_oks_metrics(
            data, match_gt, match_pred, view_index, sums, counts,
        )
        triples[view_index] = add_acceleration_metrics(
            data, match_gt, match_pred, visrun_labels, view_index, sums, counts,
        )
    return Corrected_Crowd_Sequence_Summary(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id=data.scene_id,
        tp=tp,
        fn=fn,
        fp=fp,
        metric_sample_sums=sums,
        metric_sample_counts=counts,
        accel_exact_consecutive_triple_count=triples,
    )


def reduce_corrected_crowd_summaries(
    summaries: Sequence[Corrected_Crowd_Sequence_Summary],
) -> Corrected_Crowd_Result:
    '''Reduce scene sufficient statistics in lexical scene order.'''
    ordered = tuple(sorted(summaries, key=lambda item: item.scene_id))
    if not ordered:
        raise ValueError('corrected crowd summary collection is empty')
    if len({item.scene_id for item in ordered}) != len(ordered):
        raise ValueError('corrected crowd scene IDs must be unique')
    tp = sum(item.tp for item in ordered)
    fn = sum(item.fn for item in ordered)
    fp = sum(item.fp for item in ordered)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    counts = np.sum(
        np.stack([item.metric_sample_counts for item in ordered]),
        axis=0,
        dtype=np.int64,
    )
    sums = np.zeros(counts.shape, dtype=np.float64)
    for view_index in range(len(CORRECTED_CROWD_VIEWS)):
        for metric_index in range(len(CORRECTED_CROWD_METRICS)):
            sums[view_index, metric_index] = math.fsum(
                float(item.metric_sample_sums[view_index, metric_index])
                for item in ordered
            )
    values: list[tuple[float | None, ...]] = []
    scaled_indices = set(range(10)) | {METRIC_INDEX['ACCEL-WORLD']}
    for view_index in range(len(CORRECTED_CROWD_VIEWS)):
        row: list[float | None] = []
        for metric_index in range(len(CORRECTED_CROWD_METRICS)):
            count = int(counts[view_index, metric_index])
            if count == 0:
                row.append(None)
                continue
            value = sums[view_index, metric_index] / count
            if metric_index in scaled_indices:
                value *= 1000.0
            row.append(value)
        values.append(tuple(row))
    triple_counts = tuple(
        sum(
            int(item.accel_exact_consecutive_triple_count[view_index])
            for item in ordered
        )
        for view_index in range(len(CORRECTED_CROWD_VIEWS))
    )
    return Corrected_Crowd_Result(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        tp=tp,
        fn=fn,
        fp=fp,
        precision=precision,
        recall=recall,
        f1=f1,
        metric_values=tuple(values),
        accel_exact_consecutive_triple_count=triple_counts,
    )
