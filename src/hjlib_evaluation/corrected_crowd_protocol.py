'''Corrected two-view crowd evaluation and exact micro reduction.'''
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.corrected_crowd_data import (
    CORRECTED_CROWD_METRICS,
    CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
    CORRECTED_CROWD_SCHEMA_VERSION,
    CORRECTED_CROWD_VIEWS,
    Corrected_Crowd_Result,
    Corrected_Crowd_Selected_View_Result,
    Corrected_Crowd_Selected_View_Sequence_Summary,
    Corrected_Crowd_Sequence,
    Corrected_Crowd_Sequence_Summary,
    bool_array,
    validate_corrected_crowd_sequence,
    validate_corrected_crowd_selected_view_name,
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
    precomputed: tuple[NDArray[np.float64], int] | None = None,
) -> int:
    '''Add exact-consecutive VISRUN acceleration and return triple count.'''
    if precomputed is None:
        errors, triple_count = collect_exact_acceleration_errors(
            sequence,
            match_gt,
            match_pred,
            visrun_labels,
        )
    else:
        errors, triple_count = precomputed
        errors = np.asarray(errors, dtype=np.float64)
        if errors.shape != (triple_count, 24):
            raise ValueError('precomputed acceleration population has wrong shape')
    if errors.size:
        add_metric_values(
            sums,
            counts,
            view_index,
            'ACCEL-WORLD',
            errors,
        )
    return triple_count


def iter_exact_matched_segments(
    sequence: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    visrun_labels: NDArray[np.int64],
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    '''Return stable maximal exact-consecutive matched segments.'''
    segments: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    labels = visrun_labels[match_gt]
    for label in np.unique(labels):
        selected = np.flatnonzero(labels == label)
        order = np.argsort(sequence.gt_frame_ids[match_gt[selected]], kind='stable')
        selected = selected[order]
        frames = sequence.gt_frame_ids[match_gt[selected]]
        starts = np.concatenate((
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.diff(frames) != 1).astype(np.int64) + 1,
        ))
        ends = np.concatenate((starts[1:], np.array([len(selected)], dtype=np.int64)))
        for start, end in zip(starts, ends, strict=True):
            rows = selected[int(start):int(end)]
            segments.append((match_gt[rows], match_pred[rows]))
    return segments


def collect_exact_acceleration_errors(
    sequence: Corrected_Crowd_Sequence,
    match_gt: NDArray[np.int64],
    match_pred: NDArray[np.int64],
    visrun_labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], int]:
    '''Collect temporal-major SMPL-24 acceleration residuals.'''
    populations: list[NDArray[np.float64]] = []
    triple_count = 0
    for gt_rows, pred_rows in iter_exact_matched_segments(
        sequence,
        match_gt,
        match_pred,
        visrun_labels,
    ):
        if len(gt_rows) < 3:
            continue
        errors = compute_joint_acceleration_errors(
            sequence.prediction_joints_world_m[pred_rows],
            sequence.gt_joints_world_m[gt_rows],
        )
        populations.append(errors)
        triple_count += len(errors)
    if not populations:
        return np.empty((0, 24), dtype=np.float64), 0
    return np.concatenate(populations, axis=0), triple_count


def evaluate_corrected_crowd_matched_rows(
    validated_sequence: Corrected_Crowd_Sequence,
    matched_gt_rows: NDArray[np.int64],
    matched_prediction_rows: NDArray[np.int64],
    base_visible_visrun_labels: NDArray[np.int64],
    precomputed_acceleration: tuple[NDArray[np.float64], int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int]:
    '''Evaluate one exact matched-row population using base-visible run labels.'''
    match_gt = np.asarray(matched_gt_rows, dtype=np.int64)
    match_pred = np.asarray(matched_prediction_rows, dtype=np.int64)
    if match_gt.ndim != 1 or match_pred.shape != match_gt.shape:
        raise ValueError('matched row arrays must have the same one-dimensional shape')
    labels = np.asarray(base_visible_visrun_labels, dtype=np.int64)
    if labels.shape != (len(validated_sequence.gt_frame_ids),):
        raise ValueError('base-visible VISRUN labels have the wrong shape')
    sums = np.zeros((1, len(CORRECTED_CROWD_METRICS)), dtype=np.float64)
    counts = np.zeros(sums.shape, dtype=np.int64)
    add_frame_joint_metrics(
        validated_sequence, match_gt, match_pred, 0, sums, counts,
    )
    add_sequence_joint_metrics(
        validated_sequence, match_gt, match_pred, labels, 0, sums, counts,
    )
    add_frame_layout_and_oks_metrics(
        validated_sequence, match_gt, match_pred, 0, sums, counts,
    )
    triples = add_acceleration_metrics(
        validated_sequence,
        match_gt,
        match_pred,
        labels,
        0,
        sums,
        counts,
        precomputed_acceleration,
    )
    return sums[0], counts[0], triples


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

    sums_rows: list[NDArray[np.float64]] = []
    counts_rows: list[NDArray[np.int64]] = []
    triples = np.zeros((len(CORRECTED_CROWD_VIEWS),), dtype=np.int64)
    visrun_labels = build_visrun_labels(data)
    for view_index in range(len(CORRECTED_CROWD_VIEWS)):
        selected = np.ones(tp, dtype=np.bool_)
        if view_index == 1:
            selected = data.common_gt_mask[data.matched_gt_rows]
        match_gt = data.matched_gt_rows[selected]
        match_pred = data.matched_prediction_rows[selected]
        row_sums, row_counts, triple_count = evaluate_corrected_crowd_matched_rows(
            data,
            match_gt,
            match_pred,
            visrun_labels,
        )
        sums_rows.append(row_sums)
        counts_rows.append(row_counts)
        triples[view_index] = triple_count
    return Corrected_Crowd_Sequence_Summary(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id=data.scene_id,
        tp=tp,
        fn=fn,
        fp=fp,
        metric_sample_sums=np.stack(sums_rows),
        metric_sample_counts=np.stack(counts_rows),
        accel_exact_consecutive_triple_count=triples,
    )


def evaluate_corrected_crowd_selected_view(
    sequence: Corrected_Crowd_Sequence,
    view_name: str,
    selected_gt_mask: NDArray[np.generic],
) -> Corrected_Crowd_Selected_View_Sequence_Summary:
    '''Evaluate one explicit subset without redefining base-visible run labels.'''
    data = validate_corrected_crowd_sequence(sequence)
    validated_name = validate_corrected_crowd_selected_view_name(view_name)
    selected_mask = bool_array(selected_gt_mask, 'selected_gt_mask')
    if selected_mask.shape != (len(data.gt_frame_ids),):
        raise ValueError('selected_gt_mask must have shape [G]')
    base_visible = np.any(data.gt_visibility_native > 0.0, axis=1)
    if np.any(selected_mask & ~base_visible):
        raise ValueError('selected_gt_mask must be a subset of GT-visible rows')
    matched_selected = selected_mask[data.matched_gt_rows]
    match_gt = data.matched_gt_rows[matched_selected]
    match_pred = data.matched_prediction_rows[matched_selected]
    sums, counts, triples = evaluate_corrected_crowd_matched_rows(
        data,
        match_gt,
        match_pred,
        build_visrun_labels(data),
    )
    return Corrected_Crowd_Selected_View_Sequence_Summary(
        schema_version=CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
        scene_id=data.scene_id,
        view_name=validated_name,
        selected_gt_count=int(np.count_nonzero(selected_mask)),
        matched_selected_count=len(match_gt),
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


def reduce_corrected_crowd_selected_view_summaries(
    summaries: Sequence[Corrected_Crowd_Selected_View_Sequence_Summary],
) -> Corrected_Crowd_Selected_View_Result:
    '''Reduce selected-view sufficient statistics in lexical scene order.'''
    ordered = tuple(sorted(summaries, key=lambda item: item.scene_id))
    if not ordered:
        raise ValueError('selected corrected crowd summary collection is empty')
    if len({item.scene_id for item in ordered}) != len(ordered):
        raise ValueError('selected corrected crowd scene IDs must be unique')
    view_name = ordered[0].view_name
    if any(item.view_name != view_name for item in ordered):
        raise ValueError('selected corrected crowd view names must match')
    counts = np.sum(
        np.stack([item.metric_sample_counts for item in ordered]),
        axis=0,
        dtype=np.int64,
    )
    sums = np.array([
        math.fsum(float(item.metric_sample_sums[index]) for item in ordered)
        for index in range(len(CORRECTED_CROWD_METRICS))
    ], dtype=np.float64)
    scaled_indices = set(range(10)) | {METRIC_INDEX['ACCEL-WORLD']}
    values: list[float | None] = []
    for metric_index in range(len(CORRECTED_CROWD_METRICS)):
        count = int(counts[metric_index])
        if count == 0:
            values.append(None)
            continue
        value = sums[metric_index] / count
        if metric_index in scaled_indices:
            value *= 1000.0
        values.append(value)
    return Corrected_Crowd_Selected_View_Result(
        schema_version=CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION,
        view_name=view_name,
        selected_gt_count=sum(item.selected_gt_count for item in ordered),
        matched_selected_count=sum(item.matched_selected_count for item in ordered),
        metric_values=tuple(values),
        accel_exact_consecutive_triple_count=sum(
            item.accel_exact_consecutive_triple_count for item in ordered
        ),
    )
