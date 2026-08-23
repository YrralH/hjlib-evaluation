'''Deterministic OKS association and reduction for unordered JTA people.'''
# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import cast
import json
import math

from hjlib_geometry import (
    apply_similarity_registration,
    fit_similarity_registration,
)
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]

from hjlib_evaluation.joint_error import compute_joint_position_errors
from hjlib_evaluation.jta_person_detection_data import (
    JTA_ENDPOINT_OKS_SIGMAS,
    JTA_Person_Detection_GT_Frame,
    JTA_Person_Detection_Prediction_Frame,
    JTA_Person_Detection_Result,
    validate_sha256,
)
from hjlib_evaluation.keypoint_oks import compute_keypoint_oks_matrix


OKS_THRESHOLD = 0.50
OKS_QUANTIZATION = 1_000_000_000_000


@dataclass(frozen=True, slots=True)
class JTA_Person_Association:
    '''One deterministic threshold-aware GT/prediction assignment.'''

    match_gt_indices: NDArray[np.int64]
    match_prediction_indices: NDArray[np.int64]
    matched_oks: NDArray[np.float64]
    solver_call_count: int


@dataclass(frozen=True, slots=True)
class JTA_Person_Detection_Frame_Metrics:
    '''Sufficient statistics for one fully evaluated frame.'''

    association: JTA_Person_Association
    gt_person_count: int
    prediction_person_count: int
    projection_invalid_prediction_count: int
    absolute_mpjpe_person_sum_mm: float
    pelvis_mpjpe_person_sum_mm: float
    pa_mpjpe_person_sum_mm: float
    pa_valid_person_count: int
    pa_degenerate_person_count: int


def solve_assignment_objective(
        admissible: NDArray[np.bool_],
        quantized_oks: NDArray[np.int64],
) -> tuple[int, int, NDArray[np.int64]]:
    '''Return maximum cardinality, integer OKS sum, and one optimizer mapping.'''
    row_count, prediction_count = admissible.shape
    mapping = np.full(row_count, -1, dtype=np.int64)
    if row_count == 0 or prediction_count == 0:
        return 0, 0, mapping
    edge_bound = min(row_count, prediction_count)
    cardinality_weight = edge_bound * OKS_QUANTIZATION + 1
    score = np.zeros(
        (row_count, prediction_count + row_count), dtype=np.int64,
    )
    score[:, :prediction_count] = -cardinality_weight
    real_scores = cardinality_weight + quantized_oks
    score[:, :prediction_count][admissible] = real_scores[admissible]
    assignment = cast(Callable[..., tuple[
        NDArray[np.int64], NDArray[np.int64],
    ]], linear_sum_assignment)
    rows, columns = assignment(score, maximize=True)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < prediction_count and admissible[row, column]:
            mapping[row] = column
    matched_rows = np.flatnonzero(mapping >= 0)
    cardinality = int(len(matched_rows))
    quality = int(sum(
        int(quantized_oks[row, mapping[row]]) for row in matched_rows
    ))
    return cardinality, quality, mapping


def associate_jta_people(
        oks: NDArray[np.generic],
        admissible: NDArray[np.bool_],
) -> JTA_Person_Association:
    '''Apply cardinality, quantized-OKS, then explicit row-lex objectives.'''
    similarities = np.asarray(oks, dtype=np.float64)
    edges = np.asarray(admissible)
    if similarities.ndim != 2 or edges.dtype != np.bool_ \
            or edges.shape != similarities.shape \
            or not np.isfinite(similarities).all() \
            or np.any((similarities < 0.0) | (similarities > 1.0)):
        raise ValueError('OKS association arrays are invalid')
    quantized = np.asarray(
        np.rint(similarities * OKS_QUANTIZATION), dtype=np.int64,
    )
    gt_count, prediction_count = similarities.shape
    available = list(range(prediction_count))
    fixed_mapping = np.full(gt_count, -1, dtype=np.int64)
    solver_calls = 0
    remaining_cardinality, remaining_quality, _ = solve_assignment_objective(
        edges, quantized,
    )
    solver_calls += 1
    for gt_index in range(gt_count):
        remaining_rows = np.arange(gt_index, gt_count, dtype=np.int64)
        available_array = np.asarray(available, dtype=np.int64)
        current_edges = edges[np.ix_(remaining_rows, available_array)]
        current_quantized = quantized[np.ix_(remaining_rows, available_array)]
        current_cardinality, current_quality, current_mapping = \
            solve_assignment_objective(current_edges, current_quantized)
        solver_calls += 1
        if current_cardinality != remaining_cardinality \
                or current_quality != remaining_quality:
            raise RuntimeError('assignment refinement lost the optimum')
        local_choice = int(current_mapping[0])
        current_choice = (
            available[local_choice] if local_choice >= 0 else prediction_count
        )
        chosen = current_choice
        for candidate in available:
            if candidate >= current_choice:
                break
            if not edges[gt_index, candidate]:
                continue
            rest_rows = np.arange(gt_index + 1, gt_count, dtype=np.int64)
            rest_predictions = np.asarray(
                [value for value in available if value != candidate],
                dtype=np.int64,
            )
            rest_edges = edges[np.ix_(rest_rows, rest_predictions)]
            rest_quantized = quantized[np.ix_(rest_rows, rest_predictions)]
            rest_cardinality, rest_quality, _ = solve_assignment_objective(
                rest_edges, rest_quantized,
            )
            solver_calls += 1
            if rest_cardinality + 1 == remaining_cardinality \
                    and rest_quality + int(quantized[gt_index, candidate]) \
                    == remaining_quality:
                chosen = candidate
                break
        if chosen < prediction_count:
            fixed_mapping[gt_index] = chosen
            available.remove(chosen)
            remaining_cardinality -= 1
            remaining_quality -= int(quantized[gt_index, chosen])
    if remaining_cardinality != 0 or remaining_quality != 0:
        raise RuntimeError('assignment refinement did not exhaust the optimum')
    match_gt = np.flatnonzero(fixed_mapping >= 0).astype(np.int64)
    match_prediction = fixed_mapping[match_gt].astype(np.int64)
    matched_oks = similarities[match_gt, match_prediction].astype(np.float64)
    for array in (match_gt, match_prediction, matched_oks):
        array.flags.writeable = False
    return JTA_Person_Association(
        match_gt_indices=match_gt,
        match_prediction_indices=match_prediction,
        matched_oks=matched_oks,
        solver_call_count=solver_calls,
    )


def project_predictions(
        xyz: NDArray[np.float64],
        camera_K: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    valid = np.all(xyz[:, :, 2] > 0.0, axis=1)
    projected = np.zeros((xyz.shape[0], xyz.shape[1], 2), dtype=np.float64)
    if np.any(valid):
        camera = xyz[valid] @ camera_K.T
        projected[valid] = camera[:, :, :2] / camera[:, :, 2:3]
    if not np.isfinite(projected).all():
        raise ValueError('JTA prediction projection is non-finite')
    return projected, np.asarray(valid, dtype=np.bool_)


def evaluate_jta_person_detection_frame(
        gt: JTA_Person_Detection_GT_Frame,
        prediction: JTA_Person_Detection_Prediction_Frame,
) -> JTA_Person_Detection_Frame_Metrics:
    '''Evaluate one exact frame after method-neutral 3D normalization.'''
    if (gt.scene_id, gt.frame_id) != (prediction.scene_id, prediction.frame_id):
        raise ValueError('GT and prediction frame keys differ')
    pred_xy, projection_valid = project_predictions(
        prediction.pred_xyz_camera, gt.camera_K,
    )
    bbox = gt.gt_bbox_xyxy
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    oks = compute_keypoint_oks_matrix(
        gt.gt_xy,
        pred_xy,
        areas,
        JTA_ENDPOINT_OKS_SIGMAS,
        gt.gt_visible,
    )
    admissible = (
        (oks >= OKS_THRESHOLD)
        & np.any(gt.gt_visible, axis=1)[:, None]
        & projection_valid[None, :]
    )
    association = associate_jta_people(oks, admissible)
    match_gt = association.match_gt_indices
    match_pred = association.match_prediction_indices
    absolute_sum = 0.0
    pelvis_sum = 0.0
    pa_sum = 0.0
    pa_valid_count = 0
    pa_degenerate_count = 0
    mask = np.ones(12, dtype=np.bool_)
    for gt_index, prediction_index in zip(
            match_gt.tolist(), match_pred.tolist()):
        reference = gt.gt_xyz_camera[gt_index]
        target = prediction.pred_xyz_camera[prediction_index]
        absolute_sum += float(np.mean(
            compute_joint_position_errors(target, reference),
        )) * 1000.0
        target_pelvis = np.mean(target[6:8], axis=0)
        reference_pelvis = np.mean(reference[6:8], axis=0)
        pelvis_sum += float(np.mean(compute_joint_position_errors(
            target - target_pelvis,
            reference - reference_pelvis,
        ))) * 1000.0
        target_centered = target - np.mean(target, axis=0)
        reference_centered = reference - np.mean(reference, axis=0)
        if float(np.sum(target_centered * target_centered)) == 0.0 \
                or float(np.sum(reference_centered * reference_centered)) == 0.0:
            pa_degenerate_count += 1
            continue
        fit = fit_similarity_registration(target, reference, mask)
        aligned = apply_similarity_registration(target, fit)
        pa_sum += float(np.mean(
            compute_joint_position_errors(aligned, reference),
        )) * 1000.0
        pa_valid_count += 1
    return JTA_Person_Detection_Frame_Metrics(
        association=association,
        gt_person_count=len(gt.gt_source_ids),
        prediction_person_count=len(prediction.prediction_row_ids),
        projection_invalid_prediction_count=int(
            np.count_nonzero(~projection_valid),
        ),
        absolute_mpjpe_person_sum_mm=absolute_sum,
        pelvis_mpjpe_person_sum_mm=pelvis_sum,
        pa_mpjpe_person_sum_mm=pa_sum,
        pa_valid_person_count=pa_valid_count,
        pa_degenerate_person_count=pa_degenerate_count,
    )


class JTA_Person_Detection_Reducer:
    '''Exactly-once reducer for one expected ordered frame population.'''

    def __init__(
            self,
            expected_frame_keys: tuple[tuple[str, int], ...],
            prediction_source_sha256: str,
            prediction_profile_sha256: str,
    ) -> None:
        if not expected_frame_keys or len(set(expected_frame_keys)) \
                != len(expected_frame_keys) or any(
                    not scene or type(frame) is not int or frame < 0
                    for scene, frame in expected_frame_keys
                ):
            raise ValueError('expected frame keys must be nonempty and unique')
        self.expected_frame_keys = expected_frame_keys
        self.prediction_source_sha256 = validate_sha256(
            prediction_source_sha256, 'prediction_source_sha256',
        )
        self.prediction_profile_sha256 = validate_sha256(
            prediction_profile_sha256, 'prediction_profile_sha256',
        )
        self.index_next = 0
        self.sealed = False
        self.frame_digests: list[tuple[str, str]] = []
        self.gt_count = 0
        self.prediction_count = 0
        self.match_count = 0
        self.projection_invalid_count = 0
        self.pa_degenerate_count = 0
        self.oks_sum = 0.0
        self.absolute_sum = 0.0
        self.pelvis_sum = 0.0
        self.pa_sum = 0.0
        self.pa_valid_count = 0

    def add_frame(
            self,
            gt: JTA_Person_Detection_GT_Frame,
            prediction: JTA_Person_Detection_Prediction_Frame,
    ) -> None:
        if self.sealed:
            raise RuntimeError('JTA person-detection reducer is sealed')
        if self.index_next >= len(self.expected_frame_keys):
            raise ValueError('JTA person-detection reducer received an extra frame')
        expected = self.expected_frame_keys[self.index_next]
        if (gt.scene_id, gt.frame_id) != expected \
                or (prediction.scene_id, prediction.frame_id) != expected:
            raise ValueError('JTA person-detection frame is out of order')
        if prediction.prediction_source_sha256 \
                != self.prediction_source_sha256 \
                or prediction.prediction_profile_sha256 \
                != self.prediction_profile_sha256:
            raise ValueError('prediction identity differs from reducer binding')
        metrics = evaluate_jta_person_detection_frame(gt, prediction)
        matched_count = len(metrics.association.match_gt_indices)
        values = (
            float(np.sum(metrics.association.matched_oks)),
            metrics.absolute_mpjpe_person_sum_mm,
            metrics.pelvis_mpjpe_person_sum_mm,
            metrics.pa_mpjpe_person_sum_mm,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError('JTA person-detection metric sum is invalid')
        self.frame_digests.append((gt.semantic_sha256, prediction.semantic_sha256))
        self.gt_count += metrics.gt_person_count
        self.prediction_count += metrics.prediction_person_count
        self.match_count += matched_count
        self.projection_invalid_count += metrics.projection_invalid_prediction_count
        self.pa_degenerate_count += metrics.pa_degenerate_person_count
        self.oks_sum += values[0]
        self.absolute_sum += values[1]
        self.pelvis_sum += values[2]
        self.pa_sum += values[3]
        self.pa_valid_count += metrics.pa_valid_person_count
        self.index_next += 1

    def finalize(self) -> JTA_Person_Detection_Result:
        if self.sealed:
            raise RuntimeError('JTA person-detection reducer is sealed')
        if self.index_next != len(self.expected_frame_keys):
            raise RuntimeError('JTA person-detection reducer has an incomplete prefix')
        digest_payload = {
            'expected_frame_keys': [list(key) for key in self.expected_frame_keys],
            'prediction_source_sha256': self.prediction_source_sha256,
            'prediction_profile_sha256': self.prediction_profile_sha256,
            'frame_digests': [list(value) for value in self.frame_digests],
        }
        input_digest = sha256(
            b'hjlib_evaluation.jta_person_detection_inputs.v1\0'
            + json.dumps(
                digest_payload, sort_keys=True, separators=(',', ':'),
            ).encode('utf-8'),
        ).hexdigest()
        self.sealed = True
        return JTA_Person_Detection_Result(
            expected_frame_keys=self.expected_frame_keys,
            prediction_source_sha256=self.prediction_source_sha256,
            prediction_profile_sha256=self.prediction_profile_sha256,
            frame_count=len(self.expected_frame_keys),
            gt_person_count=self.gt_count,
            prediction_person_count=self.prediction_count,
            matched_person_count=self.match_count,
            unmatched_gt_count=self.gt_count - self.match_count,
            unmatched_prediction_count=self.prediction_count - self.match_count,
            projection_invalid_prediction_count=self.projection_invalid_count,
            pa_degenerate_person_count=self.pa_degenerate_count,
            matched_oks_sum=self.oks_sum,
            absolute_mpjpe_person_sum_mm=self.absolute_sum,
            pelvis_mpjpe_person_sum_mm=self.pelvis_sum,
            pa_mpjpe_person_sum_mm=self.pa_sum,
            pa_valid_person_count=self.pa_valid_count,
            input_digest_sha256=input_digest,
        )


__all__ = [
    'JTA_Person_Association', 'JTA_Person_Detection_Frame_Metrics',
    'JTA_Person_Detection_Reducer', 'OKS_QUANTIZATION', 'OKS_THRESHOLD',
    'associate_jta_people', 'evaluate_jta_person_detection_frame',
]
