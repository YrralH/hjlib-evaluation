'''Per-frame COCO17 OKS association profiles.'''
# pyright: reportMissingTypeStubs=false
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportUnknownVariableType]

from hjlib_evaluation.keypoint_oks import compute_keypoint_oks_matrix


COCO17_OKS_SIGMAS = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
    0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
], dtype=np.float64) / 10.0
STANDARD_OKS_THRESHOLD = 0.5
AUTHOR_OKS_THRESHOLD = 1e-6
OKS_QUANTIZATION = 1_000_000_000_000
Person_OKS_Matching_Profile = Literal[
    'standard-matching-v1', 'crowd4d-author-greedy-v1',
]
STANDARD_MATCHING_PROFILE: Person_OKS_Matching_Profile = \
    'standard-matching-v1'
CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE: Person_OKS_Matching_Profile = \
    'crowd4d-author-greedy-v1'


@dataclass(frozen=True, slots=True)
class Person_OKS_Association:
    '''One full partition whose aligned pair order is profile-defined.'''

    match_gt_indices: NDArray[np.int64]
    match_prediction_indices: NDArray[np.int64]
    matched_oks: NDArray[np.float64]
    false_negative_gt_indices: NDArray[np.int64]
    false_positive_prediction_indices: NDArray[np.int64]
    projection_valid_predictions: NDArray[np.bool_]
    solver_call_count: int = 0

    def __post_init__(self) -> None:
        integer_names = (
            'match_gt_indices', 'match_prediction_indices',
            'false_negative_gt_indices',
            'false_positive_prediction_indices',
        )
        for name in integer_names:
            raw = np.asarray(getattr(self, name))
            if not np.issubdtype(raw.dtype, np.integer):
                raise TypeError('%s must have integer dtype' % name)
            array = np.array(
                raw, dtype=np.int64, copy=True, order='C',
            )
            if array.ndim != 1 or np.any(array < 0) \
                    or len(np.unique(array)) != len(array):
                raise ValueError('%s must contain unique nonnegative indices' % name)
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        matched_oks = np.array(
            self.matched_oks, dtype=np.float64, copy=True, order='C',
        )
        projection_raw = np.asarray(self.projection_valid_predictions)
        if projection_raw.dtype != np.bool_:
            raise TypeError('projection_valid_predictions must have bool dtype')
        projection_valid = np.array(
            projection_raw,
            dtype=np.bool_, copy=True, order='C',
        )
        if matched_oks.shape != self.match_gt_indices.shape \
                or self.match_prediction_indices.shape != self.match_gt_indices.shape \
                or not np.isfinite(matched_oks).all() \
                or np.any((matched_oks < 0.0) | (matched_oks > 1.0)):
            raise ValueError('matched association arrays are invalid')
        if projection_valid.ndim != 1 or type(self.solver_call_count) is not int \
                or self.solver_call_count < 0:
            raise ValueError('projection mask or solver count is invalid')
        gt_partition = np.concatenate((
            self.match_gt_indices, self.false_negative_gt_indices,
        ))
        prediction_partition = np.concatenate((
            self.match_prediction_indices,
            self.false_positive_prediction_indices,
        ))
        if not contiguous_partition(gt_partition) \
                or not contiguous_partition(prediction_partition) \
                or len(projection_valid) != len(prediction_partition) \
                or not np.all(projection_valid[self.match_prediction_indices]):
            raise ValueError('matched/FN/FP indices must form full partitions')
        matched_oks.flags.writeable = False
        projection_valid.flags.writeable = False
        object.__setattr__(self, 'matched_oks', matched_oks)
        object.__setattr__(self, 'projection_valid_predictions', projection_valid)


def project_camera_points(
        xyz: NDArray[np.generic],
        camera_K: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    '''Project person-major joints; a person is valid only at positive depth.'''
    points = np.asarray(xyz, dtype=np.float64)
    intrinsics = np.asarray(camera_K, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError('xyz must have shape (P, J, 3)')
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() \
            or not np.isfinite(points).all():
        raise ValueError('camera points and intrinsics must be finite')
    valid = np.all(points[:, :, 2] > 0.0, axis=1)
    projected = np.zeros(points.shape[:2] + (2,), dtype=np.float64)
    if np.any(valid):
        homogeneous = points[valid] @ intrinsics.T
        projected[valid] = homogeneous[:, :, :2] / homogeneous[:, :, 2:3]
    if not np.isfinite(projected).all():
        raise ValueError('projected points must be finite')
    return projected, np.asarray(valid, dtype=np.bool_)


def solve_cardinality_quality_assignment(
        admissible: NDArray[np.bool_],
        quantized_oks: NDArray[np.int64],
    ) -> tuple[int, int, NDArray[np.int64]]:
    '''Return maximum cardinality, integer quality, and one optimizer mapping.'''
    edges = np.asarray(admissible)
    quality = np.asarray(quantized_oks)
    if edges.dtype != np.bool_ or edges.ndim != 2 \
            or quality.dtype != np.int64 or quality.shape != edges.shape \
            or np.any(quality < 0) or np.any(quality > OKS_QUANTIZATION):
        raise ValueError('assignment objective arrays are invalid')
    row_count, prediction_count = edges.shape
    mapping = np.full(row_count, -1, dtype=np.int64)
    if row_count == 0 or prediction_count == 0:
        return 0, 0, mapping
    edge_bound = min(row_count, prediction_count)
    cardinality_weight = edge_bound * OKS_QUANTIZATION + 1
    score = np.zeros((row_count, prediction_count + row_count), dtype=np.int64)
    score[:, :prediction_count] = -cardinality_weight
    real_scores = cardinality_weight + quality
    score[:, :prediction_count][edges] = real_scores[edges]
    assignment = cast(Callable[..., tuple[
        NDArray[np.int64], NDArray[np.int64],
    ]], linear_sum_assignment)
    rows, columns = assignment(score, maximize=True)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        if column < prediction_count and edges[row, column]:
            mapping[row] = column
    matched_rows = np.flatnonzero(mapping >= 0)
    return (
        int(len(matched_rows)),
        int(sum(int(quality[row, mapping[row]]) for row in matched_rows)),
        mapping,
    )


def associate_cardinality_first(
        oks: NDArray[np.generic],
        admissible: NDArray[np.bool_],
        projection_valid: NDArray[np.bool_] | None = None,
    ) -> Person_OKS_Association:
    '''Apply cardinality and quantized OKS on one fixed ordered assignment.'''
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
    _, prediction_count = similarities.shape
    valid = (
        np.ones(prediction_count, dtype=np.bool_)
        if projection_valid is None else np.asarray(projection_valid)
    )
    if valid.dtype != np.bool_ or valid.shape != (prediction_count,):
        raise ValueError('projection_valid must have shape (P,) and bool dtype')
    edges = np.asarray(edges & valid[None, :], dtype=np.bool_)
    _, _, fixed = solve_cardinality_quality_assignment(
        edges, quantized,
    )
    solver_calls = int(similarities.shape[0] > 0 and prediction_count > 0)
    matched_gt = np.flatnonzero(fixed >= 0).astype(np.int64)
    return make_association_from_pairs(
        similarities,
        matched_gt,
        fixed[matched_gt].astype(np.int64),
        valid,
        solver_calls,
    )


def match_coco17_people(
        gt_xyv: NDArray[np.generic],
        pred_xyz_camera: NDArray[np.generic],
        gt_bboxes_xyxy: NDArray[np.generic],
        camera_K: NDArray[np.generic],
        matching_profile: Person_OKS_Matching_Profile = \
            STANDARD_MATCHING_PROFILE,
    ) -> Person_OKS_Association:
    '''Match one frame through one named COCO17 association profile.'''
    if matching_profile == STANDARD_MATCHING_PROFILE:
        return match_standard_coco17(
            gt_xyv, pred_xyz_camera, gt_bboxes_xyxy, camera_K,
        )
    if matching_profile == CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE:
        return match_crowd4d_author_greedy_coco17(
            gt_xyv, pred_xyz_camera, gt_bboxes_xyxy, camera_K,
        )
    raise ValueError('unsupported person OKS matching profile')


def match_standard_coco17(
        gt_xyv: NDArray[np.generic],
        pred_xyz_camera: NDArray[np.generic],
        gt_bboxes_xyxy: NDArray[np.generic],
        camera_K: NDArray[np.generic],
    ) -> Person_OKS_Association:
    '''Match one frame using the thresholded standard protocol.'''
    gt, boxes = validate_inputs(gt_xyv, pred_xyz_camera, gt_bboxes_xyxy)
    pred_xy, projection_valid = project_camera_points(pred_xyz_camera, camera_K)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if np.any(areas <= 0.0):
        raise ValueError('standard matching requires positive GT bbox area')
    visible = np.asarray(gt[:, :, 2] > 0.0, dtype=np.bool_)
    supported_gt = np.flatnonzero(np.any(visible, axis=1)).astype(np.int64)
    oks = compute_keypoint_oks_matrix(
        gt[supported_gt, :, :2], pred_xy, areas[supported_gt],
        COCO17_OKS_SIGMAS, visible[supported_gt],
    )
    admissible = (oks >= STANDARD_OKS_THRESHOLD) \
        & projection_valid[None, :]
    supported = associate_cardinality_first(
        oks, admissible, projection_valid,
    )
    matched_gt = supported_gt[supported.match_gt_indices]
    false_negative = np.setdiff1d(
        np.arange(len(gt), dtype=np.int64), matched_gt, assume_unique=True,
    )
    return Person_OKS_Association(
        matched_gt,
        supported.match_prediction_indices,
        supported.matched_oks,
        false_negative,
        supported.false_positive_prediction_indices,
        supported.projection_valid_predictions,
        supported.solver_call_count,
    )


def match_crowd4d_author_greedy_coco17(
        gt_xyv: NDArray[np.generic],
        pred_xyz_camera: NDArray[np.generic],
        gt_bboxes_xyxy: NDArray[np.generic],
        camera_K: NDArray[np.generic],
    ) -> Person_OKS_Association:
    '''Reproduce Crowd4D's per-frame first-minimum greedy association.'''
    gt, boxes = validate_inputs(gt_xyv, pred_xyz_camera, gt_bboxes_xyxy)
    pred_xy, projection_valid = project_camera_points(pred_xyz_camera, camera_K)
    areas = np.maximum(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        np.spacing(1),
    )
    visible = np.ones(gt.shape[:2], dtype=np.bool_)
    oks = compute_keypoint_oks_matrix(
        gt[:, :, :2], pred_xy, areas, COCO17_OKS_SIGMAS, visible,
    )
    if len(gt) == 0 or len(pred_xy) == 0:
        empty = np.empty(0, dtype=np.int64)
        return make_association_from_pairs(
            oks, empty, empty, projection_valid,
        )
    distance = 1.0 - oks
    distance[:, ~projection_valid] = np.inf
    minimum = np.min(distance, axis=1)
    order = np.argsort(minimum)
    use_count = np.zeros(len(pred_xy), dtype=np.int64)
    matched_gt: list[int] = []
    matched_prediction: list[int] = []
    for gt_index in order:
        best = minimum[gt_index]
        if best < 1.0 - AUTHOR_OKS_THRESHOLD:
            prediction_index = int(np.where(distance[gt_index] == best)[0][0])
            if projection_valid[prediction_index] \
                    and use_count[prediction_index] == 0:
                matched_gt.append(int(gt_index))
                matched_prediction.append(prediction_index)
            use_count[prediction_index] += 1
    return make_association_from_pairs(
        oks,
        np.asarray(matched_gt, dtype=np.int64),
        np.asarray(matched_prediction, dtype=np.int64),
        projection_valid,
    )


def validate_inputs(
        gt_xyv: NDArray[np.generic],
        pred_xyz_camera: NDArray[np.generic],
        gt_bboxes_xyxy: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    gt = np.asarray(gt_xyv, dtype=np.float64)
    pred = np.asarray(pred_xyz_camera, dtype=np.float64)
    boxes = np.asarray(gt_bboxes_xyxy, dtype=np.float64)
    if gt.ndim != 3 or gt.shape[1:] != (17, 3):
        raise ValueError('gt_xyv must have shape (G, 17, 3)')
    if pred.ndim != 3 or pred.shape[1:] != (17, 3):
        raise ValueError('pred_xyz_camera must have shape (P, 17, 3)')
    if boxes.shape != (len(gt), 4):
        raise ValueError('gt_bboxes_xyxy must have shape (G, 4)')
    if not np.isfinite(gt).all() or not np.isfinite(pred).all() \
            or not np.isfinite(boxes).all():
        raise ValueError('matching inputs must be finite')
    return gt, boxes


def make_association_from_pairs(
        oks: NDArray[np.float64],
        match_gt: NDArray[np.int64],
        match_pred: NDArray[np.int64],
        projection_valid: NDArray[np.bool_],
        solver_call_count: int = 0,
    ) -> Person_OKS_Association:
    matched_oks = oks[match_gt, match_pred].astype(np.float64)
    false_negative = np.setdiff1d(
        np.arange(oks.shape[0], dtype=np.int64), match_gt,
        assume_unique=True,
    )
    false_positive = np.setdiff1d(
        np.arange(oks.shape[1], dtype=np.int64), match_pred, assume_unique=True,
    )
    arrays = (
        match_gt, match_pred, matched_oks, false_negative, false_positive,
        projection_valid,
    )
    for array in arrays:
        array.flags.writeable = False
    return Person_OKS_Association(*arrays, solver_call_count)


def contiguous_partition(indices: NDArray[np.int64]) -> bool:
    return len(np.unique(indices)) == len(indices) \
        and np.array_equal(np.sort(indices), np.arange(len(indices)))


__all__ = [
    'CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE', 'Person_OKS_Association',
    'Person_OKS_Matching_Profile', 'STANDARD_MATCHING_PROFILE',
    'match_coco17_people', 'project_camera_points',
]
