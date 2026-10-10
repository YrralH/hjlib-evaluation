'''Paired SMPL22 validation with explicit complete support and pooled statistics.'''

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.joint_error import (
    compute_joint_position_errors, compute_pa_joint_position_errors, validate_joint_points,
)


BODY_RECONSTRUCTION_VALIDATION_SCHEMA = 'body_reconstruction_validation.v2'
type Shape_Condition = Literal['predicted_shape', 'given_gt_shape']
SHAPE_CONDITIONS: tuple[Shape_Condition, ...] = ('predicted_shape', 'given_gt_shape')


def body_validation_metric_contract() -> dict[str, Any]:
    return {'joint_layout': 'SMPL22', 'joint_indices': list(range(22)), 'root_joint': 0,
        'root_included_in_mean': True,
        'coordinate_frame': 'camera_no_trans', 'input_unit': 'm', 'report_unit': 'mm',
        'alignment': {'t_mpjpe': 'each_body_pelvis0',
            'pa_mpjpe': 'per_person_frame_positive_scale_proper_similarity'}}


def validate_body_metric_sums_record(value: dict[str, Any], expected_count: int) -> None:
    total, count, mean = value['sum_m'], value['joint_count'], value['mean_mm']
    if type(count) is not int or count != expected_count or count <= 0 \
            or not isinstance(total, (int, float)) or not isinstance(mean, (int, float)) \
            or not math.isfinite(total) or not math.isfinite(mean) or total < 0 or mean < 0 \
            or not math.isclose(mean, total * 1000. / count, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('validation metric has invalid finite sum/count/mean')


def validate_body_validation_result(result: dict[str, Any], expected_frames: int) -> None:
    '''Reject conflicting definitions, incomplete support and inconsistent statistics.'''
    if result.get('schema') != BODY_RECONSTRUCTION_VALIDATION_SCHEMA \
            or result.get('person_frame_count') != expected_frames:
        raise ValueError('validation result schema/support differs')
    for name, value in body_validation_metric_contract().items():
        if result.get(name) != value:
            raise ValueError('validation metric contract differs: %s' % name)
    if set(result['conditions']) != set(SHAPE_CONDITIONS):
        raise ValueError('validation must contain exactly both shape conditions')
    paired_support: dict[str, int] | None = None
    for condition in SHAPE_CONDITIONS:
        row = result['conditions'][condition]
        if not row['per_dataset']:
            raise ValueError('per-dataset statistics are required')
        support = {dataset: values['t_mpjpe']['joint_count']
            for dataset, values in row['per_dataset'].items()}
        if paired_support is not None and support != paired_support:
            raise ValueError('shape conditions must have identical per-dataset support')
        paired_support = support
        for dataset, values in row['per_dataset'].items():
            count = support[dataset]
            if type(count) is not int or count <= 0 or count % 22 != 0 \
                    or values['pa_mpjpe']['joint_count'] != count:
                raise ValueError('dataset T/PA support must match full SMPL22 occurrences')
        for metric in ('t_mpjpe', 'pa_mpjpe'):
            pooled = row['metrics'][metric]
            validate_body_metric_sums_record(pooled, expected_frames * 22)
            values = [dataset[metric] for dataset in row['per_dataset'].values()]
            for item in values:
                validate_body_metric_sums_record(item, item['joint_count'])
            if sum(item['joint_count'] for item in values) != pooled['joint_count'] \
                    or not math.isclose(sum(item['sum_m'] for item in values), pooled['sum_m'],
                        rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('per-dataset statistics differ from pooled statistics')


@dataclass(frozen=True)
class Body_Validation_Metric_Sums:
    t_mpjpe_sum_m: float
    pa_mpjpe_sum_m: float
    joint_count: int

    def to_record(self) -> dict[str, Any]:
        if self.joint_count <= 0:
            raise ValueError('metric support must be nonempty')
        return {'t_mpjpe': {'sum_m': self.t_mpjpe_sum_m, 'joint_count': self.joint_count,
                'mean_mm': self.t_mpjpe_sum_m * 1000.0 / self.joint_count},
            'pa_mpjpe': {'sum_m': self.pa_mpjpe_sum_m, 'joint_count': self.joint_count,
                'mean_mm': self.pa_mpjpe_sum_m * 1000.0 / self.joint_count}}


def compute_body_validation_metric_sums(
    predicted: NDArray[np.generic], reference: NDArray[np.generic],
) -> Body_Validation_Metric_Sums:
    '''Select native SMPL joints 0--21 before either centering or PA fitting.'''
    if predicted.ndim != 3 or predicted.shape[1:] != (24, 3) or reference.shape != predicted.shape:
        raise ValueError('paired joints must have shape (N,24,3)')
    selected_predicted = validate_joint_points(predicted[:, :22], 'predicted')
    selected_reference = validate_joint_points(reference[:, :22], 'reference')
    if not np.isfinite(selected_predicted).all() or not np.isfinite(selected_reference).all():
        raise ValueError('validation joints must be finite; no rows may be dropped')
    t_values = compute_joint_position_errors(
        selected_predicted - selected_predicted[:, :1],
        selected_reference - selected_reference[:, :1])
    pa_values = compute_pa_joint_position_errors(selected_predicted, selected_reference)
    return Body_Validation_Metric_Sums(float(t_values.sum()), float(pa_values.sum()), t_values.size)


class Body_Reconstruction_Validation_Reducer:
    '''Track unique paired occurrences and sufficient statistics, not predictions.'''

    def __init__(self, expected_frames: int) -> None:
        if expected_frames <= 0:
            raise ValueError('expected_frames must be positive')
        self.expected_frames = expected_frames
        self.seen: set[tuple[str, str]] = set()
        self.sums: dict[tuple[Shape_Condition, str], Body_Validation_Metric_Sums] = {}

    def add_batch(self, occurrence_ids: Sequence[str], datasets: Sequence[str],
        predicted_shape: NDArray[np.generic], given_gt_shape: NDArray[np.generic],
        reference: NDArray[np.generic]) -> None:
        count = len(occurrence_ids)
        if count == 0 or len(datasets) != count or reference.shape != (count, 24, 3):
            raise ValueError('paired occurrence metadata and joints must have equal nonempty support')
        keys = list(zip(datasets, occurrence_ids, strict=True))
        if any(not dataset or not identity for dataset, identity in keys):
            raise ValueError('dataset and occurrence IDs must be nonempty')
        if len(set(keys)) != count or self.seen.intersection(keys):
            raise ValueError('duplicate validation occurrence')
        if len(self.seen) + count > self.expected_frames:
            raise ValueError('validation exceeds declared support')
        if predicted_shape.shape != reference.shape or given_gt_shape.shape != reference.shape:
            raise ValueError('both shape conditions must match reference support')
        pending: dict[tuple[Shape_Condition, str], Body_Validation_Metric_Sums] = {}
        names = np.asarray(datasets)
        for condition, prediction in zip(SHAPE_CONDITIONS, (predicted_shape, given_gt_shape), strict=True):
            for dataset in sorted(set(datasets)):
                mask = names == dataset
                pending[condition, dataset] = compute_body_validation_metric_sums(prediction[mask], reference[mask])
        for key, value in pending.items():
            previous = self.sums.get(key, Body_Validation_Metric_Sums(0., 0., 0))
            self.sums[key] = Body_Validation_Metric_Sums(
                previous.t_mpjpe_sum_m + value.t_mpjpe_sum_m,
                previous.pa_mpjpe_sum_m + value.pa_mpjpe_sum_m,
                previous.joint_count + value.joint_count)
        self.seen.update(keys)

    def finalize(self) -> dict[str, Any]:
        if len(self.seen) != self.expected_frames:
            raise ValueError('incomplete validation: %d/%d frames' % (len(self.seen), self.expected_frames))
        conditions: dict[str, Any] = {}
        for condition in SHAPE_CONDITIONS:
            per_dataset = {dataset: value.to_record()
                for (arm, dataset), value in self.sums.items() if arm == condition}
            values = [value for (arm, _), value in self.sums.items() if arm == condition]
            pooled = Body_Validation_Metric_Sums(sum(value.t_mpjpe_sum_m for value in values),
                sum(value.pa_mpjpe_sum_m for value in values), sum(value.joint_count for value in values))
            if pooled.joint_count != self.expected_frames * 22:
                raise ValueError('incomplete joint support')
            conditions[condition] = {'metrics': pooled.to_record(), 'per_dataset': per_dataset}
        return {'schema': BODY_RECONSTRUCTION_VALIDATION_SCHEMA, 'person_frame_count': len(self.seen),
            **body_validation_metric_contract(),
            'conditions': conditions}


__all__ = ['BODY_RECONSTRUCTION_VALIDATION_SCHEMA', 'SHAPE_CONDITIONS', 'Shape_Condition',
    'body_validation_metric_contract', 'validate_body_validation_result',
    'Body_Validation_Metric_Sums', 'compute_body_validation_metric_sums',
    'Body_Reconstruction_Validation_Reducer']
