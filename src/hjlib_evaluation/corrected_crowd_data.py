'''Stable normalized data and serialization for corrected crowd evaluation.'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from hjlib_evaluation.crowd_layout import compute_ppds_scores
from hjlib_geometry import fit_similarity_registration


CORRECTED_CROWD_SCHEMA_VERSION = 1
CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION = 1
CORRECTED_CROWD_VIEWS = ('GT_VISIBLE', 'C4D_DYCROWD_COMMON')
CORRECTED_CROWD_METRICS = (
    'MPJPE-WORLD',
    'T-MPJPE',
    'RT-MPJPE',
    'PA-MPJPE',
    'SEQ-T-MPJPE-VISRUN',
    'SEQ-RT-MPJPE-VISRUN',
    'SEQ-PA-MPJPE-VISRUN',
    'SEQ-T-MPJPE-TRACK',
    'SEQ-RT-MPJPE-TRACK',
    'SEQ-PA-MPJPE-TRACK',
    'PPDS',
    'PA-PPDS',
    'PCOD-3C-0.3m',
    'OKS-VIS',
    'ACCEL-WORLD',
)
CORRECTED_CROWD_METRIC_UNITS = (
    'mm', 'mm', 'mm', 'mm',
    'mm', 'mm', 'mm', 'mm', 'mm', 'mm',
    'fraction', 'fraction', 'fraction', 'fraction', 'mm/frame^2',
)
UNMAPPED_GT_ROW = -1

Float_Array = NDArray[np.float64]
Int_Array = NDArray[np.int64]
Bool_Array = NDArray[np.bool_]
JSON_Object = dict[str, Any]


def immutable_array(value: NDArray[np.generic], dtype: np.dtype[Any]) -> NDArray[Any]:
    '''Return an owned immutable array with the exact requested dtype.'''
    copied = np.array(value, dtype=dtype, copy=True)
    return np.frombuffer(copied.tobytes(), dtype=dtype).reshape(copied.shape)


def float_array(value: NDArray[np.generic], name: str) -> Float_Array:
    '''Return an immutable finite float64 array.'''
    source = np.asarray(value)
    if not np.issubdtype(source.dtype, np.number):
        raise TypeError('%s must have numeric dtype' % name)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError('%s must have real numeric dtype' % name)
    array = np.asarray(source, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('%s must be finite' % name)
    return cast(Float_Array, immutable_array(array, np.dtype(np.float64)))


def int_array(value: NDArray[np.generic], name: str) -> Int_Array:
    '''Return an immutable exact-integer int64 array.'''
    source = np.asarray(value)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError('%s must have integer dtype' % name)
    array = np.asarray(source, dtype=np.int64)
    return cast(Int_Array, immutable_array(array, np.dtype(np.int64)))


def bool_array(value: NDArray[np.generic], name: str) -> Bool_Array:
    '''Return an immutable exact-boolean array.'''
    source = np.asarray(value)
    if source.dtype != np.bool_:
        raise TypeError('%s must have boolean dtype' % name)
    return cast(Bool_Array, immutable_array(source, np.dtype(np.bool_)))


def json_int_array(value: Any, name: str) -> Int_Array:
    '''Parse a JSON array without permitting float-to-int truncation.'''
    array = cast(NDArray[np.generic], np.asarray(value))
    return int_array(array, name)


def require_shape(array: NDArray[Any], shape: tuple[int, ...], name: str) -> None:
    '''Require one exact array shape.'''
    if array.shape != shape:
        raise ValueError('%s must have shape %s, got %s' % (name, shape, array.shape))


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_Sequence:
    '''One immutable normalized scene for corrected crowd evaluation.'''

    schema_version: int
    scene_id: str
    frame_domain: Int_Array
    gt_frame_ids: Int_Array
    gt_track_ids: Int_Array
    gt_joints_world_m: Float_Array
    gt_coco17_xy_px: Float_Array
    gt_visibility_native: Float_Array
    gt_bbox_xyxy_px: Float_Array
    gt_pelvis_camera_depth_m: Float_Array
    prediction_frame_ids: Int_Array
    prediction_local_track_ids: Int_Array
    prediction_joints_world_m: Float_Array
    prediction_coco17_xy_px: Float_Array
    prediction_coco17_camera_depth_m: Float_Array
    prediction_pelvis_camera_depth_m: Float_Array
    prediction_identity_target_gt_rows: Int_Array
    matched_gt_rows: Int_Array
    matched_prediction_rows: Int_Array
    common_gt_mask: Bool_Array
    coordinate_frame: str = 'FIXED_CAMERA_WORLD_EQUIVALENT'
    length_unit: str = 'metre'
    camera_depth_axis: str = 'POSITIVE_Z_AWAY_FROM_CAMERA'
    smpl_joint_order: str = 'SMPL_24'
    coco_joint_order: str = 'COCO_17'

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_SCHEMA_VERSION:
            raise ValueError('unsupported corrected crowd schema version')
        if not self.scene_id:
            raise ValueError('scene_id must be a non-empty string')
        literals = {
            'coordinate_frame': 'FIXED_CAMERA_WORLD_EQUIVALENT',
            'length_unit': 'metre',
            'camera_depth_axis': 'POSITIVE_Z_AWAY_FROM_CAMERA',
            'smpl_joint_order': 'SMPL_24',
            'coco_joint_order': 'COCO_17',
        }
        for name, expected in literals.items():
            if getattr(self, name) != expected:
                raise ValueError('%s must be %s' % (name, expected))
        self.normalize_arrays()
        self.validate_relations()

    def normalize_arrays(self) -> None:
        '''Install owned immutable arrays after shape-independent conversion.'''
        float_names = (
            'gt_joints_world_m', 'gt_coco17_xy_px', 'gt_visibility_native',
            'gt_bbox_xyxy_px', 'gt_pelvis_camera_depth_m',
            'prediction_joints_world_m', 'prediction_coco17_xy_px',
            'prediction_coco17_camera_depth_m',
            'prediction_pelvis_camera_depth_m',
        )
        int_names = (
            'frame_domain', 'gt_frame_ids', 'gt_track_ids',
            'prediction_frame_ids', 'prediction_local_track_ids',
            'prediction_identity_target_gt_rows', 'matched_gt_rows',
            'matched_prediction_rows',
        )
        for name in float_names:
            object.__setattr__(self, name, float_array(getattr(self, name), name))
        for name in int_names:
            object.__setattr__(self, name, int_array(getattr(self, name), name))
        object.__setattr__(
            self,
            'common_gt_mask',
            bool_array(self.common_gt_mask, 'common_gt_mask'),
        )

    def validate_relations(self) -> None:
        '''Validate shapes, keys, visibility, and association relations.'''
        frame_count = len(self.frame_domain)
        gt_count = len(self.gt_frame_ids)
        pred_count = len(self.prediction_frame_ids)
        match_count = len(self.matched_gt_rows)
        require_shape(self.frame_domain, (frame_count,), 'frame_domain')
        if frame_count and np.any(np.diff(self.frame_domain) <= 0):
            raise ValueError('frame_domain must be strictly increasing')
        require_shape(self.gt_track_ids, (gt_count,), 'gt_track_ids')
        require_shape(self.gt_joints_world_m, (gt_count, 24, 3), 'gt_joints_world_m')
        require_shape(self.gt_coco17_xy_px, (gt_count, 17, 2), 'gt_coco17_xy_px')
        require_shape(self.gt_visibility_native, (gt_count, 17), 'gt_visibility_native')
        require_shape(self.gt_bbox_xyxy_px, (gt_count, 4), 'gt_bbox_xyxy_px')
        require_shape(
            self.gt_pelvis_camera_depth_m,
            (gt_count,),
            'gt_pelvis_camera_depth_m',
        )
        require_shape(
            self.prediction_local_track_ids,
            (pred_count,),
            'prediction_local_track_ids',
        )
        require_shape(
            self.prediction_joints_world_m,
            (pred_count, 24, 3),
            'prediction_joints_world_m',
        )
        require_shape(
            self.prediction_coco17_xy_px,
            (pred_count, 17, 2),
            'prediction_coco17_xy_px',
        )
        require_shape(
            self.prediction_coco17_camera_depth_m,
            (pred_count, 17),
            'prediction_coco17_camera_depth_m',
        )
        require_shape(
            self.prediction_pelvis_camera_depth_m,
            (pred_count,),
            'prediction_pelvis_camera_depth_m',
        )
        require_shape(
            self.prediction_identity_target_gt_rows,
            (pred_count,),
            'prediction_identity_target_gt_rows',
        )
        require_shape(self.matched_prediction_rows, (match_count,), 'matched_prediction_rows')
        require_shape(self.common_gt_mask, (gt_count,), 'common_gt_mask')
        self.validate_values_and_keys()

    def validate_values_and_keys(self) -> None:
        '''Validate values and cross-array association invariants.'''
        if np.any(self.gt_track_ids <= 0):
            raise ValueError('gt_track_ids must be positive')
        if np.any(self.prediction_local_track_ids < 0):
            raise ValueError('prediction_local_track_ids must be non-negative')
        if not np.isin(self.gt_frame_ids, self.frame_domain).all():
            raise ValueError('all GT frame IDs must belong to frame_domain')
        if not np.isin(self.prediction_frame_ids, self.frame_domain).all():
            raise ValueError('all prediction frame IDs must belong to frame_domain')
        if len(set(zip(self.gt_frame_ids.tolist(), self.gt_track_ids.tolist()))) != len(
            self.gt_frame_ids,
        ):
            raise ValueError('GT person-frame keys must be unique')
        pred_keys = zip(
            self.prediction_frame_ids.tolist(),
            self.prediction_local_track_ids.tolist(),
        )
        if len(set(pred_keys)) != len(self.prediction_frame_ids):
            raise ValueError('prediction occurrence keys must be unique')
        if not np.isin(self.gt_visibility_native, (0.0, 0.5, 1.0)).all():
            raise ValueError('gt_visibility_native values must be 0, 0.5, or 1')
        widths = self.gt_bbox_xyxy_px[:, 2] - self.gt_bbox_xyxy_px[:, 0]
        heights = self.gt_bbox_xyxy_px[:, 3] - self.gt_bbox_xyxy_px[:, 1]
        if np.any(widths <= 0.0) or np.any(heights <= 0.0):
            raise ValueError('GT bbox width and height must be positive')
        visible = np.any(self.gt_visibility_native > 0.0, axis=1)
        if np.any(self.common_gt_mask & ~visible):
            raise ValueError('common_gt_mask must be a subset of GT-visible rows')
        targets = self.prediction_identity_target_gt_rows
        if np.any((targets < UNMAPPED_GT_ROW) | (targets >= len(self.gt_frame_ids))):
            raise ValueError('prediction identity target row is out of range')
        mapped = targets >= 0
        if np.any(
            self.prediction_frame_ids[mapped] != self.gt_frame_ids[targets[mapped]],
        ):
            raise ValueError('prediction identity target must have the same frame')
        if len(np.unique(self.matched_gt_rows)) != len(self.matched_gt_rows):
            raise ValueError('matched GT rows must be unique')
        if len(np.unique(self.matched_prediction_rows)) != len(
            self.matched_prediction_rows,
        ):
            raise ValueError('matched prediction rows must be unique')
        if np.any((self.matched_gt_rows < 0) | (self.matched_gt_rows >= len(visible))):
            raise ValueError('matched GT row is out of range')
        if np.any(
            (self.matched_prediction_rows < 0)
            | (self.matched_prediction_rows >= len(targets)),
        ):
            raise ValueError('matched prediction row is out of range')
        if np.any(~visible[self.matched_gt_rows]):
            raise ValueError('matched GT rows must be visible')
        if np.any(
            targets[self.matched_prediction_rows] != self.matched_gt_rows,
        ):
            raise ValueError('matched pair must equal its identity target')
        included_joint = self.gt_visibility_native[self.matched_gt_rows] > 0.0
        pred_depth = self.prediction_coco17_camera_depth_m[
            self.matched_prediction_rows
        ]
        if np.any(pred_depth[included_joint] <= 0.0):
            raise ValueError('included prediction projection depth must be positive')
        self.validate_pair_populations()

    def validate_pair_populations(self) -> None:
        '''Preflight consumed pair distances and PA-PPDS fit predicates.'''
        for common_only in (False, True):
            selected = np.ones(len(self.matched_gt_rows), dtype=np.bool_)
            if common_only:
                selected = self.common_gt_mask[self.matched_gt_rows]
            gt_rows_all = self.matched_gt_rows[selected]
            pred_rows_all = self.matched_prediction_rows[selected]
            for frame_id in np.unique(self.gt_frame_ids[gt_rows_all]):
                frame_selected = self.gt_frame_ids[gt_rows_all] == frame_id
                gt_rows = gt_rows_all[frame_selected]
                pred_rows = pred_rows_all[frame_selected]
                if len(gt_rows) < 2:
                    continue
                predicted = self.prediction_joints_world_m[pred_rows, 0]
                reference = self.gt_joints_world_m[gt_rows, 0]
                compute_ppds_scores(predicted, reference)
                fit_similarity_registration(
                    predicted,
                    reference,
                    np.ones(len(gt_rows), dtype=np.bool_),
                )


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_Sequence_Summary:
    '''Immutable sufficient statistics for one corrected scene.'''

    schema_version: int
    scene_id: str
    tp: int
    fn: int
    fp: int
    metric_sample_sums: Float_Array
    metric_sample_counts: Int_Array
    accel_exact_consecutive_triple_count: Int_Array

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_SCHEMA_VERSION:
            raise ValueError('unsupported corrected crowd schema version')
        if not self.scene_id:
            raise ValueError('scene_id must be non-empty')
        for name in ('tp', 'fn', 'fp'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('%s must be a non-negative exact int' % name)
        sums = float_array(self.metric_sample_sums, 'metric_sample_sums')
        counts = int_array(self.metric_sample_counts, 'metric_sample_counts')
        triples = int_array(
            self.accel_exact_consecutive_triple_count,
            'accel_exact_consecutive_triple_count',
        )
        shape = (len(CORRECTED_CROWD_VIEWS), len(CORRECTED_CROWD_METRICS))
        require_shape(sums, shape, 'metric_sample_sums')
        require_shape(counts, shape, 'metric_sample_counts')
        require_shape(triples, (len(CORRECTED_CROWD_VIEWS),), 'triple_count')
        if np.any(counts < 0) or np.any(triples < 0):
            raise ValueError('summary counts must be non-negative')
        if np.any((counts == 0) & (sums != 0.0)):
            raise ValueError('empty metric populations must have zero sum')
        if np.any(sums < 0.0):
            raise ValueError('metric sample sums must be non-negative')
        object.__setattr__(self, 'metric_sample_sums', sums)
        object.__setattr__(self, 'metric_sample_counts', counts)
        object.__setattr__(self, 'accel_exact_consecutive_triple_count', triples)


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_Result:
    '''Final two-view corrected result in display units.'''

    schema_version: int
    tp: int
    fn: int
    fp: int
    precision: float
    recall: float
    f1: float
    metric_values: tuple[tuple[float | None, ...], ...]
    accel_exact_consecutive_triple_count: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_SCHEMA_VERSION:
            raise ValueError('unsupported corrected crowd schema version')
        for name in ('tp', 'fn', 'fp'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('%s must be a non-negative exact int' % name)
        for name in ('precision', 'recall', 'f1'):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError('%s must be finite in [0,1]' % name)
        if len(self.metric_values) != len(CORRECTED_CROWD_VIEWS):
            raise ValueError('metric_values must have one row per view')
        for row in self.metric_values:
            if len(row) != len(CORRECTED_CROWD_METRICS):
                raise ValueError('metric_values row has wrong metric count')
            for value in row:
                if value is not None and not np.isfinite(value):
                    raise ValueError('metric values must be finite or None')
            for value in row[10:14]:
                if value is not None and not 0.0 <= value <= 1.0:
                    raise ValueError('dimensionless metric values must be in [0,1]')
        if len(self.accel_exact_consecutive_triple_count) != len(
            CORRECTED_CROWD_VIEWS,
        ):
            raise ValueError('triple count must have one value per view')
        if any(type(value) is not int or value < 0 for value in self.accel_exact_consecutive_triple_count):
            raise ValueError('triple counts must be non-negative exact ints')


def validate_corrected_crowd_selected_view_name(view_name: str) -> str:
    '''Validate a non-legacy name for an independently selected GT view.'''
    if type(view_name) is not str or not view_name:
        raise ValueError('selected corrected crowd view name must be non-empty')
    if view_name in CORRECTED_CROWD_VIEWS:
        raise ValueError('selected corrected crowd view name is reserved by legacy schema')
    return view_name


def validate_exact_consecutive_window_support(
    matched_count: int,
    triple_count: int,
    quadruple_count: int | None = None,
) -> None:
    '''Validate derivative-window support against matched occurrences.'''
    if triple_count > max(matched_count - 2, 0):
        raise ValueError('triple count exceeds matched exact-window support')
    if quadruple_count is None:
        return
    if quadruple_count > max(matched_count - 3, 0):
        raise ValueError('quadruple count exceeds matched exact-window support')
    if quadruple_count > 0 and quadruple_count >= triple_count:
        raise ValueError('quadruple count must be smaller than triple count')


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_Selected_View_Sequence_Summary:
    '''Sufficient statistics for one explicitly selected GT population.'''

    schema_version: int
    scene_id: str
    view_name: str
    selected_gt_count: int
    matched_selected_count: int
    metric_sample_sums: Float_Array
    metric_sample_counts: Int_Array
    accel_exact_consecutive_triple_count: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION:
            raise ValueError('unsupported selected corrected crowd schema version')
        if type(self.scene_id) is not str or not self.scene_id:
            raise ValueError('scene_id must be non-empty')
        validate_corrected_crowd_selected_view_name(self.view_name)
        for name in ('selected_gt_count', 'matched_selected_count'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('%s must be a non-negative exact int' % name)
        if self.matched_selected_count > self.selected_gt_count:
            raise ValueError('matched_selected_count cannot exceed selected_gt_count')
        if (
            type(self.accel_exact_consecutive_triple_count) is not int
            or self.accel_exact_consecutive_triple_count < 0
        ):
            raise ValueError('triple count must be a non-negative exact int')
        validate_exact_consecutive_window_support(
            self.matched_selected_count,
            self.accel_exact_consecutive_triple_count,
        )
        sums = float_array(self.metric_sample_sums, 'metric_sample_sums')
        counts = int_array(self.metric_sample_counts, 'metric_sample_counts')
        shape = (len(CORRECTED_CROWD_METRICS),)
        require_shape(sums, shape, 'metric_sample_sums')
        require_shape(counts, shape, 'metric_sample_counts')
        if np.any(counts < 0) or np.any(sums < 0.0):
            raise ValueError('selected summary statistics must be non-negative')
        if np.any((counts == 0) & (sums != 0.0)):
            raise ValueError('empty metric populations must have zero sum')
        expected_joint_count = 24 * self.matched_selected_count
        if np.any(counts[:10] != expected_joint_count):
            raise ValueError('joint metric counts differ from matched support')
        if counts[13] != self.matched_selected_count:
            raise ValueError('OKS metric count differs from matched support')
        pair_counts = counts[10:13]
        if not np.all(pair_counts == pair_counts[0]):
            raise ValueError('pair metric counts must share one population')
        max_pair_count = (
            self.matched_selected_count * (self.matched_selected_count - 1) // 2
        )
        if pair_counts[0] > max_pair_count:
            raise ValueError('pair metric count exceeds matched support')
        expected_accel_count = 24 * self.accel_exact_consecutive_triple_count
        if counts[14] != expected_accel_count:
            raise ValueError('ACCEL metric count differs from triple support')
        object.__setattr__(self, 'metric_sample_sums', sums)
        object.__setattr__(self, 'metric_sample_counts', counts)


@dataclass(frozen=True, slots=True)
class Corrected_Crowd_Selected_View_Result:
    '''Reduced metric result for one explicitly selected GT population.'''

    schema_version: int
    view_name: str
    selected_gt_count: int
    matched_selected_count: int
    metric_values: tuple[float | None, ...]
    accel_exact_consecutive_triple_count: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError('schema_version must be an exact int')
        if self.schema_version != CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION:
            raise ValueError('unsupported selected corrected crowd schema version')
        validate_corrected_crowd_selected_view_name(self.view_name)
        for name in ('selected_gt_count', 'matched_selected_count'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError('%s must be a non-negative exact int' % name)
        if self.matched_selected_count > self.selected_gt_count:
            raise ValueError('matched_selected_count cannot exceed selected_gt_count')
        if len(self.metric_values) != len(CORRECTED_CROWD_METRICS):
            raise ValueError('metric_values has wrong metric count')
        for value in self.metric_values:
            if value is not None and not np.isfinite(value):
                raise ValueError('metric values must be finite or None')
        for value in self.metric_values[10:14]:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError('dimensionless metric values must be in [0,1]')
        if (
            type(self.accel_exact_consecutive_triple_count) is not int
            or self.accel_exact_consecutive_triple_count < 0
        ):
            raise ValueError('triple count must be a non-negative exact int')
        validate_exact_consecutive_window_support(
            self.matched_selected_count,
            self.accel_exact_consecutive_triple_count,
        )
        matched_empty = self.matched_selected_count == 0
        required_indices = tuple(range(10)) + (13,)
        if any(
            (self.metric_values[index] is None) != matched_empty
            for index in required_indices
        ):
            raise ValueError('joint/OKS availability differs from matched support')
        if self.matched_selected_count < 2 and any(
            self.metric_values[index] is not None
            for index in range(10, 13)
        ):
            raise ValueError('pair metric availability exceeds matched support')
        pair_available = tuple(
            self.metric_values[index] is not None
            for index in range(10, 13)
        )
        if len(set(pair_available)) != 1:
            raise ValueError('pair metric availability must share one population')
        if (
            (self.metric_values[14] is None)
            != (self.accel_exact_consecutive_triple_count == 0)
        ):
            raise ValueError('ACCEL availability differs from triple support')


def validate_corrected_crowd_sequence(
    sequence: Corrected_Crowd_Sequence,
) -> Corrected_Crowd_Sequence:
    '''Return a fresh immutable validated snapshot.'''
    values = {
        field.name: getattr(sequence, field.name)
        for field in sequence.__dataclass_fields__.values()
    }
    return Corrected_Crowd_Sequence(**values)


def corrected_crowd_summary_to_json(
    summary: Corrected_Crowd_Sequence_Summary,
) -> JSON_Object:
    '''Serialize one stable scene summary.'''
    return {
        'schema_version': summary.schema_version,
        'scene_id': summary.scene_id,
        'views': list(CORRECTED_CROWD_VIEWS),
        'metrics': list(CORRECTED_CROWD_METRICS),
        'metric_units': list(CORRECTED_CROWD_METRIC_UNITS),
        'tp': summary.tp,
        'fn': summary.fn,
        'fp': summary.fp,
        'metric_sample_sums': summary.metric_sample_sums.tolist(),
        'metric_sample_counts': summary.metric_sample_counts.tolist(),
        'accel_exact_consecutive_triple_count': (
            summary.accel_exact_consecutive_triple_count.tolist()
        ),
    }


def corrected_crowd_summary_from_json(
    value: Mapping[str, Any],
) -> Corrected_Crowd_Sequence_Summary:
    '''Parse one exact stable scene-summary JSON object.'''
    expected = {
        'schema_version', 'scene_id', 'views', 'metrics', 'metric_units',
        'tp', 'fn', 'fp', 'metric_sample_sums', 'metric_sample_counts',
        'accel_exact_consecutive_triple_count',
    }
    if set(value) != expected:
        raise ValueError('corrected crowd summary JSON fields do not match schema')
    if tuple(value['views']) != CORRECTED_CROWD_VIEWS:
        raise ValueError('corrected crowd view order does not match schema')
    if tuple(value['metrics']) != CORRECTED_CROWD_METRICS:
        raise ValueError('corrected crowd metric order does not match schema')
    if tuple(value['metric_units']) != CORRECTED_CROWD_METRIC_UNITS:
        raise ValueError('corrected crowd metric units do not match schema')
    return Corrected_Crowd_Sequence_Summary(
        schema_version=value['schema_version'],
        scene_id=value['scene_id'],
        tp=value['tp'],
        fn=value['fn'],
        fp=value['fp'],
        metric_sample_sums=np.asarray(value['metric_sample_sums'], dtype=np.float64),
        metric_sample_counts=json_int_array(
            value['metric_sample_counts'],
            'metric_sample_counts',
        ),
        accel_exact_consecutive_triple_count=json_int_array(
            value['accel_exact_consecutive_triple_count'],
            'accel_exact_consecutive_triple_count',
        ),
    )


def corrected_crowd_result_to_json(result: Corrected_Crowd_Result) -> JSON_Object:
    '''Serialize one stable final corrected result.'''
    return {
        'schema_version': result.schema_version,
        'views': list(CORRECTED_CROWD_VIEWS),
        'metrics': list(CORRECTED_CROWD_METRICS),
        'metric_units': list(CORRECTED_CROWD_METRIC_UNITS),
        'completeness': {
            'tp': result.tp,
            'fn': result.fn,
            'fp': result.fp,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
        },
        'metric_values': [list(row) for row in result.metric_values],
        'accel_exact_consecutive_triple_count': list(
            result.accel_exact_consecutive_triple_count,
        ),
    }


def corrected_crowd_selected_view_summary_to_json(
    summary: Corrected_Crowd_Selected_View_Sequence_Summary,
) -> JSON_Object:
    '''Serialize one selected-view scene summary with an independent schema.'''
    return {
        'schema_version': summary.schema_version,
        'scene_id': summary.scene_id,
        'view_name': summary.view_name,
        'metrics': list(CORRECTED_CROWD_METRICS),
        'metric_units': list(CORRECTED_CROWD_METRIC_UNITS),
        'selected_gt_count': summary.selected_gt_count,
        'matched_selected_count': summary.matched_selected_count,
        'metric_sample_sums': summary.metric_sample_sums.tolist(),
        'metric_sample_counts': summary.metric_sample_counts.tolist(),
        'accel_exact_consecutive_triple_count': (
            summary.accel_exact_consecutive_triple_count
        ),
    }


def corrected_crowd_selected_view_summary_from_json(
    value: Mapping[str, Any],
) -> Corrected_Crowd_Selected_View_Sequence_Summary:
    '''Parse one exact selected-view scene-summary JSON object.'''
    expected = {
        'schema_version', 'scene_id', 'view_name', 'metrics', 'metric_units',
        'selected_gt_count', 'matched_selected_count', 'metric_sample_sums',
        'metric_sample_counts', 'accel_exact_consecutive_triple_count',
    }
    if set(value) != expected:
        raise ValueError('selected corrected crowd summary fields do not match schema')
    if tuple(value['metrics']) != CORRECTED_CROWD_METRICS:
        raise ValueError('corrected crowd metric order does not match schema')
    if tuple(value['metric_units']) != CORRECTED_CROWD_METRIC_UNITS:
        raise ValueError('corrected crowd metric units do not match schema')
    return Corrected_Crowd_Selected_View_Sequence_Summary(
        schema_version=value['schema_version'],
        scene_id=value['scene_id'],
        view_name=value['view_name'],
        selected_gt_count=value['selected_gt_count'],
        matched_selected_count=value['matched_selected_count'],
        metric_sample_sums=np.asarray(value['metric_sample_sums'], dtype=np.float64),
        metric_sample_counts=json_int_array(
            value['metric_sample_counts'],
            'metric_sample_counts',
        ),
        accel_exact_consecutive_triple_count=value[
            'accel_exact_consecutive_triple_count'
        ],
    )


def corrected_crowd_selected_view_result_to_json(
    result: Corrected_Crowd_Selected_View_Result,
) -> JSON_Object:
    '''Serialize one selected-view reduced result.'''
    return {
        'schema_version': result.schema_version,
        'view_name': result.view_name,
        'metrics': list(CORRECTED_CROWD_METRICS),
        'metric_units': list(CORRECTED_CROWD_METRIC_UNITS),
        'selected_gt_count': result.selected_gt_count,
        'matched_selected_count': result.matched_selected_count,
        'metric_values': list(result.metric_values),
        'accel_exact_consecutive_triple_count': (
            result.accel_exact_consecutive_triple_count
        ),
    }


def corrected_crowd_selected_view_result_from_json(
    value: Mapping[str, Any],
) -> Corrected_Crowd_Selected_View_Result:
    '''Parse one exact selected-view reduced-result JSON object.'''
    expected = {
        'schema_version', 'view_name', 'metrics', 'metric_units',
        'selected_gt_count', 'matched_selected_count', 'metric_values',
        'accel_exact_consecutive_triple_count',
    }
    if set(value) != expected:
        raise ValueError('selected corrected crowd result fields do not match schema')
    if tuple(value['metrics']) != CORRECTED_CROWD_METRICS:
        raise ValueError('corrected crowd metric order does not match schema')
    if tuple(value['metric_units']) != CORRECTED_CROWD_METRIC_UNITS:
        raise ValueError('corrected crowd metric units do not match schema')
    raw_values = value['metric_values']
    if not isinstance(raw_values, list):
        raise TypeError('metric_values must be a JSON array')
    typed_values = cast(list[Any], raw_values)
    return Corrected_Crowd_Selected_View_Result(
        schema_version=value['schema_version'],
        view_name=value['view_name'],
        selected_gt_count=value['selected_gt_count'],
        matched_selected_count=value['matched_selected_count'],
        metric_values=tuple(
            None if item is None else float(item)
            for item in typed_values
        ),
        accel_exact_consecutive_triple_count=value[
            'accel_exact_consecutive_triple_count'
        ],
    )
