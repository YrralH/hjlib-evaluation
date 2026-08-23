'''Typed inputs and results for unordered JTA person evaluation.'''
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any, Final, Mapping, cast
import json

import numpy as np
from numpy.typing import NDArray


JTA_PERSON_DETECTION_SCHEMA_VERSION: Final = 1
JTA_IMAGE_SIZE_WH: Final = (1920, 1080)
JTA_CAMERA_K: Final = np.asarray([
    [1158.0, 0.0, 960.0],
    [0.0, 1158.0, 540.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
JTA_ENDPOINT_INDICES: Final = np.asarray(
    [8, 4, 9, 5, 10, 6, 19, 16, 20, 17, 21, 18],
    dtype=np.int64,
)
JTA_ENDPOINT_NAMES: Final = (
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
)
SMPL54_ENDPOINT_INDICES: Final = np.asarray(
    [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8],
    dtype=np.int64,
)
JTA_ENDPOINT_OKS_SIGMAS: Final = np.asarray([
    0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
    0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
], dtype=np.float64)


def immutable_array(
        value: NDArray[np.generic],
        dtype: np.dtype[Any],
) -> NDArray[Any]:
    output = np.array(value, dtype=dtype, copy=True, order='C')
    output.flags.writeable = False
    return output


def validate_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(
            character not in '0123456789abcdef' for character in value):
        raise ValueError('%s must be a lowercase SHA-256' % name)
    return value


def frame_digest(
        domain: bytes,
        scene_id: str,
        frame_id: int,
        identities: tuple[str, ...],
        arrays: tuple[NDArray[np.generic], ...],
) -> str:
    digest = sha256(domain)
    digest.update(scene_id.encode('utf-8'))
    digest.update(b'\0%d\0' % frame_id)
    for identity in identities:
        digest.update(identity.encode('ascii'))
        digest.update(b'\0')
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode('ascii'))
        digest.update(json.dumps(contiguous.shape).encode('ascii'))
        digest.update(contiguous.tobytes(order='C'))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class JTA_Person_Detection_GT_Frame:
    '''Immutable evaluation-owned JTA GT for one frame.'''

    scene_id: str
    frame_id: int
    gt_source_ids: NDArray[np.int64]
    gt_xy: NDArray[np.float64]
    gt_visible: NDArray[np.bool_]
    gt_xyz_camera: NDArray[np.float64]
    gt_bbox_xyxy: NDArray[np.float64]
    camera_K: NDArray[np.float64]
    semantic_sha256: str = ''

    def __post_init__(self) -> None:
        if not self.scene_id or type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError('GT frame identity is invalid')
        ids = immutable_array(self.gt_source_ids, np.dtype(np.int64))
        xy = immutable_array(self.gt_xy, np.dtype(np.float64))
        visible = immutable_array(self.gt_visible, np.dtype(np.bool_))
        xyz = immutable_array(self.gt_xyz_camera, np.dtype(np.float64))
        bbox = immutable_array(self.gt_bbox_xyxy, np.dtype(np.float64))
        camera = immutable_array(self.camera_K, np.dtype(np.float64))
        count = ids.shape[0] if ids.ndim == 1 else -1
        if ids.shape != (count,) or np.any(ids < 0) \
                or len(np.unique(ids)) != count \
                or (count > 1 and np.any(ids[1:] <= ids[:-1])):
            raise ValueError('GT source IDs must be unique ascending nonnegative int64')
        if xy.shape != (count, 12, 2) \
                or visible.shape != (count, 12) \
                or xyz.shape != (count, 12, 3) \
                or bbox.shape != (count, 4) \
                or camera.shape != (3, 3):
            raise ValueError('GT frame arrays have invalid shapes')
        if not np.isfinite(xy).all() or not np.isfinite(xyz).all() \
                or not np.isfinite(bbox).all() or not np.array_equal(
                    camera, JTA_CAMERA_K,
                ):
            raise ValueError('GT frame arrays are non-finite or use the wrong K')
        if count and np.any(
                (bbox[:, 2] <= bbox[:, 0]) | (bbox[:, 3] <= bbox[:, 1])):
            raise ValueError('GT bbox area must be positive')
        digest = frame_digest(
            b'hjlib_evaluation.jta_person_detection_gt_frame.v1\0',
            self.scene_id,
            self.frame_id,
            (),
            (ids, xy, visible, xyz, bbox, camera),
        )
        if self.semantic_sha256 and self.semantic_sha256 != digest:
            raise ValueError('GT frame semantic digest mismatch')
        object.__setattr__(self, 'gt_source_ids', ids)
        object.__setattr__(self, 'gt_xy', xy)
        object.__setattr__(self, 'gt_visible', visible)
        object.__setattr__(self, 'gt_xyz_camera', xyz)
        object.__setattr__(self, 'gt_bbox_xyxy', bbox)
        object.__setattr__(self, 'camera_K', camera)
        object.__setattr__(self, 'semantic_sha256', digest)


@dataclass(frozen=True, slots=True)
class JTA_Person_Detection_Prediction_Frame:
    '''Immutable method-neutral 3D predictions for one JTA frame.'''

    scene_id: str
    frame_id: int
    prediction_source_sha256: str
    prediction_profile_sha256: str
    prediction_row_ids: NDArray[np.int64]
    pred_xyz_camera: NDArray[np.float64]
    semantic_sha256: str = ''

    def __post_init__(self) -> None:
        if not self.scene_id or type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError('prediction frame identity is invalid')
        source = validate_sha256(
            self.prediction_source_sha256, 'prediction_source_sha256',
        )
        profile = validate_sha256(
            self.prediction_profile_sha256, 'prediction_profile_sha256',
        )
        ids = immutable_array(self.prediction_row_ids, np.dtype(np.int64))
        xyz = immutable_array(self.pred_xyz_camera, np.dtype(np.float64))
        count = ids.shape[0] if ids.ndim == 1 else -1
        if ids.shape != (count,) or np.any(ids < 0) \
                or len(np.unique(ids)) != count:
            raise ValueError('prediction row IDs must be unique nonnegative int64')
        if xyz.shape != (count, 12, 3) or not np.isfinite(xyz).all():
            raise ValueError('prediction joints must be finite [P,12,3]')
        digest = frame_digest(
            b'hjlib_evaluation.jta_person_detection_prediction_frame.v1\0',
            self.scene_id,
            self.frame_id,
            (source, profile),
            (ids, xyz),
        )
        if self.semantic_sha256 and self.semantic_sha256 != digest:
            raise ValueError('prediction frame semantic digest mismatch')
        object.__setattr__(self, 'prediction_row_ids', ids)
        object.__setattr__(self, 'pred_xyz_camera', xyz)
        object.__setattr__(self, 'semantic_sha256', digest)


def normalize_flag_array(
        value: NDArray[np.generic],
        name: str,
        shape: tuple[int, int],
) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.shape != shape or not (
            np.issubdtype(array.dtype, np.bool_)
            or np.issubdtype(array.dtype, np.integer)):
        raise ValueError('%s must be a boolean/integer array of exact shape' % name)
    if np.any((array != 0) & (array != 1)):
        raise ValueError('%s values must be zero or one' % name)
    return np.asarray(array, dtype=np.bool_)


def make_jta_person_detection_gt_frame(
        scene_id: str,
        frame_id: int,
        source_person_ids: NDArray[np.generic],
        joints_2d_22: NDArray[np.generic],
        joints_3d_camera_22: NDArray[np.generic],
        occluded_22: NDArray[np.generic],
        self_occluded_22: NDArray[np.generic],
) -> JTA_Person_Detection_GT_Frame:
    '''Construct the exact raw-JTA population and twelve-endpoint GT view.'''
    ids_raw = np.asarray(source_person_ids)
    xy_raw = np.asarray(joints_2d_22)
    xyz_raw = np.asarray(joints_3d_camera_22)
    if ids_raw.ndim != 1 or not np.issubdtype(ids_raw.dtype, np.integer):
        raise ValueError('source_person_ids must be an integer vector')
    count = ids_raw.shape[0]
    if xy_raw.shape != (count, 22, 2) or xyz_raw.shape != (count, 22, 3) \
            or not np.issubdtype(xy_raw.dtype, np.number) \
            or not np.issubdtype(xyz_raw.dtype, np.number):
        raise ValueError('raw JTA joint arrays have invalid shapes or dtypes')
    ids = np.asarray(ids_raw, dtype=np.int64)
    if np.any(ids < 0) or len(np.unique(ids)) != count:
        raise ValueError('source person IDs must be unique and nonnegative')
    xy = np.asarray(xy_raw, dtype=np.float64)
    xyz = np.asarray(xyz_raw, dtype=np.float64)
    occluded = normalize_flag_array(occluded_22, 'occluded_22', (count, 22))
    self_occluded = normalize_flag_array(
        self_occluded_22, 'self_occluded_22', (count, 22),
    )
    finite = np.isfinite(xy).all(axis=(1, 2)) & np.isfinite(xyz).all(axis=(1, 2))
    safe_xy = np.where(np.isfinite(xy), xy, 0.0)
    minimum = np.min(safe_xy, axis=1) if count else np.empty((0, 2))
    maximum = np.max(safe_xy, axis=1) if count else np.empty((0, 2))
    bbox = np.concatenate((minimum, maximum), axis=1)
    positive = (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
    intersects = (
        (bbox[:, 2] > 0.0) & (bbox[:, 3] > 0.0)
        & (bbox[:, 0] < JTA_IMAGE_SIZE_WH[0])
        & (bbox[:, 1] < JTA_IMAGE_SIZE_WH[1])
    )
    admitted = finite & positive & intersects
    selected = np.flatnonzero(admitted)
    if len(selected):
        selected = selected[np.argsort(ids[selected], kind='stable')]
    endpoint_xy = xy[selected][:, JTA_ENDPOINT_INDICES]
    endpoint_xyz = xyz[selected][:, JTA_ENDPOINT_INDICES]
    endpoint_occ = occluded[selected][:, JTA_ENDPOINT_INDICES]
    endpoint_self_occ = self_occluded[selected][:, JTA_ENDPOINT_INDICES]
    on_screen = (
        (endpoint_xy[:, :, 0] >= 0.0)
        & (endpoint_xy[:, :, 0] < JTA_IMAGE_SIZE_WH[0])
        & (endpoint_xy[:, :, 1] >= 0.0)
        & (endpoint_xy[:, :, 1] < JTA_IMAGE_SIZE_WH[1])
    )
    return JTA_Person_Detection_GT_Frame(
        scene_id=scene_id,
        frame_id=frame_id,
        gt_source_ids=ids[selected],
        gt_xy=endpoint_xy,
        gt_visible=(~endpoint_occ) & (~endpoint_self_occ) & on_screen,
        gt_xyz_camera=endpoint_xyz,
        gt_bbox_xyxy=bbox[selected],
        camera_K=JTA_CAMERA_K,
    )


@dataclass(frozen=True, slots=True)
class JTA_Person_Detection_Result:
    '''Immutable sufficient-stat result for one complete prediction source.'''

    expected_frame_keys: tuple[tuple[str, int], ...]
    prediction_source_sha256: str
    prediction_profile_sha256: str
    frame_count: int
    gt_person_count: int
    prediction_person_count: int
    matched_person_count: int
    unmatched_gt_count: int
    unmatched_prediction_count: int
    projection_invalid_prediction_count: int
    pa_degenerate_person_count: int
    matched_oks_sum: float
    absolute_mpjpe_person_sum_mm: float
    pelvis_mpjpe_person_sum_mm: float
    pa_mpjpe_person_sum_mm: float
    pa_valid_person_count: int
    input_digest_sha256: str

    @property
    def matched_mean_oks(self) -> float | None:
        return (
            self.matched_oks_sum / self.matched_person_count
            if self.matched_person_count else None
        )

    @property
    def all_gt_mean_oks(self) -> float | None:
        return (
            self.matched_oks_sum / self.gt_person_count
            if self.gt_person_count else None
        )

    @property
    def recall(self) -> float | None:
        return (
            self.matched_person_count / self.gt_person_count
            if self.gt_person_count else None
        )

    @property
    def absolute_mpjpe_mm(self) -> float | None:
        return (
            self.absolute_mpjpe_person_sum_mm / self.matched_person_count
            if self.matched_person_count else None
        )

    @property
    def pelvis_mpjpe_mm(self) -> float | None:
        return (
            self.pelvis_mpjpe_person_sum_mm / self.matched_person_count
            if self.matched_person_count else None
        )

    @property
    def pa_mpjpe_mm(self) -> float | None:
        return (
            self.pa_mpjpe_person_sum_mm / self.pa_valid_person_count
            if self.pa_valid_person_count else None
        )


def result_payload(result: JTA_Person_Detection_Result) -> dict[str, object]:
    return {
        'schema': 'hjlib_evaluation.jta_person_detection_result',
        'version': JTA_PERSON_DETECTION_SCHEMA_VERSION,
        **{
            field.name: (
                [list(key) for key in result.expected_frame_keys]
                if field.name == 'expected_frame_keys'
                else getattr(result, field.name)
            )
            for field in fields(result)
        },
        'metrics': {
            'matched_mean_oks': result.matched_mean_oks,
            'all_gt_mean_oks': result.all_gt_mean_oks,
            'recall': result.recall,
            'absolute_mpjpe_mm': result.absolute_mpjpe_mm,
            'pelvis_mpjpe_mm': result.pelvis_mpjpe_mm,
            'pa_mpjpe_mm': result.pa_mpjpe_mm,
        },
        'denominators': {
            'matched_person': result.matched_person_count,
            'matched_joint': result.matched_person_count * 12,
            'all_gt_person': result.gt_person_count,
            'pa_person': result.pa_valid_person_count,
            'pa_joint': result.pa_valid_person_count * 12,
        },
    }


def jta_person_detection_result_to_json(
        result: JTA_Person_Detection_Result,
) -> bytes:
    payload = result_payload(result)
    digest = sha256(
        b'hjlib_evaluation.jta_person_detection_result_json.v1\0'
        + json.dumps(
            payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode('utf-8'),
    ).hexdigest()
    payload['semantic_sha256'] = digest
    return json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')


def jta_person_detection_result_from_json(
        value: bytes | str,
) -> JTA_Person_Detection_Result:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError('JTA person-detection result JSON must be an object')
    payload = cast(dict[str, object], decoded)
    claimed = payload.pop('semantic_sha256', None)
    expected = sha256(
        b'hjlib_evaluation.jta_person_detection_result_json.v1\0'
        + json.dumps(
            payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode('utf-8'),
    ).hexdigest()
    if claimed != expected \
            or payload.pop('schema', None) \
            != 'hjlib_evaluation.jta_person_detection_result' \
            or payload.pop('version', None) != JTA_PERSON_DETECTION_SCHEMA_VERSION:
        raise ValueError('JTA person-detection result identity is invalid')
    payload.pop('metrics', None)
    payload.pop('denominators', None)
    keys_raw = payload.get('expected_frame_keys')
    if not isinstance(keys_raw, list):
        raise ValueError('expected frame keys are invalid')
    keys: list[tuple[str, int]] = []
    for key in cast(list[object], keys_raw):
        if not isinstance(key, list):
            raise ValueError('expected frame key item is invalid')
        key_values = cast(list[object], key)
        if len(key_values) != 2 or type(key_values[0]) is not str \
                or type(key_values[1]) is not int:
            raise ValueError('expected frame key item is invalid')
        keys.append((key_values[0], key_values[1]))
    payload['expected_frame_keys'] = tuple(keys)
    field_names = {field.name for field in fields(JTA_Person_Detection_Result)}
    if set(payload) != field_names:
        raise ValueError('JTA person-detection result fields are invalid')
    return JTA_Person_Detection_Result(**cast(Mapping[str, Any], payload))


__all__ = [
    'JTA_CAMERA_K', 'JTA_ENDPOINT_INDICES', 'JTA_ENDPOINT_NAMES',
    'JTA_ENDPOINT_OKS_SIGMAS', 'JTA_IMAGE_SIZE_WH',
    'JTA_PERSON_DETECTION_SCHEMA_VERSION', 'SMPL54_ENDPOINT_INDICES',
    'JTA_Person_Detection_GT_Frame', 'JTA_Person_Detection_Prediction_Frame',
    'JTA_Person_Detection_Result', 'jta_person_detection_result_from_json',
    'jta_person_detection_result_to_json',
    'make_jta_person_detection_gt_frame',
]
