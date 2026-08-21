'''Person-frame sampling and same-ray evaluation for ground estimation.'''

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from hjlib_detection import Tracked_Scene
from hjlib_geometry import intersect_rays_with_planes
from hjlib_ground_solver import solve_ground_param_by_top_bottom_given_K


type Ground_Estimator = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[np.ndarray, np.ndarray],
]


def owned_readonly_array(
        value: object,
        name: str,
        dtype: np.dtype[np.generic],
        shape_tail: tuple[int, ...],
    ) -> NDArray[np.generic]:
    '''Validate one array and return an owned read-only copy.'''
    if not isinstance(value, np.ndarray):
        raise TypeError('%s must be a numpy array' % name)
    array = cast(NDArray[np.generic], value)
    if array.dtype != dtype:
        raise TypeError('%s must have dtype %s' % (name, dtype))
    if array.ndim != len(shape_tail) + 1 or array.shape[1:] != shape_tail:
        raise ValueError('%s has invalid shape %r' % (name, array.shape))
    if np.issubdtype(array.dtype, np.floating) and not bool(np.isfinite(array).all()):
        raise ValueError('%s must be finite' % name)
    output = array.copy(order='C')
    output.setflags(write=False)
    return output


def validate_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError('%s must be a Python integer' % name)
    output = value
    if output < 0:
        raise ValueError('%s must be nonnegative' % name)
    return output


def validated_camera_K(value: object) -> NDArray[np.float64]:
    '''Return one owned finite nonsingular float64 camera matrix.'''
    if not isinstance(value, np.ndarray):
        raise TypeError('K must be a numpy array')
    array = cast(NDArray[np.generic], value)
    K: NDArray[np.float64] = array.astype(np.float64, copy=True)
    if K.shape != (3, 3) or not bool(np.isfinite(K).all()):
        raise ValueError('K must be finite with shape (3,3)')
    determinant = float(np.linalg.det(K))
    if not np.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise ValueError('K must be nonsingular')
    return K


@dataclass(frozen=True, slots=True)
class Ground_Observation_Set:
    '''Canonical person-frame top/bottom observations for one scene.'''

    frame_indices: NDArray[np.int64]
    person_ids: NDArray[np.int64]
    top_xy_px: NDArray[np.float64]
    bottom_xy_px: NDArray[np.float64]
    quality: NDArray[np.float64]
    bottom_pair_bbox_width_ratio: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        frames = cast(NDArray[np.int64], owned_readonly_array(
            self.frame_indices,
            'frame_indices',
            np.dtype(np.int64),
            (),
        ))
        persons = cast(NDArray[np.int64], owned_readonly_array(
            self.person_ids,
            'person_ids',
            np.dtype(np.int64),
            (),
        ))
        top = cast(NDArray[np.float64], owned_readonly_array(
            self.top_xy_px,
            'top_xy_px',
            np.dtype(np.float64),
            (2,),
        ))
        bottom = cast(NDArray[np.float64], owned_readonly_array(
            self.bottom_xy_px,
            'bottom_xy_px',
            np.dtype(np.float64),
            (2,),
        ))
        quality = cast(NDArray[np.float64], owned_readonly_array(
            self.quality,
            'quality',
            np.dtype(np.float64),
            (),
        ))
        ratio = (
            None
            if self.bottom_pair_bbox_width_ratio is None
            else cast(NDArray[np.float64], owned_readonly_array(
                self.bottom_pair_bbox_width_ratio,
                'bottom_pair_bbox_width_ratio',
                np.dtype(np.float64),
                (),
            ))
        )
        count = frames.shape[0]
        if any(array.shape[0] != count for array in (persons, top, bottom, quality)):
            raise ValueError('ground-observation arrays must share first-axis count')
        if ratio is not None and ratio.shape[0] != count:
            raise ValueError('ground-observation ratio must share first-axis count')
        if bool(np.any(frames < 0)) or bool(np.any(persons < 0)):
            raise ValueError('ground-observation identities must be nonnegative')
        if count > 0:
            order = np.lexsort((persons, frames))
            if not np.array_equal(order, np.arange(count, dtype=np.int64)):
                raise ValueError('ground observations must use canonical frame/person order')
            pairs = np.stack([frames, persons], axis=1)
            if bool(np.any(np.all(pairs[1:] == pairs[:-1], axis=1))):
                raise ValueError('ground-observation identities must be unique')
            lengths = np.linalg.norm(top - bottom, axis=1)
            if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 0.0)):
                raise ValueError('ground observations must be nondegenerate')
            if ratio is not None and bool(np.any(ratio < 0.0)):
                raise ValueError('bottom-pair bbox-width ratios must be nonnegative')
        object.__setattr__(self, 'frame_indices', frames)
        object.__setattr__(self, 'person_ids', persons)
        object.__setattr__(self, 'top_xy_px', top)
        object.__setattr__(self, 'bottom_xy_px', bottom)
        object.__setattr__(self, 'quality', quality)
        object.__setattr__(self, 'bottom_pair_bbox_width_ratio', ratio)

    @property
    def count(self) -> int:
        return int(self.frame_indices.size)


@dataclass(frozen=True, slots=True)
class Ground_Effect_Support:
    '''Ordered GT ray support shared by methods for one scene.'''

    frame_ids: NDArray[np.int64]
    gt_track_ids: NDArray[np.int64]
    image_xy_px: NDArray[np.float64]
    gt_intersections_camera_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        frames = cast(NDArray[np.int64], owned_readonly_array(
            self.frame_ids,
            'frame_ids',
            np.dtype(np.int64),
            (),
        ))
        tracks = cast(NDArray[np.int64], owned_readonly_array(
            self.gt_track_ids,
            'gt_track_ids',
            np.dtype(np.int64),
            (),
        ))
        pixels = cast(NDArray[np.float64], owned_readonly_array(
            self.image_xy_px,
            'image_xy_px',
            np.dtype(np.float64),
            (2,),
        ))
        intersections = cast(NDArray[np.float64], owned_readonly_array(
            self.gt_intersections_camera_m,
            'gt_intersections_camera_m',
            np.dtype(np.float64),
            (3,),
        ))
        count = frames.shape[0]
        if count == 0:
            raise ValueError('ground-effect support cannot be empty')
        if any(array.shape[0] != count for array in (tracks, pixels, intersections)):
            raise ValueError('ground-effect support arrays must share first-axis count')
        if bool(np.any(frames < 0)) or bool(np.any(tracks < 0)):
            raise ValueError('ground-effect support identities must be nonnegative')
        order = np.lexsort((tracks, frames))
        if not np.array_equal(order, np.arange(count, dtype=np.int64)):
            raise ValueError('ground-effect support must use canonical frame/track order')
        pairs = np.stack([frames, tracks], axis=1)
        if bool(np.any(np.all(pairs[1:] == pairs[:-1], axis=1))):
            raise ValueError('ground-effect support identities must be unique')
        object.__setattr__(self, 'frame_ids', frames)
        object.__setattr__(self, 'gt_track_ids', tracks)
        object.__setattr__(self, 'image_xy_px', pixels)
        object.__setattr__(
            self,
            'gt_intersections_camera_m',
            intersections,
        )

    @property
    def count(self) -> int:
        return int(self.frame_ids.size)


@dataclass(frozen=True, slots=True)
class Ground_Estimation_Result:
    '''One selected observation set and its solved camera-frame plane.'''

    observations: Ground_Observation_Set
    plane_camera_abcd: NDArray[np.float64]
    objective: float

    def __post_init__(self) -> None:
        plane_dynamic = np.asarray(self.plane_camera_abcd)
        if plane_dynamic.dtype != np.dtype(np.float64):
            raise TypeError('plane_camera_abcd must have dtype float64')
        if plane_dynamic.shape != (4,) or not bool(np.isfinite(plane_dynamic).all()):
            raise ValueError('plane_camera_abcd must be finite with shape (4,)')
        normal_norm = float(np.linalg.norm(plane_dynamic[:3]))
        if not np.isfinite(normal_norm) or not np.isclose(
                normal_norm,
                1.0,
                rtol=0.0,
                atol=1e-5,
            ):
            raise ValueError('plane_camera_abcd must have a unit normal')
        if not isinstance(self.objective, float):
            raise TypeError('objective must be float')
        if not np.isfinite(self.objective) or self.objective < 0.0:
            raise ValueError('objective must be finite and nonnegative')
        plane = plane_dynamic.copy()
        plane.setflags(write=False)
        object.__setattr__(
            self,
            'plane_camera_abcd',
            plane,
        )


@dataclass(frozen=True, slots=True)
class Ground_Plane_Diagnostics:
    '''Normalized sign-aligned predicted/GT plane diagnostics.'''

    normalized_pred_plane_camera_abcd: NDArray[np.float64]
    normalized_gt_plane_camera_abcd: NDArray[np.float64]
    normal_angle_deg: float
    distance_ratio: float

    def __post_init__(self) -> None:
        planes: list[NDArray[np.float64]] = []
        for value, name in (
                (
                    self.normalized_pred_plane_camera_abcd,
                    'normalized_pred_plane_camera_abcd',
                ),
                (
                    self.normalized_gt_plane_camera_abcd,
                    'normalized_gt_plane_camera_abcd',
                ),
            ):
            dynamic = np.asarray(value)
            if dynamic.dtype != np.dtype(np.float64):
                raise TypeError('%s must have dtype float64' % name)
            if dynamic.shape != (4,) or not bool(np.isfinite(dynamic).all()):
                raise ValueError('%s must be finite with shape (4,)' % name)
            plane = dynamic.copy()
            if not np.isclose(
                    np.linalg.norm(plane[:3]),
                    1.0,
                    rtol=0.0,
                    atol=1e-12,
                ):
                raise ValueError('%s must have a unit normal' % name)
            plane.setflags(write=False)
            planes.append(plane)
        if float(np.dot(planes[0][:3], planes[1][:3])) < -1e-12:
            raise ValueError('diagnostic plane normals must be sign-aligned')
        if type(self.normal_angle_deg) is not float:
            raise TypeError('normal_angle_deg must be a Python float')
        if (
                not np.isfinite(self.normal_angle_deg)
                or not 0.0 <= self.normal_angle_deg <= 90.0 + 1e-10
            ):
            raise ValueError('normal_angle_deg is invalid')
        if type(self.distance_ratio) is not float:
            raise TypeError('distance_ratio must be a Python float')
        if not np.isfinite(self.distance_ratio) or self.distance_ratio < 0.0:
            raise ValueError('distance_ratio must be finite and nonnegative')
        object.__setattr__(self, 'normalized_pred_plane_camera_abcd', planes[0])
        object.__setattr__(self, 'normalized_gt_plane_camera_abcd', planes[1])


@dataclass(frozen=True, slots=True)
class Ground_Effect_Decomposition:
    '''Fixed-normal oracle-distance and distance-only ground-effect errors.'''

    oracle_distance_m: float
    normal_oracle_error_m: NDArray[np.float64]
    distance_only_error_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        if type(self.oracle_distance_m) is not float:
            raise TypeError('oracle_distance_m must be a Python float')
        if not np.isfinite(self.oracle_distance_m):
            raise ValueError('oracle_distance_m must be finite')
        normal_errors = cast(NDArray[np.float64], owned_readonly_array(
            self.normal_oracle_error_m,
            'normal_oracle_error_m',
            np.dtype(np.float64),
            (),
        ))
        distance_errors = cast(NDArray[np.float64], owned_readonly_array(
            self.distance_only_error_m,
            'distance_only_error_m',
            np.dtype(np.float64),
            (),
        ))
        if normal_errors.size == 0 or distance_errors.shape != normal_errors.shape:
            raise ValueError('decomposition error arrays must be nonempty and aligned')
        if bool(np.any(normal_errors < 0.0)) or bool(np.any(distance_errors < 0.0)):
            raise ValueError('decomposition errors must be nonnegative')
        object.__setattr__(self, 'normal_oracle_error_m', normal_errors)
        object.__setattr__(self, 'distance_only_error_m', distance_errors)


def collect_ground_observations(
        tracked_scene: Tracked_Scene,
        top_joint_pair: tuple[int, int],
        bottom_joint_pair: tuple[int, int],
        confidence_threshold: float,
        maximum_bottom_pair_bbox_width_ratio: float | None = None,
    ) -> Ground_Observation_Set:
    '''Collect high-confidence present person-frames in canonical order.'''
    if tracked_scene.keypoint_shape is None:
        raise ValueError('tracked_scene must contain keypoints')
    if tracked_scene.keypoint_shape[1] < 3:
        raise ValueError('tracked_scene keypoints must contain x/y/score')
    joint_values = top_joint_pair + bottom_joint_pair
    if (
            len(top_joint_pair) != 2
            or len(bottom_joint_pair) != 2
            or any(type(index) is not int or index < 0 for index in joint_values)
        ):
        raise ValueError('joint pairs must contain nonnegative Python integers')
    if max(joint_values) >= tracked_scene.keypoint_shape[0]:
        raise ValueError('joint pair exceeds tracked_scene keypoint count')
    if not isinstance(confidence_threshold, float) or not np.isfinite(confidence_threshold):
        raise TypeError('confidence_threshold must be a finite float')
    if maximum_bottom_pair_bbox_width_ratio is not None:
        if (
                type(maximum_bottom_pair_bbox_width_ratio) is not float
                or not np.isfinite(maximum_bottom_pair_bbox_width_ratio)
                or maximum_bottom_pair_bbox_width_ratio <= 0.0
            ):
            raise ValueError('maximum bottom-pair bbox-width ratio must be positive')

    frames_parts: list[NDArray[np.int64]] = []
    persons_parts: list[NDArray[np.int64]] = []
    top_parts: list[NDArray[np.float64]] = []
    bottom_parts: list[NDArray[np.float64]] = []
    quality_parts: list[NDArray[np.float64]] = []
    ratio_parts: list[NDArray[np.float64]] = []
    selected_joints = np.asarray(joint_values, dtype=np.int64)
    for person in tracked_scene.persons:
        if person.keypoints is None:
            raise ValueError('tracked_scene person is missing keypoints')
        if maximum_bottom_pair_bbox_width_ratio is not None and person.bboxes is None:
            raise ValueError('bbox modality is required for bottom-pair ratio filtering')
        present = (
            np.ones(person.num_observation, dtype=np.bool_)
            if person.keypoints_mask is None
            else person.keypoints_mask
        )
        keypoints = person.keypoints[present].astype(np.float64, copy=False)
        frames = person.frame_indices[present]
        if keypoints.shape[0] == 0:
            continue
        quality = np.min(keypoints[:, selected_joints, 2], axis=1)
        top = np.mean(keypoints[:, top_joint_pair, :2], axis=1)
        bottom = np.mean(keypoints[:, bottom_joint_pair, :2], axis=1)
        lengths = np.linalg.norm(top - bottom, axis=1)
        if (
                not bool(np.isfinite(top).all())
                or not bool(np.isfinite(bottom).all())
                or not bool(np.isfinite(quality).all())
                or bool(np.any(lengths <= 0.0))
            ):
            raise ValueError('present ground observation is invalid')
        selected = quality > confidence_threshold
        ratio: NDArray[np.float64] | None = None
        if maximum_bottom_pair_bbox_width_ratio is not None:
            assert person.bboxes is not None
            bboxes = person.bboxes[present].astype(np.float64)
            widths = bboxes[:, 3] - bboxes[:, 2]
            bottom_separation = np.linalg.norm(
                keypoints[:, bottom_joint_pair[0], :2]
                - keypoints[:, bottom_joint_pair[1], :2],
                axis=1,
            )
            if (
                    not bool(np.isfinite(widths).all())
                    or not bool(np.isfinite(bottom_separation).all())
                    or bool(np.any(widths <= 0.0))
                ):
                raise ValueError('present bbox-ratio observation is invalid')
            ratio = bottom_separation / widths
            selected = selected & (ratio < maximum_bottom_pair_bbox_width_ratio)
        if not bool(np.any(selected)):
            continue
        frames = frames[selected]
        quality = quality[selected]
        top = top[selected]
        bottom = bottom[selected]
        if ratio is not None:
            ratio = ratio[selected]
        frames_parts.append(cast(NDArray[np.int64], frames.copy()))
        persons_parts.append(np.full(frames.size, person.person_id, dtype=np.int64))
        top_parts.append(top)
        bottom_parts.append(bottom)
        quality_parts.append(quality)
        if ratio is not None:
            ratio_parts.append(ratio)

    if not frames_parts:
        return Ground_Observation_Set(
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            (
                np.empty((0,), dtype=np.float64)
                if maximum_bottom_pair_bbox_width_ratio is not None
                else None
            ),
        )
    frames_all = np.concatenate(frames_parts)
    persons_all = np.concatenate(persons_parts)
    top_all = np.concatenate(top_parts)
    bottom_all = np.concatenate(bottom_parts)
    quality_all = np.concatenate(quality_parts)
    ratio_all = (
        np.concatenate(ratio_parts)
        if maximum_bottom_pair_bbox_width_ratio is not None
        else None
    )
    order = np.lexsort((persons_all, frames_all))
    return Ground_Observation_Set(
        frames_all[order],
        persons_all[order],
        top_all[order],
        bottom_all[order],
        quality_all[order],
        None if ratio_all is None else ratio_all[order],
    )


def take_ground_observations(
        observations: Ground_Observation_Set,
        indices: NDArray[np.int64],
    ) -> Ground_Observation_Set:
    '''Take one ordered unique row subset.'''
    if indices.dtype != np.dtype(np.int64) or indices.ndim != 1:
        raise TypeError('indices must be int64 with shape (N,)')
    if indices.size > 0:
        if bool(np.any(indices < 0)) or bool(np.any(indices >= observations.count)):
            raise ValueError('ground-observation index is out of range')
        if bool(np.any(indices[1:] <= indices[:-1])):
            raise ValueError('ground-observation indices must be strictly increasing')
    return Ground_Observation_Set(
        observations.frame_indices[indices],
        observations.person_ids[indices],
        observations.top_xy_px[indices],
        observations.bottom_xy_px[indices],
        observations.quality[indices],
        (
            None
            if observations.bottom_pair_bbox_width_ratio is None
            else observations.bottom_pair_bbox_width_ratio[indices]
        ),
    )


def select_ground_observations_at_frame(
        observations: Ground_Observation_Set,
        frame_index: int,
    ) -> Ground_Observation_Set:
    '''Select every observation at one global frame.'''
    frame = validate_nonnegative_int(frame_index, 'frame_index')
    indices = np.flatnonzero(observations.frame_indices == frame).astype(np.int64)
    return take_ground_observations(observations, indices)


def sample_ground_observations(
        observations: Ground_Observation_Set,
        max_count: int,
        seed: int,
    ) -> Ground_Observation_Set:
    '''Uniformly sample person-frames without replacement, then restore order.'''
    count_max = validate_nonnegative_int(max_count, 'max_count')
    seed_value = validate_nonnegative_int(seed, 'seed')
    if observations.count <= count_max:
        indices = np.arange(observations.count, dtype=np.int64)
    else:
        generator = np.random.default_rng(seed_value)
        indices = generator.choice(
            observations.count,
            size=count_max,
            replace=False,
        ).astype(np.int64)
        indices.sort()
    return take_ground_observations(observations, indices)


def estimate_ground_from_observations(
        observations: Ground_Observation_Set,
        K: NDArray[np.generic],
        estimator: Ground_Estimator = solve_ground_param_by_top_bottom_given_K,
    ) -> Ground_Estimation_Result:
    '''Run one injected top/bottom estimator on a selected observation set.'''
    if observations.count < 3:
        raise ValueError('ground estimation requires at least three observations')
    camera_K = validated_camera_K(K)
    plane_dynamic, objective_dynamic = estimator(
        observations.top_xy_px,
        observations.bottom_xy_px,
        camera_K,
    )
    plane = np.asarray(plane_dynamic, dtype=np.float64)
    objective_array = np.asarray(objective_dynamic)
    if objective_array.shape != ():
        raise ValueError('ground estimator objective must be scalar')
    objective = float(objective_array)
    return Ground_Estimation_Result(observations, plane, objective)


def validate_ground_effect_support_against_K(
        support: Ground_Effect_Support,
        K: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Validate stored GT intersections against their pixels under current K.'''
    camera_K = validated_camera_K(K)
    points = support.gt_intersections_camera_m
    if bool(np.any(points[:, 2] <= 0.0)):
        raise ValueError('GT ground intersections must have positive camera depth')
    homogeneous = points @ camera_K.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.allclose(
            pixels,
            support.image_xy_px,
            rtol=0.0,
            atol=1e-8,
        ):
        raise ValueError('ground-effect support does not match current K')
    return camera_K


def compute_same_ray_ground_errors(
        support: Ground_Effect_Support,
        K: NDArray[np.generic],
        plane_camera_abcd: NDArray[np.generic],
    ) -> NDArray[np.float64]:
    '''Intersect support rays with one plane and return 3D errors in metres.'''
    camera_K = validate_ground_effect_support_against_K(support, K)
    plane = np.asarray(plane_camera_abcd, dtype=np.float64)
    if plane.shape != (4,) or not bool(np.isfinite(plane).all()):
        raise ValueError('plane_camera_abcd must be finite with shape (4,)')
    homogeneous_pixels = np.concatenate(
        [support.image_xy_px, np.ones((support.count, 1), dtype=np.float64)],
        axis=1,
    )
    directions = np.linalg.solve(camera_K, homogeneous_pixels.T).T
    origins = np.zeros((support.count, 3), dtype=np.float64)
    planes = np.broadcast_to(plane, (support.count, 4)).copy()
    intersections, unused_distances = intersect_rays_with_planes(
        origins,
        directions,
        planes,
        min_abs_cosine=1e-10,
    )
    del unused_distances
    errors = np.linalg.norm(
        intersections - support.gt_intersections_camera_m,
        axis=1,
    )
    if not bool(np.isfinite(errors).all()):
        raise ValueError('same-ray ground errors must be finite')
    output = errors.astype(np.float64, copy=True)
    output.setflags(write=False)
    return output


def normalized_plane_coefficients(
        value: NDArray[np.generic],
        name: str,
    ) -> NDArray[np.float64]:
    plane = np.asarray(value, dtype=np.float64)
    if plane.shape != (4,) or not bool(np.isfinite(plane).all()):
        raise ValueError('%s must be finite with shape (4,)' % name)
    normal_scale = float(np.max(np.abs(plane[:3])))
    if not np.isfinite(normal_scale) or normal_scale <= 0.0:
        raise ValueError('%s normal must be nonzero' % name)
    scaled = plane / normal_scale
    normal_norm = float(np.linalg.norm(scaled[:3]))
    if not np.isfinite(normal_norm) or normal_norm <= 0.0:
        raise ValueError('%s normal has invalid norm' % name)
    output = scaled / normal_norm
    if not bool(np.isfinite(output).all()):
        raise ValueError('%s normalization produced nonfinite coefficients' % name)
    return output


def compute_ground_plane_diagnostics(
        pred_plane_camera_abcd: NDArray[np.generic],
        gt_plane_camera_abcd: NDArray[np.generic],
    ) -> Ground_Plane_Diagnostics:
    '''Normalize, sign-align, and compare one predicted/GT plane pair.'''
    predicted = normalized_plane_coefficients(
        pred_plane_camera_abcd,
        'pred_plane_camera_abcd',
    )
    ground_truth = normalized_plane_coefficients(
        gt_plane_camera_abcd,
        'gt_plane_camera_abcd',
    )
    if abs(float(ground_truth[3])) <= 1e-12:
        raise ValueError('GT plane distance must be nonzero')
    if float(np.dot(predicted[:3], ground_truth[:3])) < 0.0:
        predicted = -predicted
    cosine = float(np.clip(
        np.dot(predicted[:3], ground_truth[:3]),
        -1.0,
        1.0,
    ))
    angle = float(np.rad2deg(np.arccos(cosine)))
    ratio = abs(float(predicted[3])) / abs(float(ground_truth[3]))
    return Ground_Plane_Diagnostics(
        predicted,
        ground_truth,
        angle,
        ratio,
    )


def lower_weighted_median(
        values: NDArray[np.generic],
        positive_weights: NDArray[np.generic],
    ) -> float:
    '''Return the canonical lower weighted median of finite float64 rows.'''
    values_dynamic = np.asarray(values)
    weights_dynamic = np.asarray(positive_weights)
    if values_dynamic.dtype != np.dtype(np.float64) or values_dynamic.ndim != 1:
        raise TypeError('values must be float64 with shape (N,)')
    if weights_dynamic.dtype != np.dtype(np.float64) or weights_dynamic.ndim != 1:
        raise TypeError('positive_weights must be float64 with shape (N,)')
    values_float64 = cast(NDArray[np.float64], values_dynamic)
    weights_float64 = cast(NDArray[np.float64], weights_dynamic)
    if values_float64.size == 0 or weights_float64.shape != values_float64.shape:
        raise ValueError('weighted-median arrays must be nonempty and aligned')
    if (
            not bool(np.isfinite(values_float64).all())
            or not bool(np.isfinite(weights_float64).all())
            or bool(np.any(weights_float64 <= 0.0))
        ):
        raise ValueError('weighted-median values/weights must be finite and weights positive')
    canonical_position = np.arange(values_float64.size, dtype=np.int64)
    order = np.lexsort((canonical_position, values_float64))
    ordered_values = values_float64[order]
    ordered_weights = weights_float64[order]
    threshold = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side='left'))
    return float(ordered_values[index])


def compute_ground_effect_decomposition(
        support: Ground_Effect_Support,
        K: NDArray[np.generic],
        pred_plane_camera_abcd: NDArray[np.generic],
        gt_plane_camera_abcd: NDArray[np.generic],
    ) -> Ground_Effect_Decomposition:
    '''Compute fixed-normal oracle-distance and distance-only errors.'''
    camera_K = validate_ground_effect_support_against_K(support, K)
    diagnostics = compute_ground_plane_diagnostics(
        pred_plane_camera_abcd,
        gt_plane_camera_abcd,
    )
    predicted = diagnostics.normalized_pred_plane_camera_abcd
    ground_truth = diagnostics.normalized_gt_plane_camera_abcd
    residual = support.gt_intersections_camera_m @ ground_truth[:3] + ground_truth[3]
    if not bool(np.isfinite(residual).all()) or bool(np.any(np.abs(residual) > 1e-8)):
        raise ValueError('GT intersections do not lie on the GT plane')
    homogeneous_pixels = np.column_stack([
        support.image_xy_px,
        np.ones(support.count, dtype=np.float64),
    ])
    rays = np.linalg.solve(camera_K, homogeneous_pixels.T).T
    ray_norms = np.linalg.norm(rays, axis=1)
    denominators = rays @ predicted[:3]
    normalized_cosines = np.abs(denominators) / ray_norms
    if (
            not bool(np.isfinite(ray_norms).all())
            or bool(np.any(ray_norms <= 0.0))
            or not bool(np.isfinite(normalized_cosines).all())
            or bool(np.any(normalized_cosines <= 1e-10))
        ):
        raise ValueError('support ray is parallel or too close to predicted normal')
    distance_values = -(support.gt_intersections_camera_m @ predicted[:3])
    geometric_coefficients = ray_norms / np.abs(denominators)
    if (
            not bool(np.isfinite(distance_values).all())
            or not bool(np.isfinite(geometric_coefficients).all())
            or bool(np.any(geometric_coefficients <= 0.0))
        ):
        raise ValueError('oracle-distance values/coefficients are invalid')
    oracle_distance = lower_weighted_median(
        distance_values,
        geometric_coefficients,
    )
    oracle_plane = np.concatenate([
        predicted[:3],
        np.array([oracle_distance], dtype=np.float64),
    ])
    distance_only_plane = np.concatenate([
        ground_truth[:3],
        np.array([predicted[3]], dtype=np.float64),
    ])
    normal_errors = compute_same_ray_ground_errors(support, camera_K, oracle_plane)
    distance_errors = compute_same_ray_ground_errors(
        support,
        camera_K,
        distance_only_plane,
    )
    return Ground_Effect_Decomposition(
        oracle_distance,
        normal_errors,
        distance_errors,
    )


def summarize_ground_errors(
        error_m: NDArray[np.generic],
    ) -> dict[str, int | float]:
    '''Compute person-frame-micro descriptive statistics in metres.'''
    errors_dynamic = np.asarray(error_m)
    if errors_dynamic.dtype != np.dtype(np.float64) or errors_dynamic.ndim != 1:
        raise TypeError('error_m must be float64 with shape (N,)')
    errors = cast(NDArray[np.float64], errors_dynamic)
    if errors.size == 0 or not bool(np.isfinite(errors).all()):
        raise ValueError('error_m must be finite and nonempty')
    quantiles = np.quantile(errors, [0.5, 0.9, 0.95, 0.99], method='linear')
    return {
        'count': int(errors.size),
        'mean_m': float(np.mean(errors)),
        'std_m': float(np.std(errors, ddof=0)),
        'median_m': float(quantiles[0]),
        'p90_m': float(quantiles[1]),
        'p95_m': float(quantiles[2]),
        'p99_m': float(quantiles[3]),
        'min_m': float(np.min(errors)),
        'max_m': float(np.max(errors)),
    }


__all__ = [
    'Ground_Effect_Decomposition',
    'Ground_Effect_Support',
    'Ground_Estimation_Result',
    'Ground_Estimator',
    'Ground_Observation_Set',
    'Ground_Plane_Diagnostics',
    'collect_ground_observations',
    'compute_ground_effect_decomposition',
    'compute_ground_plane_diagnostics',
    'compute_same_ray_ground_errors',
    'estimate_ground_from_observations',
    'sample_ground_observations',
    'select_ground_observations_at_frame',
    'lower_weighted_median',
    'summarize_ground_errors',
    'take_ground_observations',
    'validate_ground_effect_support_against_K',
]
