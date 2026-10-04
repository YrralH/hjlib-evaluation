'''baseline001 as one call: its selection boundaries, the arguments every solve
receives, and that it recovers a synthetic plane.

The expected constants are written here as literals on purpose: comparing the
entry with a composition that imports the same constants would pass with any
value.
'''

import math
from types import SimpleNamespace

import numpy as np
import pytest

import hjlib_evaluation.ground_estimation_baseline001 as baseline_module
from hjlib_detection import Tracked_Person, Tracked_Scene, frame_indices_to_ranges
from hjlib_evaluation import estimate_ground_baseline001


K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])


def person(person_id: int, tops: np.ndarray, bottoms: np.ndarray,
           score: float, ankle_gap: float, bbox_width: float) -> Tracked_Person:
    '''One person, one frame per (top, bottom) pair.'''
    count = int(tops.shape[0])
    keypoints = np.zeros((count, 133, 3), dtype=np.float32)
    keypoints[:, :, 2] = score
    keypoints[:, 5, :2] = tops + [-5.0, 0.0]
    keypoints[:, 6, :2] = tops + [5.0, 0.0]
    keypoints[:, 15, :2] = bottoms + [-ankle_gap / 2.0, 0.0]
    keypoints[:, 16, :2] = bottoms + [ankle_gap / 2.0, 0.0]
    bboxes = np.zeros((count, 5), dtype=np.float32)
    bboxes[:, 0] = tops[:, 1] - 10.0
    bboxes[:, 1] = bottoms[:, 1] + 10.0
    bboxes[:, 2] = bottoms[:, 0] - bbox_width / 2.0
    bboxes[:, 3] = bottoms[:, 0] + bbox_width / 2.0
    bboxes[:, 4] = 1.0
    return Tracked_Person(
        person_id=person_id,
        list_ranges=frame_indices_to_ranges(list(range(count))),
        bboxes=bboxes,
        keypoints=keypoints,
        keypoints_mask=np.ones(count, dtype=np.bool_),
        num_frame_scene=count,
    )


def scene(persons: list[Tracked_Person], num_frame: int) -> Tracked_Scene:
    return Tracked_Scene(
        num_frame=num_frame,
        source_person_axis_size=len(persons),
        has_bboxes=True,
        keypoint_shape=(133, 3),
        persons=tuple(persons),
    )


def synthetic_ground(count: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''People 1.35 m tall standing on a plane 8 m below a camera pitched 30 deg.

    Returns the camera-frame plane (unit normal pointing from the ground to the
    camera, n.x + d = 0 with d = camera height), and the projected tops and
    bottoms.
    '''
    pitch = math.radians(30.0)
    up = np.array([0.0, -math.cos(pitch), -math.sin(pitch)])
    forward = np.array([0.0, math.sin(pitch), -math.cos(pitch)]) * -1.0
    lateral = np.array([1.0, 0.0, 0.0])
    height = 8.0
    rng = np.random.default_rng(0)
    along = rng.uniform(12.0, 40.0, count)
    across = rng.uniform(-6.0, 6.0, count)
    feet = -height * up + along[:, None] * forward + across[:, None] * lateral
    heads = feet + 1.35 * up

    def project(points: np.ndarray) -> np.ndarray:
        pixels = points @ K.T
        return pixels[:, :2] / pixels[:, 2:3]

    plane = np.append(up, height)
    return plane, project(heads), project(feet)


def test_selection_boundaries_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    tops = np.array([[900.0, 300.0], [1000.0, 320.0]])
    bottoms = np.array([[900.0, 600.0], [1000.0, 640.0]])
    captured: list[int] = []

    def solver_spy(top: np.ndarray, bottom: np.ndarray, K_: np.ndarray,
                   **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        del bottom, K_, kwargs
        captured.append(int(top.shape[0]))
        return np.array([0.0, -1.0, 0.0, 8.0]), np.array(0.0)

    def density_stub(bottom: np.ndarray, K_: np.ndarray, normal: np.ndarray,
                     **kwargs: float) -> SimpleNamespace:
        # Only which observations reach the solves is under test here.
        del K_, normal, kwargs
        return SimpleNamespace(
            normalized_observation_weights=np.ones(bottom.shape[0]))

    tracked = scene([
        person(0, tops, bottoms, score=5.0, ankle_gap=10.0, bbox_width=100.0),
        person(1, tops, bottoms, score=5.01, ankle_gap=10.0, bbox_width=100.0),
        person(2, tops, bottoms, score=6.0, ankle_gap=20.0, bbox_width=100.0),
        person(3, tops, bottoms, score=6.0, ankle_gap=19.0, bbox_width=100.0),
    ], num_frame=2)
    monkeypatch.setattr(baseline_module, 'solve_ground_param_by_top_bottom_given_K',
                        solver_spy)
    monkeypatch.setattr(baseline_module, 'compute_ground_observation_kde_density',
                        density_stub)
    estimate_ground_baseline001(tracked, K)
    # Score exactly 5.0 and ratio exactly 0.20 are out; 5.01 and 0.19 are in:
    # persons 1 and 3, two frames each, in both solves.
    assert captured == [4, 4]


def test_every_solve_receives_the_frozen_parameters() -> None:
    plane, tops, bottoms = synthetic_ground(60)
    tracked = scene([person(0, tops, bottoms, 6.0, 10.0, 100.0)], tops.shape[0])
    solver_calls: list[dict[str, object]] = []
    density_calls: list[dict[str, object]] = []
    real_solver = baseline_module.solve_ground_param_by_top_bottom_given_K
    real_density = baseline_module.compute_ground_observation_kde_density

    def solver_spy(top: np.ndarray, bottom: np.ndarray, K_: np.ndarray,
                   **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        solver_calls.append(dict(kwargs))
        return real_solver(top, bottom, K_, **kwargs)  # type: ignore[arg-type]

    def density_spy(bottom: np.ndarray, K_: np.ndarray, normal: np.ndarray,
                    **kwargs: float):  # type: ignore[no-untyped-def]
        density_calls.append(dict(kwargs))
        return real_density(bottom, K_, normal, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(baseline_module, 'solve_ground_param_by_top_bottom_given_K',
                        solver_spy)
    monkeypatch.setattr(baseline_module, 'compute_ground_observation_kde_density',
                        density_spy)
    try:
        estimate = estimate_ground_baseline001(tracked, K, distance_max_m=123.0)
    finally:
        monkeypatch.undo()
    assert len(solver_calls) == 2
    for call in solver_calls:
        assert call['H_prior'] == 1.35
        assert call['distance_min'] == -5.0
        assert call['distance_step'] == 0.1
        assert call['distance_max'] == 123.0
    assert solver_calls[0]['observation_weights'] is None
    weights = solver_calls[1]['observation_weights']
    assert isinstance(weights, np.ndarray)
    weights_array = np.asarray(weights, dtype=np.float64)
    assert np.array_equal(weights_array,
                          estimate.density.normalized_observation_weights)
    assert density_calls == [{'minimum_pre_normalization_weight': 0.25,
                              'maximum_pre_normalization_weight': 4.0}]
    assert estimate.distance_max_m == 123.0
    del plane


def test_default_search_range_is_the_evaluation_one() -> None:
    plane, tops, bottoms = synthetic_ground(60)
    del plane
    tracked = scene([person(0, tops, bottoms, 6.0, 10.0, 100.0)], tops.shape[0])
    assert estimate_ground_baseline001(tracked, K).distance_max_m == 80.0


def test_recovers_a_synthetic_plane() -> None:
    plane, tops, bottoms = synthetic_ground()
    tracked = scene([person(0, tops, bottoms, 6.0, 10.0, 100.0)], tops.shape[0])
    estimate = estimate_ground_baseline001(tracked, K)
    solved = np.asarray(estimate.weighted.plane_camera_abcd)
    sign = 1.0 if float(solved[:3] @ plane[:3]) > 0.0 else -1.0
    solved = sign * solved
    angle = math.degrees(math.acos(min(1.0, float(solved[:3] @ plane[:3]))))
    assert angle < 0.5
    assert abs(float(solved[3]) - float(plane[3])) < 0.15
    assert estimate.weighted.observations.count == tops.shape[0]


def test_rejects_a_bad_search_range() -> None:
    plane, tops, bottoms = synthetic_ground(10)
    del plane
    tracked = scene([person(0, tops, bottoms, 6.0, 10.0, 100.0)], tops.shape[0])
    with pytest.raises(ValueError):
        estimate_ground_baseline001(tracked, K, distance_max_m=0.0)
