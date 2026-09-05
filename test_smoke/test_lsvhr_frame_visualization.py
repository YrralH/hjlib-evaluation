from __future__ import annotations

import numpy as np
import pytest

from hjlib_camera import Camera_Intrinsics, Extrinsics_World_to_Camera
from hjlib_evaluation import (
    LSVHR_Method_Camera,
    LSVHR_Renderable_Frame,
    LSVHR_Renderable_Person,
    project_lsvhr_world_points,
)
from hjlib_meshes.mesh import Mesh


def camera(
        K: np.ndarray | None = None,
        RT: np.ndarray | None = None,
    ) -> LSVHR_Method_Camera:
    camera_K = np.array([
        [100.0, 0.0, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ]) if K is None else K
    camera_RT = np.eye(4) if RT is None else RT
    return LSVHR_Method_Camera(
        Camera_Intrinsics(camera_K, (40, 30)),
        Extrinsics_World_to_Camera(camera_RT),
        'method/test-camera',
    )


def test_projects_with_method_intrinsics_and_extrinsics() -> None:
    RT = np.eye(4)
    RT[:3, 3] = np.array([1.0, -2.0, 3.0])
    value = camera(RT=RT)
    points = np.array([[[0.0, 2.0, 2.0], [1.0, 3.0, 7.0]]])
    pixels, depths = project_lsvhr_world_points(value, points)
    expected_camera = points + RT[:3, 3]
    expected_pixels = expected_camera[..., :2] / expected_camera[..., 2:3]
    expected_pixels[..., 0] = expected_pixels[..., 0] * 100.0 + 20.0
    expected_pixels[..., 1] = expected_pixels[..., 1] * 120.0 + 15.0
    assert np.allclose(depths, expected_camera[..., 2])
    assert np.allclose(pixels, expected_pixels)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_intrinsics(value: float) -> None:
    K = np.array([
        [100.0, 0.0, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ])
    K[0, 0] = value
    with pytest.raises(ValueError, match='finite'):
        camera(K=K)


def test_rejects_nonpositive_focal_length() -> None:
    K = np.array([
        [-100.0, 0.0, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(ValueError, match='focal'):
        camera(K=K)


@pytest.mark.parametrize('dtype', [np.float32, np.int64])
def test_rejects_non_float64_camera_values(
        dtype: type[np.generic],
    ) -> None:
    K = np.array([
        [100.0, 0.0, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ], dtype=dtype)
    with pytest.raises(TypeError, match='K.*float64'):
        camera(K=K)
    normalized_camera = camera(RT=np.eye(4, dtype=dtype))
    assert normalized_camera.extrinsics.RT.dtype == np.float64


def test_camera_owner_rejects_skew() -> None:
    K = np.array([
        [100.0, 0.5, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(AssertionError, match='skew'):
        camera(K=K)


def test_renderable_frame_requires_sorted_unique_people() -> None:
    vertices = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    person2 = LSVHR_Renderable_Person(2, Mesh(vertices, faces))
    person1 = LSVHR_Renderable_Person(1, Mesh(vertices, faces))
    with pytest.raises(ValueError, match='ordered'):
        LSVHR_Renderable_Frame('scene2', 0, camera(), (person2, person1))
    frame = LSVHR_Renderable_Frame('scene2', 0, camera(), (person1, person2))
    assert tuple(person.native_track_id for person in frame.people) == (1, 2)
    assert not frame.people[0].mesh_world_m.verts.flags.writeable


def smoke_test_lsvhr_frame_visualization() -> None:
    test_projects_with_method_intrinsics_and_extrinsics()
    for value in (np.nan, np.inf, -np.inf):
        test_rejects_nonfinite_intrinsics(value)
    test_rejects_nonpositive_focal_length()
    for dtype in (np.float32, np.int64):
        test_rejects_non_float64_camera_values(dtype)
    test_camera_owner_rejects_skew()
    test_renderable_frame_requires_sorted_unique_people()
