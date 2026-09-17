from __future__ import annotations

import numpy as np
import pytest

from hjlib_camera import (
    Camera_Intrinsics,
    Camera_Intrinsics_With_Distortion,
    Camera_with_Pose,
    Extrinsics_World_to_Camera,
)
from hjlib_evaluation import (
    LSVHR_Renderable_Frame,
    LSVHR_Renderable_Person,
)
from hjlib_meshes.mesh import Mesh


def camera(
        K: np.ndarray | None = None,
        RT: np.ndarray | None = None,
    ) -> Camera_with_Pose:
    camera_K = np.array([
        [100.0, 0.0, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ]) if K is None else K
    camera_RT = np.eye(4) if RT is None else RT
    return Camera_with_Pose(
        Camera_Intrinsics(camera_K, (40, 30)),
        Extrinsics_World_to_Camera(camera_RT),
    )


def person(native_track_id: int = 1) -> LSVHR_Renderable_Person:
    vertices = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return LSVHR_Renderable_Person(
        native_track_id,
        Mesh(vertices, faces),
    )


def frame_with_camera(value: Camera_with_Pose) -> LSVHR_Renderable_Frame:
    return LSVHR_Renderable_Frame(
        'scene2',
        0,
        value,
        (person(),),
        'method/test-camera',
    )


def test_projects_with_method_intrinsics_and_extrinsics() -> None:
    RT = np.eye(4)
    RT[:3, 3] = np.array([1.0, -2.0, 3.0])
    value = camera(RT=RT)
    points = np.array([[[0.0, 2.0, 2.0], [1.0, 3.0, 7.0]]])
    leading = points.shape[:-1]
    pixels_flat, depths_flat = value.project_world_points(
        points.reshape(-1, 3)
    )
    pixels = pixels_flat.reshape(*leading, 2)
    depths = depths_flat.reshape(leading)
    expected_camera = points + RT[:3, 3]
    expected_pixels = expected_camera[..., :2] / expected_camera[..., 2:3]
    expected_pixels[..., 0] = expected_pixels[..., 0] * 100.0 + 20.0
    expected_pixels[..., 1] = expected_pixels[..., 1] * 120.0 + 15.0
    assert np.allclose(depths, expected_camera[..., 2])
    assert np.allclose(pixels, expected_pixels)


def test_camera_owner_rejects_skew() -> None:
    K = np.array([
        [100.0, 0.5, 20.0],
        [0.0, 120.0, 15.0],
        [0.0, 0.0, 1.0],
    ])
    with pytest.raises(AssertionError, match='skew'):
        camera(K=K)


def test_renderable_frame_rejects_distortion_and_empty_provenance() -> None:
    K = camera().intrinsics.K
    distorted = Camera_with_Pose(
        Camera_Intrinsics_With_Distortion(
            K,
            (40, 30),
            np.zeros(5, dtype=np.float64),
        ),
        Extrinsics_World_to_Camera(np.eye(4)),
    )
    with pytest.raises(TypeError, match='exact pinhole'):
        frame_with_camera(distorted)
    with pytest.raises(ValueError, match='non-empty'):
        LSVHR_Renderable_Frame('scene2', 0, camera(), (person(),), '')


def test_renderable_frame_requires_sorted_unique_people() -> None:
    person2 = person(2)
    person1 = person(1)
    with pytest.raises(ValueError, match='ordered'):
        LSVHR_Renderable_Frame(
            'scene2', 0, camera(), (person2, person1), 'method/test-camera')
    frame = LSVHR_Renderable_Frame(
        'scene2', 0, camera(), (person1, person2), 'method/test-camera')
    assert tuple(person.native_track_id for person in frame.people) == (1, 2)
    assert not frame.people[0].mesh_world_m.verts.flags.writeable


def test_renderable_frame_accepts_empty_people() -> None:
    frame = LSVHR_Renderable_Frame(
        'scene2', 0, camera(), (), 'method/test-camera')
    assert frame.people == ()


def smoke_test_lsvhr_frame_visualization() -> None:
    test_projects_with_method_intrinsics_and_extrinsics()
    test_camera_owner_rejects_skew()
    test_renderable_frame_rejects_distortion_and_empty_provenance()
    test_renderable_frame_requires_sorted_unique_people()
    test_renderable_frame_accepts_empty_people()
