'''Method-owned camera and mesh contracts for LSV-HR frame visualization.'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from hjlib_camera import (
    Camera_Intrinsics,
    Camera_with_Pose,
    Extrinsics_World_to_Camera,
)
from hjlib_meshes.mesh import Mesh


def require_frame_identity(value: str, name: str) -> str:
    '''Require one non-empty exact string identity.'''
    if type(value) is not str:
        raise TypeError('%s must be an exact str' % name)
    if not value:
        raise ValueError('%s must be non-empty' % name)
    return value


@dataclass(frozen=True, slots=True)
class LSVHR_Method_Camera:
    '''One method-declared pinhole camera used by projection and rendering.'''

    intrinsics: Camera_Intrinsics
    extrinsics: Extrinsics_World_to_Camera
    source_id: str

    def __post_init__(self) -> None:
        if type(self.intrinsics) is not Camera_Intrinsics:
            raise TypeError('intrinsics must be an exact Camera_Intrinsics')
        if type(self.extrinsics) is not Extrinsics_World_to_Camera:
            raise TypeError(
                'extrinsics must be an exact Extrinsics_World_to_Camera'
            )
        require_frame_identity(self.source_id, 'camera source_id')
        camera_K = self.intrinsics.K
        camera_RT = self.extrinsics.RT
        if camera_K.dtype != np.dtype(np.float64):
            raise TypeError('camera K must have dtype float64')
        if camera_RT.dtype != np.dtype(np.float64):
            raise TypeError('camera extrinsics must have dtype float64')
        if not bool(np.isfinite(camera_K).all()):
            raise ValueError('camera K must be finite')
        if self.intrinsics.fx <= 0.0 or self.intrinsics.fy <= 0.0:
            raise ValueError('camera focal lengths must be positive')
        determinant = float(np.linalg.det(camera_K))
        if not np.isfinite(determinant) or determinant <= 1e-12:
            raise ValueError('camera K must have positive nonsingular determinant')
        if not bool(np.isfinite(camera_RT).all()):
            raise ValueError('camera extrinsics must be finite')

    @property
    def image_size(self) -> tuple[int, int]:
        '''Return native `(width, height)`.'''
        return self.intrinsics.image_size

    @property
    def camera(self) -> Camera_with_Pose:
        '''Return the camera-owner composite used for projection.'''
        return Camera_with_Pose(self.intrinsics, self.extrinsics)


def immutable_float64_array(
        value: NDArray[np.generic],
        name: str,
    ) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True, order='C')
    if not bool(np.isfinite(array).all()):
        raise ValueError('%s must be finite' % name)
    array.setflags(write=False)
    return array


def immutable_int64_array(
        value: NDArray[np.generic],
        name: str,
    ) -> NDArray[np.int64]:
    source = np.asarray(value)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError('%s must have integer dtype' % name)
    array = np.array(source, dtype=np.int64, copy=True, order='C')
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LSVHR_Renderable_Person:
    '''One method prediction interpreted as a world-space triangle mesh.'''

    native_track_id: int
    mesh_world_m: Mesh

    def __post_init__(self) -> None:
        if type(self.native_track_id) is not int or self.native_track_id < 0:
            raise ValueError('native_track_id must be a non-negative exact int')
        if type(self.mesh_world_m) is not Mesh:
            raise TypeError('mesh_world_m must be an exact Mesh')
        vertices = immutable_float64_array(
            self.mesh_world_m.verts,
            'mesh_world_m.verts',
        )
        faces = immutable_int64_array(self.mesh_world_m.faces, 'mesh_world_m.faces')
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError('mesh_world_m verts must have non-empty shape (V, 3)')
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError('mesh_world_m faces must have non-empty shape (F, 3)')
        if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
            raise ValueError('mesh_world_m face index is outside verts')
        object.__setattr__(self, 'mesh_world_m', Mesh(vertices, faces))


@dataclass(frozen=True, slots=True)
class LSVHR_Renderable_Frame:
    '''One full method frame ready for a method-neutral mesh renderer.'''

    scene_id: str
    frame_id: int
    camera: LSVHR_Method_Camera
    people: tuple[LSVHR_Renderable_Person, ...]

    def __post_init__(self) -> None:
        require_frame_identity(self.scene_id, 'scene_id')
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError('frame_id must be a non-negative exact int')
        if type(self.camera) is not LSVHR_Method_Camera:
            raise TypeError('camera must be an exact LSVHR_Method_Camera')
        if type(self.people) is not tuple or not self.people:
            raise ValueError('people must be a non-empty tuple')
        if any(type(person) is not LSVHR_Renderable_Person for person in self.people):
            raise TypeError('people must contain exact LSVHR_Renderable_Person values')
        identities = tuple(person.native_track_id for person in self.people)
        if identities != tuple(sorted(set(identities))):
            raise ValueError('people must be ordered by unique native_track_id')


class LSVHR_Frame_Visualization_Provider(Protocol):
    '''Method-owned provider consumed by registered-output visualization.'''

    def load_renderable_frame(
            self,
            scene_id: str,
            frame_id: int,
        ) -> LSVHR_Renderable_Frame: ...


def project_lsvhr_world_points(
        camera: LSVHR_Method_Camera,
        points_world_m: NDArray[np.generic],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Project arbitrary leading axes through one method-owned camera.'''
    if type(camera) is not LSVHR_Method_Camera:
        raise TypeError('camera must be an exact LSVHR_Method_Camera')
    points = np.asarray(points_world_m)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError('points_world_m must have shape (..., 3)')
    if not bool(np.isfinite(points).all()):
        raise ValueError('points_world_m must be finite')
    leading = points.shape[:-1]
    pixels_flat, depth_flat = camera.camera.project_world_points(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    pixels = np.asarray(pixels_flat, dtype=np.float64).reshape(*leading, 2)
    depths = np.asarray(depth_flat, dtype=np.float64).reshape(leading)
    if not bool(np.isfinite(pixels).all()) \
            or not bool(np.isfinite(depths).all()):
        raise ValueError('method-camera projection produced non-finite values')
    return pixels, depths


__all__ = [
    'LSVHR_Frame_Visualization_Provider',
    'LSVHR_Method_Camera',
    'LSVHR_Renderable_Frame',
    'LSVHR_Renderable_Person',
    'project_lsvhr_world_points',
]
