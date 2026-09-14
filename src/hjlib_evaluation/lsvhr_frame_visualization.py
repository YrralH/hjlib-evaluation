'''Legacy LSV-HR renderable-frame transport over generic camera values.'''
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
    '''One full method frame using a generic per-frame camera value.'''

    scene_id: str
    frame_id: int
    camera: Camera_with_Pose
    people: tuple[LSVHR_Renderable_Person, ...]
    camera_source_id: str

    def __post_init__(self) -> None:
        require_frame_identity(self.scene_id, 'scene_id')
        if type(self.frame_id) is not int or self.frame_id < 0:
            raise ValueError('frame_id must be a non-negative exact int')
        if type(self.camera) is not Camera_with_Pose:
            raise TypeError('camera must be an exact Camera_with_Pose')
        if type(self.camera.intrinsics) is not Camera_Intrinsics:
            raise TypeError('camera intrinsics must be exact pinhole intrinsics')
        if type(self.camera.extrinsics) is not Extrinsics_World_to_Camera:
            raise TypeError('camera extrinsics must be exact world-to-camera')
        require_frame_identity(self.camera_source_id, 'camera source_id')
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


__all__ = [
    'LSVHR_Frame_Visualization_Provider',
    'LSVHR_Renderable_Frame',
    'LSVHR_Renderable_Person',
]
