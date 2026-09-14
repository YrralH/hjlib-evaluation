# 我有逐帧 generic camera 与 world meshes

旧 visualization adapter 已经持有一个 `hjlib-camera.Camera_with_Pose` 时，可把它
与 provenance 和 mesh transport 交给 legacy renderer：

```python
import numpy as np

from hjlib_camera import (
    Camera_Intrinsics,
    Camera_with_Pose,
    Extrinsics_World_to_Camera,
)
from hjlib_evaluation import (
    LSVHR_Renderable_Frame,
    LSVHR_Renderable_Person,
)
from hjlib_meshes.mesh import Mesh

camera = Camera_with_Pose(
    Camera_Intrinsics(K, (width, height)),
    Extrinsics_World_to_Camera(RT_world_to_camera),
)
leading = joints_world_m.shape[:-1]
pixels_flat, depths_flat = camera.project_world_points(
    np.asarray(joints_world_m, dtype=np.float64).reshape(-1, 3)
)
pixels = pixels_flat.reshape(*leading, 2)
depths = depths_flat.reshape(leading)

person = LSVHR_Renderable_Person(
    native_track_id=track_id,
    mesh_world_m=Mesh(
        verts=np.asarray(vertices_world_m, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    ),
)
frame = LSVHR_Renderable_Frame(
    scene_id=scene_id,
    frame_id=frame_id,
    camera=camera,
    people=(person,),
    camera_source_id='method/run/scene',
)
```

`image_size` 顺序是 `(width, height)`；vertices 和 projected points 使用同一个
world frame 与 metre unit。`people` 按唯一 `native_track_id` 升序。

新的 standard-result caller 不应自行拼这个 transport。它从
`hjlib-experiments-results.LSVHR_Loaded_Method_Camera.camera_at(frame_id)` 取得
generic camera，并沿用 loaded camera 的 `source_id`。

## Signatures

```python
LSVHR_Renderable_Person(
    native_track_id: int,
    mesh_world_m: Mesh,
)

LSVHR_Renderable_Frame(
    scene_id: str,
    frame_id: int,
    camera: Camera_with_Pose,
    people: tuple[LSVHR_Renderable_Person, ...],
    camera_source_id: str,
)

LSVHR_Frame_Visualization_Provider.load_renderable_frame(
    scene_id: str,
    frame_id: int,
) -> LSVHR_Renderable_Frame
```

## Picking between options

| You have | Use |
|---|---|
| A standard scene result | `result.camera.camera_at(frame_id)` and the loaded camera's `source_id` |
| An existing legacy frame provider | `LSVHR_Renderable_Frame` as the temporary renderer transport |
| Only world points to project | `Camera_with_Pose.project_world_points(...)` from `hjlib-camera` |
