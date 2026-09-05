# 我有 method camera 与 world meshes

当调用方已经把某个 LSV-HR method 的原生输出解释成 world-metre triangle meshes，
用本页 value contract 把同一个 method camera 交给 OKS 和 visualization：

```python
import numpy as np

from hjlib_camera import Camera_Intrinsics, Extrinsics_World_to_Camera
from hjlib_evaluation import (
    LSVHR_Method_Camera,
    LSVHR_Renderable_Frame,
    LSVHR_Renderable_Person,
    project_lsvhr_world_points,
)
from hjlib_meshes.mesh import Mesh

camera = LSVHR_Method_Camera(
    intrinsics=Camera_Intrinsics(K, (width, height)),
    extrinsics=Extrinsics_World_to_Camera(RT_world_to_camera),
    source_id='method/run/scene',
)
pixels, depths = project_lsvhr_world_points(camera, joints_world_m)

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
)
```

`image_size` 顺序是 `(width, height)`；vertices 和 projected points 都使用 method
camera 所声明的 world frame 与 metre unit。`people` 必须按唯一
`native_track_id` 升序。background pixels 不属于这些构造函数，也没有 camera
authority。

## Picking between options

| 你已有的数据 | 使用入口 |
| --- | --- |
| world points，需要与 render 共用 method camera | `project_lsvhr_world_points(...)` |
| 单人的 world mesh | `LSVHR_Renderable_Person(...)` |
| 一整帧有序 people + method camera | `LSVHR_Renderable_Frame(...)` |
| method-specific lazy loader | 实现 `LSVHR_Frame_Visualization_Provider` |
