# LSV-HR renderable-frame transport

## Scope

本单元只保留 legacy LSV-HR visualization consumer 所需的 mesh/frame transport。
LSV-HR scene-complete camera artifact 由 `hjlib-experiments-results` 的
`LSVHR_Loaded_Method_Camera` 拥有；相机数学与逐帧 value 由 `hjlib-camera` 拥有。
本仓不再定义 LSVHR-named camera wrapper 或 projection helper。

## Renderable values

- `LSVHR_Renderable_Person`：非负 native track ID，以及一个 canonical
  `hjlib-meshes.Mesh`；verts 是 finite `float64[V,3]` world-metre vertices，
  faces 是有效 `int64[F,3]` indices。
- `LSVHR_Renderable_Frame`：scene/frame、一个 exact generic
  `Camera_with_Pose`、单独的 non-empty `camera_source_id`，以及按唯一 native
  track ID 严格排序的非空 people tuple。
- `LSVHR_Frame_Visualization_Provider`：legacy structural protocol，只暴露
  `load_renderable_frame(scene_id, frame_id)`。

frame boundary 只要求 exact generic undistorted camera types 与 provenance；
numeric camera validation 由 method-result loader 在 artifact load 时完成一次，
不在逐帧 transport 重复。逐帧 projection 直接调用
`Camera_with_Pose.project_world_points(N,3)`；有额外 leading axes 的 caller 先
flatten，再恢复 shape。

mesh 数组在 construction 时复制并设为 read-only。body 参数解释、scene roster、
render/publication 和 camera artifact loading 均属于上层 owner。

## Migration boundary

generic camera 与 `camera_source_id` 分字段只是旧 provider 的 compatibility
transport，不是第二 camera truth。最终 standard loader path 必须从同一个
`LSVHR_Loaded_Method_Camera` 派生逐帧 generic camera 与 provenance，并由
binding-spec consumer 退休这个 legacy provider。

## Extension rule

不要在本仓新增 LSVHR-named camera value 或 projection helper。新的 method-result
camera semantic 应扩展 `hjlib-experiments-results` 的 standard loaded artifact；新的
generic projection 能力应扩展 `hjlib-camera`。只有 renderer 确实需要新的 method-neutral
mesh/frame transport 字段时，才扩展本单元，并同步更新 provider Protocol、usage signature
和 smoke test。
