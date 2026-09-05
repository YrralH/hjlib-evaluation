# LSV-HR method-camera 与 renderable-frame 契约

## Scope

本单元定义 LSV-HR evaluation/visualization 共用的 method-neutral value
contract。它只拥有 camera、`hjlib-meshes.Mesh` 的 LSV-HR 组合约束和 provider
形状，不读取 dataset、method artifact、SMPL、registry 或 renderer。

## Camera authority

`LSVHR_Method_Camera` 由 `hjlib-camera` 的 `Camera_Intrinsics` 与
`Extrinsics_World_to_Camera` 组成，并增加非空 `source_id`、finite、正焦距和
non-singular K 约束。投影必须调用 `project_lsvhr_world_points(...)`；该函数继续委托
`Camera_with_Pose.project_world_points(...)`，不维护第二套相机数学。

每个 method provider 必须返回自己认为的 camera。dataset camera、GT camera 和
background reader 都不能替换它。同一个 `LSVHR_Method_Camera` value 同时供 OKS
projection 与 mesh renderer 使用，避免指标和图像各自选择不同相机。

## Renderable values

- `LSVHR_Renderable_Person`：非负 native track ID，以及一个 canonical
  `hjlib-meshes.Mesh`；其 verts 必须是 finite `float64[V,3]` world-metre
  vertices，faces 必须是有效 `int64[F,3]` indices。
- `LSVHR_Renderable_Frame`：scene/frame、一个 method camera，以及按唯一 native
  track ID 严格排序的非空 people tuple。
- `LSVHR_Frame_Visualization_Provider`：只暴露
  `load_renderable_frame(scene_id, frame_id)` 的 structural protocol。

mesh 数组在 value construction 时复制并设为 read-only，避免 renderer 改写 method
loader 的预测。通用 mesh container 与 trimesh bridge 仍由 `hjlib-meshes` 拥有；
body 参数如何解释、scene roster 如何选择、如何渲染与发布，均由上层 method adapter
或 exp-res composition 拥有。

## Extension rule

新 method 只需提供该 provider；不要在本仓增加 method-specific loader。若需要新的
camera model，先扩展 `hjlib-camera`，再组合进新的明确契约，不能在 evaluation 中写
ad-hoc projection。
