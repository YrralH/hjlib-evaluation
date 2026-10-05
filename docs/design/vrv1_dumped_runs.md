# VRv1 whole-run 与 dump-backed GT

`VRv1_TestSet_Builder` 实现 `TestSet_Builder_Base`，按 assembly registry 的
canonical folder/FPS 路由，并用 assembly-owned scene roster 拒绝跨 camera family。
仅接受 `full` policy，不读 washed filter store。按 split scenes 与 dump runs 在同一循环
构造 seq-local `Filtered_Sub_Seq_Divider` 与 scene-level `Test_Segment`，保持 flat-index
和半开 frame bounds 一致；不删除任何 observation-invalid frame。

`Dumped_SMPL_GT_Provider(name_dataset, path_root_label, eval_meta)` 是从 WP provider
抽出的公共 full-label reader。它保留原有 SMPL joints/params 读取和 original-start
slice，不引入 raw reader 或重新拟合。`WP_GT_Provider` 是保留构造签名和 metric identity
的薄 wrapper。VRv1 两种 canonical 分别绑定 `VRV1_EVAL_META` 和
`VRV1_GOPRO_EVAL_META`：SMPL_24_full indices 0..23、pelvis 0、metres、shared K/RT、
`2026-05-20_v1`、无 OKS。

`get_testset_builder` 现在返回 base interface；WP/JTA 仍为原有 filter-backed generic
builder，VRv1 为 full-run builder。扩展其他 dump-backed SMPL datasets 时复用 GT reader，
metric identity 与 population admission 仍由该 dataset 明确声明。

Task 设计在
[existing-artifact absorption](../../../hjlib-dataset-assembly/docs/design/tasks/vrv1_existing_artifact_absorption/README.md)。
此处仅记录稳定 evaluation 边界；真实资产接纳、detection/cache 验证与 readiness receipt
由 assembly/experiments 拥有。合成 smoke 用真实 label codec 验证四 main 场景/5100 frames、
两个 FPS、whole-run bounds、negative view/policy admission、GT offset slicing 和 WP wrapper
行为；不运行 detector、ViT backbone 或模型 benchmark。
