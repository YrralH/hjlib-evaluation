# 已有 VRv1 fitted-SMPL dumps

默认选择 `vrv1_smpl_fitted`，只包含 S21_0、S22_0、S23_0、S24_0，30 FPS。
需要 GoPro 时显式选择 `vrv1_gopro_smpl_fitted`，50 FPS；两者的 label folder
同名且分开。Bare `vrv1` 报 `ValueError`，提示 canonical name。

```python
from hjlib_evaluation import get_gt_provider, get_testset_builder

testset = get_testset_builder(
    'vrv1_smpl_fitted', path_dump_root=path_dump_root,
).build(policy='full', split='all')
gt = get_gt_provider('vrv1_smpl_fitted', path_dump_root=path_dump_root)
segment = testset.get_test_segment(0)
joints = gt.get_smpl_joints_54_world(
    segment.name_scene, segment.name_seq,
    (segment.index_frame_original_start, segment.index_frame_original_end),
)
```

此 builder 不需要 `path_filter_stats_base`，只接受 `full` policy。每个 dump run
保留全部同步 scene frames，split 读取已吸收 folder 内的 train/val/test 文本；`all`
为这些 split 的 union，没有 split 内容时枚举该 folder 的全部 scenes。
混入另一 camera family 或未知 scene 会失败。GT 从 full label 的
`smpl_joints_54_world` 和 `smpl_param_world` 取值，frame range 是同步 scene timeline
的半开区间，provider 按 dump 的 original start 做 slice。

| 选择 | FPS | GT/metric |
|---|---|---|
| `vrv1_smpl_fitted` | 30 | full dumped fit；SMPL_24_full，pelvis 0，m |
| `vrv1_gopro_smpl_fitted` | 50 | 同一 preserved metric；独立 camera population |

两者 metric version 为 `2026-05-20_v1`，无 2D OKS。TestSet/GT 可构造不代表
feature cache 或模型输入已经 ready；cache transform 和 detector observations 的
接纳由 assembly 与 experiments absorption receipt 验证。
