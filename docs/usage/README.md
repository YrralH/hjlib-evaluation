# 用法 —— hjlib-evaluation

调用方视角:我有一个数据集 + 一批预测(已存 dump,或一个待评的 ckpt),怎么算出
世界空间 MPJPE / T-MPJPE。

## 一句话索引

评测分两段、由 per-segment 预测 dump 衔接(`{'segment': Test_Segment, 'pred':
{'joints_54_world': (L,54,3)}}`):**推理**(逐 segment 跑网络 → 落 dump)与**归约**
(读 dump + GT → 指标表)。两段解耦,所以**评已有 dump 不需要 live 网络**。
默认归约字段是 `pred['joints_54_world']`,即 monolith 等价的 dump-side tamed
协议;如需诊断 no-invalid-tame raw 输出,显式传 `pred_joints_key='joints_54_world_raw'`。

需要在 legacy corrected population 上做显式子集比较时，见
[selected corrected-crowd population](corrected_crowd_selected_population.md)。

VirtualCrowd 的默认与 Crowd4D-native profile 选择见
[VirtualCrowd evaluation profiles](virtualcrowd_evaluation_profiles.md)。未显式
指定其他 profile 时，“在 VC 上测试”指 `VC_HJ_DEFAULT_V1`。

真实数据测试与 dump 归约脚本使用 tracked contract 配置本机 roots：

```bash
cp test/local_setting_test.py.example test/local_setting_test.py
```

复制后只编辑本机 runtime。不要把另一台机器的完整 runtime 复制过来，也不要把
真实路径或秘密写入 tracked example。`test_smoke/` 不需要这份 runtime。

## 决策树:我该调哪个

```
我已经有 per-segment 预测 dump(含 monolith 真 dump)?
└─ 直接归约 -> Tester(...).stage_eval(dump_dir)  或  eval_dumps_against_gt(...)

我要评一个新 ckpt(还没 dump)?
└─ 先推理再归约 -> Tester(..., network_driver=<driver>).stage_inference(dump_dir)
                  然后 .stage_eval(dump_dir)
   注:live network_driver 当前 deferred(见 design;评新 ckpt 才需要)。

我只想看测试集构成 / 接线 sanity?
└─ testset.summary()  或  Tester(...).stage_list_segments()

我已经有每帧 scalar residual + 最终 bool mask?
└─ summarize_trajectory_residuals(...)，再用 reduce_trajectory_residual_summaries(...)
   做 trajectory-weighted macro 与 valid-frame-weighted micro 归约。
```

## 端到端示例(评已有 dump —— 迁移 / parity 主路径)

```python
from hjlib_evaluation import (
    get_testset_builder, get_gt_provider, build_test_assembly, Tester,
)

# 1. 测试集:经 washed filter store 建(复用 assembly 的 Filtered_Sub_Seq_Divider)
testset = get_testset_builder(
    'worldpose_smpl',                                    # canonical 后缀名(registry 已废弃 bare 名)
    path_dump_root='/.../__As_Single_Bbox_hjlib__',     # dumped <dataset>_<leaf> 根的父目录
    path_filter_stats_base='/.../Data_Process',          # <token>_filter_stats/ 的父目录
).build(policy='full', split='test')

# 2. GT provider:WP 从 assembly dump full label;jta/jta_ext 需 path_raw_data_root=
gt = get_gt_provider('worldpose_smpl', path_dump_root='/.../__As_Single_Bbox_hjlib__')

# 3. 归约 dump_dir 里的 per-segment 预测 → 打印 per-scene + ALL 指标表
tester = Tester(testset, build_test_assembly(testset), gt_provider=gt)
tester.stage_eval(path_dump_dir='/.../inference_dumps/worldpose/full/kp_rtmlib/<exp_tag>')

# 诊断 raw no-invalid-tame 字段(仅当 dump 提供 joints_54_world_raw)
tester.stage_eval(
    path_dump_dir='/.../inference_dumps/worldpose/full/kp_rtmlib/<exp_tag>',
    pred_joints_key='joints_54_world_raw',
)
```

`stage_eval` 内部按 testset flat-index 找每段的 `<segment_tag>.pkl`(tag 由
`build_segment_tag` 定),读 dump(`dump_reader` 路由 monolith qualname,不 import
monolith)、读取 `pred_joints_key` 指定的世界空间 joint 字段、对齐 GT、按 `Eval_Meta`
算 MPJPE / T-MPJPE / Jitter。**dump 必须覆盖 testset 的每一段**(否则报缺失);用
`list_dump_segment_tags(dump_dir)` 核对覆盖。

## 公开件

| 公开件 | 用途 |
|---|---|
| `get_testset_builder(name_dataset, path_dump_root, path_filter_stats_base, filter_version=None)` | → 配好的 `TestSet_Builder`;`.build(policy, split)` 出 `TestSet`。policy ∈ {full, visualize}(curated v2/v3 deferred) |
| `get_gt_provider(name_dataset, path_dump_root, path_raw_data_root=None, path_raw_more_label=None)` | → `GT_Provider`。worldpose 只用 dump_root;jta/jta_ext 需 raw_data_root |
| `build_test_assembly(testset, encoder=None, kp_manager=None, ...)` | `TestSet` → 推理输入 `Dataset_Single_Seq_Assembly`(经 assembly 工厂 `divider=` 注入) |
| `Tester(testset, assembly, gt_provider=None, network_driver=None)` | 评测驱动:`stage_eval` / `stage_list_segments` / `stage_inference` |
| `eval_dumps_against_gt(testset, gt_provider, path_dump_dir, path_pkl_for_segment, pred_joints_key='joints_54_world')` | 无 Tester 直接归约(stage_eval 委托给它);默认 monolith tamed 字段,raw 诊断传 `'joints_54_world_raw'` |
| `build_segment_tag` / `path_pkl_for_segment` / `list_dump_segment_tags` | dump 文件名规则 + 覆盖核对 |
| `load_inference_dump(path_pkl)` | 读一个 dump → `(Test_Segment, pred_dict)`(qualname 路由;legacy bare `name_dataset`→canonical 变体,FIX-1) |
| `Eval_Meta` / `Metric_Spec_3D` / `Metric_Spec_2D_OKS` | 评测契约(指标子集 / 对齐根 / 2D 投影空间) |
| `TestSet` / `Test_Segment` / `Filter_Stats` | 测试集容器 / per-segment 元数据 / 过滤统计 |
| `GT_Provider_Base` / `Network_Driver_Base` | per-dataset GT / 推理 driver 的 ABC |
| `TestSet_Builder` / `TestSet_Builder_Base` | 配好的测试集 builder(一般经 `get_testset_builder` 取实例)/ 其 ABC |
| `compute_jitter(joints (T,J,3), fps)` | 绝对 jerk 平滑度(m/s^3) |
| `compute_joint_position_errors(target, reference)` | 相同 `(...,J,3)` 数组的未归约 per-joint Euclidean error；不持有 root/alignment/unit/reduction policy |
| `compute_keypoint_oks_matrix(reference_xy, target_xy, areas, sigmas, valid)` | method-neutral `(G,P)` OKS matrix；不持有 bbox/epsilon/matching/aggregation policy |
| `summarize_trajectory_residuals(residual, valid_frame_mask)` | 对 caller 已定义的 scalar residual population 计算 count/sum/mean/median/p95/MSE/RMSE；详见 [trajectory_residual.md](trajectory_residual.md) |
| `reduce_trajectory_residual_summaries(summaries)` | 从 sufficient statistics 精确计算 trajectory-weighted macro 与 frame-weighted micro mean/MSE |
| `Corrected_Crowd_Sequence` / `evaluate_corrected_crowd_sequence(...)` | 校验一个 normalized crowd scene，并产生可跨 scene 精确归约的 immutable summary |
| `reduce_corrected_crowd_summaries(...)` | 按 lexical scene order 合并 full/common 两个 view；输出独立 completeness、15 个 corrected metrics 与具名 ACCEL triple count |
| `compute_ppds_scores(...)` / `compute_pcod_3class_matches(...)` | 未归约的 unordered-pair crowd-layout leaves |
| `compute_joint_acceleration_errors(...)` | 未归约的 3D vector acceleration residual，单位继承输入/frame² |
| `compute_joint_jerk_errors(...)` | 未归约的 3D vector jerk residual，单位继承输入/frame³ |
| `make_coco17_visible_ge9_common_mask(...)` | 在 caller 提供的 old-common mask 上选择 mapped COCO-17 source channel `>0` count `>=9`；`0.5` 计入 |
| `evaluate_corrected_crowd_selected_view(...)` | 复用同一 15-metric 数学，输出独立 selected-view summary |
| `reduce_corrected_crowd_selected_view_summaries(...)` | lexical scene order 的 selected-view exact reduction |
| `evaluate_corrected_crowd_selected_view_and_world_dynamics(...)` | 一次 validation 同时产生 legacy 15-metric 与四项 world-dynamics scene summaries |
| `evaluate_corrected_crowd_world_dynamics(...)` | 只计算具名 selected population 的四项 world-dynamics scene summary |
| `reduce_corrected_crowd_world_dynamics_summaries(...)` | 归约 `ACC-JOINT` / `ACC-ROOT` / `JERK-JOINT` / `JERK-ROOT` exact-window sufficient statistics |
| `corrected_crowd_summary_to_json(...)` / `corrected_crowd_summary_from_json(...)` | Stable versioned worker-summary JSON round trip |

Corrected crowd input/output 的精确 shape、单位、view/metric order、invalid
contract 与 empty=`None` 语义见
[VirtualCrowd corrected protocol](../design/tasks/virtualcrowd-corrected-metric-protocol/README.md)。

## 常见注意

- **policy 范围**:`full`(评测策略)/ `visualize`(其超集);curated `visualize_v2/v3/*_smoke`
  会 narrow 场景到 split 以下,当前 raise NotImplementedError(vis-only,deferred)。
- **GT 稀疏性**:JTA/JTA_Ext 的 GT 只在 12 个 limb endpoint SMPL 槽有值,其余 NaN;
  归约只读 `Eval_Meta` 声明的指标 index(那些保证非 NaN)。NaN 出现在指标 index = 报错(契约要求两边稠密)。
- **预测字段口径**:`joints_54_world` 是标准 / monolith 等价口径。monolith 在 dump 侧用
  detector-input invalid frame mask 对 world translation 做插值/tame,然后写入该字段。
  `joints_54_world_raw` 是 no-invalid-tame 诊断口径;无 KP 帧的 raw root/world translation
  可为 0、fallback 或大幅漂移,因此 MPJPE 变差是预期诊断信号,不是新的标准协议。
- **camera/ground/video GT** 当前 deferred(base raise):服务 2D OKS(未实现)+ vis(out of scope)。
- **vrv1**:`get_*_provider('vrv1')` raise(assembly 当前 out-of-scope)。

详细决策 / 设计见 [../design/README.md](../design/README.md)。
