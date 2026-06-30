# 用法 —— hjlib-evaluation

调用方视角:我有一个数据集 + 一批预测(已存 dump,或一个待评的 ckpt),怎么算出
世界空间 MPJPE / T-MPJPE。

## 一句话索引

评测分两段、由 per-segment 预测 dump 衔接(`{'segment': Test_Segment, 'pred':
{'joints_54_world': (L,54,3)}}`):**推理**(逐 segment 跑网络 → 落 dump)与**归约**
(读 dump + GT → 指标表)。两段解耦,所以**评已有 dump 不需要 live 网络**。
默认归约字段是 `pred['joints_54_world']`,即 monolith 等价的 dump-side tamed
协议;如需诊断 no-invalid-tame raw 输出,显式传 `pred_joints_key='joints_54_world_raw'`。

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
