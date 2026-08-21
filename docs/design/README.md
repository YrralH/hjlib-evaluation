# 设计 —— hjlib-evaluation

修改本仓的唯一 onboarding 入口(family 规定:每仓仅一个 onboarding doc,无
`docs/CLAUDE.md`)。

## Scope

- **是什么**:动态场景评测协议的执行层 harness。给一个训好的
  `hjlib-network` estimator(`Seq_Estimator`,`L.LightningModule`)+ 一个 filter
  配置好的 `hjlib-dataset-assembly` Dataset(测试集),它:
  1. 枚举 test segment(哪个 dataset / scene / raw-seq / person / 帧区间);
  2. 逐 segment 跑网络推理(`Network_Driver`,载 ckpt + 适配 `dict_batch`);
  3. 从 per-dataset `GT_Provider` 取世界空间原始 GT(SMPL joints / params / 相机
     K,RT / 地面);
  4. 按 per-dataset `Eval_Meta` 把预测对齐 GT、归约出指标表(MPJPE / T-MPJPE / Jitter,
     2D OKS 占位)。
- **不做什么**:不持有任何**模型定义**(模型在 hjlib-network)或**数据集定义 / 取数
  逻辑**(在 hjlib-dataset-assembly);不自建训练;不做 filter 生产(复用 assembly
  已 washed 的 `Filter_Modifications_Store`,见下「与 assembly 的接线」)。
- **来源**:从 monolith `lib_dynamic_hvip/test/protocol_dynamic/`(pin
  `2bc42db4`)整体迁出。见 [migration.md](migration.md)。

## 依赖方向(铁律)

```
hjlib-experiments → { hjlib-evaluation, hjlib-network, vis 仓 }      # app/编排层
hjlib-evaluation  → { hjlib-dataset-assembly, hjlib-dataset-std,
                      hjlib-skeleton, hjlib-geometry, hjlib-detection,
                      hjlib-ground-solver }             # 执行层(本仓);已 pin
                  ( + hjlib-network / hjlib-smpl )      # live driver 落地时再 pin
```

> dep 图增补(2026-06-24,用户确认):设计 SSOT 原列 `{assembly, network} (+smpl/geometry)`;
> Phase 3 GT provider 需 raw 数据集 GT(JTA 22-joint + WP joints/param),故加
> **hjlib-dataset-std**(raw GT 访问)+ **hjlib-skeleton**(JTA-22→SMPL-54 joint name map)。
> `[tool.hjlibm.deps]` pin assembly / dataset-std / skeleton / geometry。
> geometry supplies corrected crowd point-registration fits; network / smpl
> remain deferred to live network_driver. Dependency direction remains acyclic.

- 与 **hjlib-trainer 对称**:都是执行层(消费 network + assembly),`hjlib-experiments`
  在其上编排。trainer 是 model-/dataset-agnostic 的纯 leaf;evaluation 因为要做
  per-dataset GT 接线 + 指标,**允许**依赖 assembly + network(这是二者的设计差异,
  评测的 GT/指标天然 dataset-coupled)。
- **无环**:`hjlib-network` 是纯下游,不反向依赖 assembly / trainer / evaluation
  (建仓前由 validity_filtering_analysis.md §5 D5 核实)。

## 四层架构里的位置

| 层 | 仓 | 职责 |
|---|---|---|
| app / 编排 | `hjlib-experiments` | train/eval 入口 + adapter seam |
| 训练编排 | `hjlib-trainer` | model-/dataset-agnostic 训练 harness(纯 leaf) |
| **评测编排** | **`hjlib-evaluation`(本仓)** | 评测 harness + per-dataset GT/testset/指标接线 |
| 模型 | `hjlib-network` | 纯定义;对外契约仅 `dict_batch` |
| 数据 | `hjlib-dataset-assembly` | 产出 Dataset + 按参数选 split/filter;owns collate |

设计 SSOT:`hjlib-dataset-assembly/docs/design/validity_filtering_analysis.md`
(§5 四层架构 / §6 track 2 / §7.2 evaluation protocol filtering)。

## 与 hjlib-dataset-assembly 的接线(关键)

**复用 assembly 已 washed 的 filter,不港 415 行 filter 生产**:filter 数据已 washed 进
`Filter_Modifications_Store` v1(根 `<dataset>_filter_stats/seq_modifications_jsonbin`)。
本仓的 `TestSet_Builder` 读该 store,用 assembly 的 `partition_filtered_sub_seq` 在**一个
循环里**同时造 `Filtered_Sub_Seq_Divider` 与 scene-level `Test_Segment` 列表(保证 GT 查找
按 flat-index 对齐),构成 `TestSet`。

**测试集 Dataset 经 assembly 工厂的 `divider=` 注入构造**(`build_test_assembly` →
`get_dataset_seq_assembly(path_root, name_dataset, fps, divider=testset.divider, ...)`):

- assembly 的 `get_dataset_seq_assembly` 加了 additive 的 `divider: Optional[Seq_Divider]`
  注入参(本仓引入,见 migration.md DIV-6),工厂用注入的 divider、跳过 split/config 的
  divider 构造,但仍封装 label-manager / encoder / Dataset 接线。
- **为何注入 divider,而非 by-parameter `Assembly_Config_Filtered_Seq`**:eval 必须把 divider
  与 `Test_Segment` 在同一循环造(对齐)、且要支持 `TestSet.restrict_to_scenes` + curated 策略,
  这些 by-parameter config 路径表达不了(它从 `split` 重建 divider,无法收窄/对齐)。所以 eval
  **不走** `Assembly_Config_Filtered_Seq`(那是 assembly 给「按参数选 filter」调用方的另一条路)。
  详见 migration.md DIV-1。

## 与 hjlib-network 的接线(network_driver)

`hjlib-network` 的 `Seq_Estimator`(`L.LightningModule`)对外只吃 `dict_batch`
(见 network `batch_input/CONTRACT.md`)。assembly 的 `Single_Seq_Sample_Batch` →
`dict_batch` 的翻译当前由 experiments 的 adapter
(`hjlib-experiments/smpl_ief_global/real_dataset.py::adapt_batch_to_dict_batch`)做。
本仓的 `Network_Driver` 需要类似适配(参考它,**别重造**)。

## Repo layout(monolith protocol_dynamic → src/hjlib_evaluation)

file-mapping port（monolith `test/protocol_dynamic/` → 本仓;逐文件对应见
[migration.md](migration.md) §4）。

```
src/hjlib_evaluation/
    __init__.py            顶层 re-export(全部公开件)
    py.typed               PEP 561 标记
    test_segment.py        Test_Segment(per-item 元数据;scene-level 帧坐标)
    testset.py             TestSet + Filter_Stats(只读容器,持 divider + scene-level segments + build recipe)
    testset_builder_base.py  TestSet_Builder_Base(ABC)
    testset_builder.py     TestSet_Builder(单一泛型,合并 monolith wp/jta/jta_ext;读 washed filter store)
    eval_meta.py           Eval_Meta / Metric_Spec_3D / Metric_Spec_2D_OKS(评测契约)
    eval_reducer.py        eval_dumps_against_gt(预测 vs GT → MPJPE/T-MPJPE/Jitter;pred_joints_key 字段选择)+ compute_jitter
    joint_error.py         method-neutral unreduced per-joint Euclidean errors
    joint_acceleration.py  method-neutral GT-relative joint acceleration residuals
    joint_jerk.py          method-neutral GT-relative joint jerk residuals
    keypoint_oks.py        method-neutral pairwise OKS matrix leaf
    trajectory_residual.py generic scalar trajectory residual summary + macro/micro reduction
    corrected_crowd_population.py named additive selected-population mask
    corrected_crowd_world_dynamics.py additive exact-window world dynamics summary/result
    ground_estimation_protocol.py generic Tracked_Scene observation selection/sampling + RCR solve + plane/same-ray diagnostics
    dump_reader.py         load_inference_dump(qualname-routing unpickler,读 monolith 真 dump 零 monolith import;legacy bare name→canonical 归一,FIX-1)
    gt_provider_base.py    GT_Provider_Base(ABC;joints/param/eval_meta 抽象,camera/ground/video deferred→raise)
    network_driver_base.py Network_Driver_Base(ABC,infer(dict_item)->dict;live driver deferred)
    assembly_factory.py    build_test_assembly(经 assembly 工厂 divider= 注入建测试集 Dataset)
    get_by_dataset.py      name_dataset → TestSet_Builder / GT_Provider 工厂(注入数据根)
    tester.py              Tester(stage_eval / list_segments / inference)+ build_segment_tag + path_pkl_for_segment
    per_dataset/
        wp_eval_meta.py        WP_EVAL_META(全 SMPL-24)
        jta_eval_meta.py       JTA_EVAL_META / JTA_EXT_EVAL_META(12·10 limb endpoints)
        gt_provider_wp.py      WP_GT_Provider(从 assembly dump full label)
        gt_provider_jta.py     JTA_GT_Provider / JTA_Ext_GT_Provider(从 dataset-std raw 22-joint)
docs/{usage,design}/       用法 / 设计两棵树(design/README 本文件 = 唯一 onboarding 入口)
script/evaluate_virtualcrowd_rcr_ground.py  VirtualCrowd dataset-specific RCR experiment entry
script/evaluate_virtualcrowd_density_balanced_rcr_ground.py strict single-arm unweighted/KDE operation
script/evaluate_virtualcrowd_density_balanced_rcr_cartesian.py prepared 3x2x2 matrix; default prepare-only, explicit future execution
test_smoke/                合成数据 smoke(test_testset + test_gt + master runner)
test/                      真实数据 FAIL-not-skip(testset / GT / eval-on-dumps + local_setting_test)
```

**vrv1 先占位/跳过**(无 per_dataset 实装):assembly 当前把 vrv1 列为 out-of-scope(washed
时跳过),故 vrv1 评测待 assembly 支持后再纳入(用户明确)。**curated 策略**(visualize_v2/v3/
smoke)deferred(narrow scenes below split,vis-only)。

## Family conventions inherited

- pyright `strict`(`pyrightconfig.json`),见 family memory `pyright-strict-default`。
- 覆盖父类方法加 `@override`(stdlib typing),`reportImplicitOverride=error` 强制;见
  family memory `pyright-override`。
- 字符串 `'%s' % x` 单引号;注释英文标点;缩进 4 空格整数倍。
- 命名:类名实词大写+蛇形 / acronym 整段大写;禁 `utils_` 前缀。见
  family memory `naming-style`（canonical owner=`hjlib-agent`）。
- 测试两棵树 `test_smoke/`(无数据)vs `test/`(真实数据 FAIL-not-skip),见
  `hjlib_standard/test_layout.md` 与 [test.md](test.md)。

## Must-read(建议顺序)

1. 本文件「依赖方向铁律」+「与 assembly / network 的接线」—— 改本仓前必须守住的线。
2. [migration.md](migration.md) —— 从 monolith 迁出的 port 表 / divergence / 测试状态。
3. [jta_protocol_parity_and_standup.md](jta_protocol_parity_and_standup.md) —— campaign 02
   标准 eval standup / JTA parity 的实测驻地(读数和阶段计划)。
4. [testing_protocol_from_monolith.md](testing_protocol_from_monolith.md) —— monolith
   test filter 设计复盘 / 当前筛选逻辑 / 筛选比例驻地。
5. `hjlib-dataset-assembly/docs/design/validity_filtering_analysis.md` —— 本 initiative
   设计 SSOT(§5 四层架构 / §6 track 2)。
6. [test.md](test.md) —— 测试两棵树本仓如何实例化。
7. [fixed_window_testset.md](fixed_window_testset.md) —— `TestSet.make_fixed_window_testset(...)`
   的 fixed-window subset contract; cached-fusion eval consumes this API from experiments.
8. [trajectory_residual.md](trajectory_residual.md) —— generic scalar residual summary/reduction 的数学与 ownership 边界。
9. [VirtualCrowd corrected metric protocol](tasks/virtualcrowd-corrected-metric-protocol/README.md)
   —— Campaign 03 T3 的 reviewed population、completeness、metric 与
   reduction 数学，以及后续 Code Architecture residence。
10. [VirtualCrowd RCR ground evaluation](tasks/virtualcrowd-rcr-ground-evaluation/README.md)
    —— `Tracked_Scene` observation protocol、RCR baseline 与 same-ray ground-effect
    evaluation 的数学和代码边界。
11. [VirtualCrowd density-balanced RCR ground](tasks/virtualcrowd-density-balanced-rcr-ground/README.md)
    —— strict full-observation population、density intermediate、weighted RCR、
    normal/D decomposition 与四版本真实结果。
12. [VirtualCrowd corrected selected population](tasks/virtualcrowd-corrected-population-profile/README.md)
    —— 在不改变 legacy corrected schema/result 的前提下，定义
    `C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9` 的 mask、selected-view schema 与
    exact reduction boundary。
13. [Corrected-crowd world dynamics](tasks/corrected-crowd-world-dynamics/README.md)
    —— additive `ACC-JOINT` / `ACC-ROOT` / `JERK-JOINT` / `JERK-ROOT`
    GT-relative world-space temporal metrics、reference status 与 exact-window
    reduction boundary。
14. [VirtualCrowd evaluation profiles](tasks/virtualcrowd-evaluation-profiles/README.md)
    —— 区分 population / association / metric / complete evaluation profile，
    冻结 `VC_HJ_DEFAULT_V1` 与 `VC_CROWD4D_NATIVE_V1` 的名字、组成和比较边界。

## Dump prediction field contract

`eval_reducer.eval_dumps_against_gt` 和 `Tester.stage_eval` 默认读取
`pred['joints_54_world']`。这是本仓的标准 protocol 字段,也与 monolith 的
`protocol_dynamic` eval 等价:monolith 在 dump 侧先根据 detector/input KP 的
`invalid_frame_mask` 对 `transl_raw` 做线性插值/tame,再把同一个 translation delta
平移到 `joints_54_world`。reducer 本身不按 invalid mask 分支。

`pred_joints_key='joints_54_world_raw'` 是显式诊断通道,用于观察 no-invalid-tame
raw 输出。它不是新标准 protocol:无 KP / 无观测帧的 raw root translation 可能来自
0、fallback 或任意失败值,所以绝对 MPJPE 变差是预期信号;T-MPJPE 常常接近不变,因为
主要差异落在每帧平移上。

若未来要把 invalid 处理改成数据侧 validity(bbox/visibility/in-image/GT availability),
应作为 protocol redesign 单独落地,不得默默替换默认 `joints_54_world` 口径。

## State of the world

- Data-free smoke: 59 passed on 2026-08-19. The changed corrected/dynamics
  modules and test use strict targeted pyright with 0 errors.

- **2026-08-19 corrected world dynamics**：新增独立 schema-v1 dynamics
  summary/result，不改变原 15-metric corrected schema-v1。GroupRec 在同一
  159,405-occurrence population 上完成 8 scenes：156,263 acceleration triples、
  154,883 jerk quadruples；重新发布的 legacy result 与接受版本 SHA-256
  `f7c36b7...c2c36` byte-identical。

- **2026-08-19 GroupRec three-method corrected evaluation**：新增 additive
  selected-population contract；旧 167,243-key common manifest/result 未改写。
  `C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9` 精确包含 159,405 occurrences，三种
  method 均完成 8 scenes、159,405 matched occurrences 与 156,263 acceleration
  triples 的同口径 reduction。完整 provenance/result 见 Campaign 04。

- **2026-08-18 VirtualCrowd density-balanced RCR**：在严格 confidence>4、
  ankle/bbox ratio<0.20 的完整 17,992 observations 上完成 exact-LOO Gaussian
  KDE + Scott bandwidth single arm，并在同一 167,243 GT-ray support 评估。Global
  mean same-ray error 从 unweighted 18.8961 m 降到 KDE 16.4576 m（12.90%），但
  normal-oracle / distance-only 没有同步改善，记录为包含组合误差 cancellation。
  历史 k=16/32/64 结果保留为 fixed-scale 探索。后续 3 confidence × 2
  ankle-ratio × 2 density-mode matrix 已完成 12-arm 求解、plain result 写入和
  independent source reconstruction/readback。六组 KDE 均降低 combined mean；
  raw best 为 `confidence>5.0, ankle<0.20, KDE = 15.7277 m`，但该 arm 最小
  scene 仅 256 observations，且全部六组 normal-oracle/distance-only means 同时
  变差，故保留低 support 与 error-cancellation 限制。

- **2026-08-18 VirtualCrowd RCR baseline**：新增 generic ground-observation
  sampling / RCR solve / same-ray evaluation public API 与 dataset-specific script。
  Portable smoke 43 passed；本次新增 module/script/test targeted Pyright strict 0。
  真实 headline 使用 8×5,000 sampled observations，完整 167,243 support 的
  mean/median ground-effect error 为 14.0449/10.4662 m；first-frame diagnostic
  有 5/8 scene 因非 forward full-support intersection 被整体标 invalid。

- **2026-06-30 field-select eval**:`eval_dumps_against_gt(..., pred_joints_key=...)` 与
  `Tester.stage_eval(..., pred_joints_key=...)` 支持同一套 TestSet/GT 同时读
  monolith-equivalent `joints_54_world` 和 raw diagnostic `joints_54_world_raw`;
  默认值保持 `joints_54_world`。新增 smoke 覆盖字段选择;pyright strict 0。
- 2026-06-24 仓初建 + **端口 Phase 1-5 done**(脚手架 / testset 层 / GT 层 / network_driver ABC /
  tester + eval_reducer + dump_reader);**Phase 6 parity + behavior 已在 `hjlib-migration-tests/evaluation/`
  落地并跑绿**(2026-06-25,只差 SHA pin —— 见 What's open)。
- **FIX-1(2026-06-25,parity 时发现)**:`dump_reader` 把 legacy monolith dump 的 bare `name_dataset`
  (`worldpose` 等)硬映射到 canonical 变体(`worldpose_smpl` 等)。否则 reducer 的 `seg==seg_on_disk`
  全等会因 assembly 把 bare 名 deprecate 而炸(只 `name_dataset` 一字段不符)。见 migration.md §7 / DIV-9。
- `pip install -e .` PASS；本次 `src` + `test_smoke` targeted pyright **strict, 0 errors**
  （CLI 须带 `--pythonpath <hjlib_py312>/bin/python`）。`test/` 的两个真实数据模块
  在 import 时要求 gitignored `local_setting_test`，因此未配置机器上不构成 data-free
  full-repo Pyright 0；这是既有 repo-level open item。
- **Phase 2(testset)**:`test_segment` / `testset`(+ Filter_Stats)/ `testset_builder`(单一泛型,
  合并 wp/jta/jta_ext)/ `testset_builder_base` / `get_by_dataset` / `assembly_factory`
  (`build_test_assembly` 经工厂 `divider=` 注入)。
- **Phase 3(GT,scoped 指标 baseline)**:`eval_meta` + per-dataset metas + `gt_provider_base`
  (ABC;joints/param/eval_meta abstract,camera/ground/video deferred→raise)+ `WP_GT_Provider`
  (从 assembly dump full label)+ `JTA/JTA_Ext_GT_Provider`(从 dataset-std raw 22-joint,22→54
  limb remap)+ `get_gt_provider`。**新增 deps = hjlib-dataset-std + hjlib-skeleton**(用户确认)。
  - **real-data 验证绿**:worldpose/test build=195 seg、jta_ext/test=122 seg(divider↔segment
    对齐 + scene offset + assembly len);WP GT joints (L,54,3) SMPL_24 非 NaN + param;jta_ext GT
    joints (L,54,3) 12 limb slot 非 NaN + param raise。smoke(无数据)`test_testset` + `test_gt` 全绿。
- **Phase 4(network_driver)**:`Network_Driver_Base` ABC 迁;**live driver deferred**(DIV-10)
  —— 迁移 + parity 走 monolith 真 inference dump,不需 live 网络。
- **Phase 5(tester + eval)**:`eval_meta`(已在 Phase 3)+ `eval_reducer`(`eval_dumps_against_gt`,
  MPJPE/T-MPJPE/Jitter,verbatim 数学,`compute_jitter` inline)+ `dump_reader`(qualname-routing,
  读 monolith 真 dump 零 monolith import,DIV-9)+ `tester`(stage_eval / stage_list_segments /
  stage_inference;stage_vis 不迁)。
  - **real-data 验证绿(eval-on-real-dumps)**:`test/test_eval_on_dumps_with_data.py` 在 **195
    个 monolith 真 worldpose 预测 dump**(`ablation_a00...ep0004`)上跑 `stage_eval` →
    SMPL_24_full ALL **MPJPE=655.97 / T-MPJPE=57.56 mm**(145754 frames / 11 scenes;split=test
    与 dump 段完全对齐)。data-dependent suite 全绿;smoke + pyright strict 0。
- **跨仓改动(随 assembly 同批落)**:assembly `get_dataset_seq_assembly` 加 `divider=` 注入参 +
  `py.typed`(migration.md DIV-6),且把 bare 数据集名 deprecate 为 canonical 后缀名(`get_dataset_facts`
  对 bare 名 raise)—— FIX-1 即应对此。`[tool.hjlibm.deps]` 钉 assembly / dataset-std / skeleton
  (assembly SHA 待 commit 后 `hjlibm update` bump)。
- GitHub remote：`YrralH/hjlib-evaluation` 已建立；`main` 由 family gitsync
  workflow 维护与 `origin/main` 的同步。

## What's open

- **Phase 6 parity + behavior**:已在 `hjlib-migration-tests/evaluation/` 落地并跑绿(capstone
  end-to-end wp×2+jta×2 + enumeration/GT 含 jta_ext + behavior;monolith 侧经 `py312th280cu128`
  subprocess 跑活的——本仓 env 进不去 monolith 重栈 yacs)。**只差 SHA pin**:待 assembly + eval commit
  后钉 `(monolith 2bc42db4, new_lib_sha)` 进 migration-tests README + 本仓 migration.md Phase 2 + 台账。
- **live network_driver**(DIV-10 deferred):评新 ckpt 用(载 ckpt + dict_batch 适配,参考
  experiments `adapt_batch_to_dict_batch`);非迁移 parity 关键路径。
- **GT camera/ground/video**(DIV-8 deferred):2D OKS(stage_eval 未实现)+ vis,待取用时经
  dataset-std 接。
- **curated 策略(visualize_v2/v3/smoke)**:deferred(narrow scenes below split,工厂
  split 路径表达不了;vis-only,非指标 baseline)。见 EVAL Cross-lib TODO。
- **evaluation protocol filtering 重设计**(§7.2 D6,multi-path / optional 模型):在
  评测 baseline 之后,**不在本端口范围**。
