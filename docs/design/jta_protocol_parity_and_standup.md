# JTA 协议 parity + 接入 experiments 的 standup(驻地)

> **本文是「把标准 eval 站起来服务 campaign 02」这件事的驻地**(2026-06-29 立)。
> 它收:① 厘清后的关系(hjlib-evaluation ↔ monolith 录值 ↔ campaign 02 的 JTA gap);
> ② 实测证据(monolith 真 dump 过本仓协议的结果 vs 录值);③ 两阶段计划(先通后保真)。
> 关联:experiments `campaigns/02_reproduce_monolith`(消费方)、`docs/issues/jta_unfiltered_eval_gap.md`
> (`EXP-JTA-FAR`)、本仓 [migration.md](migration.md) DIV-7(JTA GT 源改动)。

## 0. 一句话

本仓的协议**代码已忠实迁出且 filter 逐帧 bit-faithful**(GPU-free 跑 monolith 真 dump 可验),
**但它现在还复现不出 monolith 论文录头的 JTA 421** —— 因为 ① 缺论文头用的 **`h2p5_p75`
scene filter**(= "lowcam" cull),② JTA GT 源(DIV-7,dataset-std 22→54 remap)与 monolith
`JTA_Raw_Label` 有残余 parity 差。两件补齐前,它不能当 campaign 02 的闭合 yardstick。

## 1. 实测证据(2026-06-29,GPU-free eval-on-dumps)

把 monolith 真 JTA `full` inference dump(ckpt `ief_global_v06_det_iter_v3_noalign_ep0017`,
1967 segments)喂本仓 `get_testset_builder('jta_smpl_fitted', policy='full', split='test')` +
`get_gt_provider` + `eval_dumps_against_gt`,对比 monolith master JSON 同 ckpt 录值:

| 口径(同一 ckpt 的同一批预测) | scenes | frames | MPJPE(mm) | T-MPJPE(mm) |
|---|---|---|---|---|
| monolith **h2p5_p75**(论文录头「filtered」= 421) | 22 | 218,174 | **420.96** | 88.27 |
| monolith **nofilter** | 61 | 515,149 | 820.35 | 85.77 |
| **本仓 `full`(seq_modifications)** —— 实测 | — | 549,593 | **1037.89** | 116.29 |

录值来源:`dynamic_hvip/data/eval/quantitative_eval/master/JTA_full_test{,__nofilter}.json`
(filter `h2p5_p75` = `height_m>=2.5 AND pitch_deg<=75`,scene-level OR-keep)。
实测脚本:`/tmp/claude-scratch/2026-06-29/89704573_eval_jta_protocol_confirm.py`。

**读数:**
- 本仓 `full`(549k 帧)≈ monolith **nofilter**(515k 帧)——同一个「per-frame seq_modifications
  基本全过」的人口;**不是**论文头的 h2p5_p75 人口(那把 61→22 scene 砍到 218k 帧)。
- 故 **本仓现在的 `policy='full'` 对应的是 monolith 的 nofilter 档,而非论文 filtered 档**。
- 即便和 nofilter 比,仍有差:**T-MPJPE 116 vs 86(+35%)** + MPJPE 1038 vs 820。T-MPJPE 是
  去平移(root-relative)的,故这**不是远人/平移**问题,而是 **GT 关节本身不同** —— 指向 DIV-7。

## 2. 厘清的关系(三个 filter 别再混)

| regime | 在哪 | 帧口径 | 谁用 |
|---|---|---|---|
| **`seq_modifications`** 6-dim per-frame trim | 已 washed 进 `jta_filter_stats/seq_modifications_jsonbin`;本仓 `policy='full'` 读它 | ~549k(≈nofilter) | **本仓现状** |
| **`h2p5_p75`** scene-level camera height/pitch OR-keep | `dynamic_hvip/data/jta_camera_height_and_pitch/jta_kept_h2p5_p75.txt` | 218k(22 scene) | **monolith 论文录头(421)**;本仓**未实现** |
| **ad-hoc `max_side>60 + min_run120`** 装配 trim | campaign scratch(assembly `_v2` leaf) | 601k | campaign §9 的 0.851 m(另一个新 ckpt) |

> campaign README/§9 把「monolith eval-协议 full 549,593 帧」当成了协议;实则论文头 421 用的是
> **h2p5_p75**(更狠的 scene cull)。本文实测坐实:本仓 `full` = nofilter 档。这是对 campaign
> 既有理解的更正(已在本文锚定,后续回填 campaign 决策日志)。

## 3. 计划(user 2026-06-29 定:**两步走,先通后保真**)

### Phase 1 —— 通管线(live driver,让协议能评新 ckpt)
**现状**:experiments **零接** hjlib-evaluation;`eval_final.py` 走 trainer `mgr.validate` 的简易
metric(无 mask plain mean,= 5.5/0.851 那套)。要评 campaign 自己的新 ckpt 必须先有 live driver。

**落点(设计,待 Phase-1 session 确认/微调)**:
- **driver 放 experiments(app 层),不放本仓** —— 依赖铁律 `experiments → {evaluation, network}`;
  concrete driver 要 `hjlib-network` 载 `Seq_Estimator` + experiments 自己的 dict_batch adapter。本仓只留
  `Network_Driver_Base` ABC(已在),Phase 1 **不动本仓**(它已 parity-verified)。
- 新件:experiments `smpl_ief_global/` 下一个 `Seq_Estimator_Eval_Driver(Network_Driver_Base)`:
  `infer({'sample': Single_Seq_Sample}) -> {'joints_54_world': (L,54,3)}`。
  - 载 ckpt:参 `eval_final.py` 的 mgr 路径 / `train_assembly.py` ckpt 发现。
  - batch 适配:**复用** `smpl_ief_global/real_dataset.py::adapt_batch_to_dict_batch`
    (+ `collate_assembly_to_dict_batch`),**别重造**(design README「与 network 的接线」)。
  - 输出键对齐本仓 reducer 期望的 `{'segment','pred':{'joints_54_world'}}`。
- 新 entry:experiments 一个 `eval_protocol_run.py`(或扩 `eval_final.py`):
  `get_testset_builder/get_gt_provider/build_test_assembly` + driver → `Tester.stage_inference(dump_dir)`
  → `stage_eval(dump_dir)`。先在**已有新栈 ckpt `ief_v07_bedlam/epoch-0009`** 上跑通(JTA `full`),
  拿到新栈预测过协议的数(与 §1 的 1037 档同口径,apples-to-apples)。
- 验收:end-to-end 出一张 JTA `full` 指标表(无需 GPU 授权时可先 worldpose 小集 smoke;JTA 全量需授权后台)。

**Phase 1 实测(2026-06-29 完成,里程碑:新栈预测首次过标准协议):**

driver 落点(均在 **experiments** app 层,本仓 hjlib-evaluation 一字未动):
- `smpl_ief_global/eval_protocol_driver.py` —— `Seq_Estimator_Eval_Driver(Network_Driver_Base)`:
  `infer({'sample': Single_Seq_Sample}) -> {'joints_54_world': (L,54,3)}`(米)。复用
  `real_dataset.collate_assembly_to_dict_batch`(B=1)+ `forward_iterations(flag_with_gt=False)` +
  `preds_world.joints`;独立驱动需自己 `network.transfer_batch_to_device`(Lightning 平时搬 batch,
  `Batch_Input` 只搬一部分张量,RT_world_to_camera 等留 CPU)。
- `smpl_ief_global/eval_protocol_run.py` —— entry:`get_testset_builder/get_gt_provider/build_test_assembly`
  + `Tester.stage_inference -> stage_eval`;`--num_take`/`--restrict_scenes` 截断 TestSet(smoke)。
- assembly 走 `build_test_assembly(encoder=, kp_manager=, flag_test_mode=False)`:注入和训练同款
  feature-cache encoder + 检测 KP,从而原样复用 adapt(GT 只为满足 collate 而载,推理 flag_with_gt=False 丢弃)。
- 新增 `local_setting.py` 两根:`PATH_FILTER_STATS_BASE` / `PATH_JTA_DATA_ROOT`。

新栈 ckpt `ief_v07_bedlam/epoch-0009` 的 JTA `full` 指标表(**口径 1967 seg / 549,593 帧 = §1「本仓 full」**,
apples-to-apples;fps=30,meta=2026-05-02_v1):

| metric variant | MPJPE(mm) | T-MPJPE(mm) | JitRatio | frames |
|---|---|---|---|---|
| limb_endpoints_with_hip | 1290.82 | 93.05 | 60.93 | 549,593 |
| limb_endpoints_no_hip | 1292.31 | 104.40 | 60.18 | 549,593 |

读数:管线已通,标准协议能端到端评新栈 ckpt。**注意此 1291 与 §1 的 1037 ckpt 不同**(§1 = monolith 老
ckpt `ief_global_v06_det_iter_v3` 的 dump;此处 = 新栈 v07_bedlam ep9,只 train bedlam+h36m 10 epoch,JTA 是
zero-shot 测试集),故二者是同口径不同 ckpt,非直接优劣对比;Phase 1 的验收点是「新栈预测过协议」本身,数值
基线随训练成熟而动。worldpose smoke(num_take 4)亦绿(MPJPE 456.94 / T-MPJPE 58.41)。

**Phase 1 追加:v06 inference-parity(2026-06-29,坐实 live driver 忠实复现 monolith 前向):**

同一 v06 权重(`ief_global_v06_det/epoch-0005`,monolith 训)remap-load 进新栈 `Seq_Estimator`
(driver 加 `--ckpt_kind monolith`,复用 `parity_vs_monolith.build_new_trained`:剥 `pipeline.` 前缀
+ copy 5 个 `mean_*` buffer —— 裸 `load_from_checkpoint` 会 key mismatch),与 monolith 现成 per-segment
dump 逐 segment 比 `joints_54_world`(monolith 端零 GPU):

- **worldpose**(195 段):valid-frame pred mean-abs-diff **2.3e-6 m**;valid-frame **MPJPE 401.27 /
  T-MPJPE 59.32 vs monolith 录值 401.36 / 59.34**(差 <0.1mm)→ **inference parity 坐实**。
- **JTA**(monolith dump 695 段 ⊂ 新栈 1967,比交集 223,885 valid 帧):pred mean **3.1e-6 m**;
  99.977% 帧 <1mm,仅 2/223885 帧 >10cm(孤立单帧数值分叉,非系统性)→ **pred-parity 坐实**。
  (JTA metric 不与 monolith 录值比 —— 录值用 h2p5_p75/nofilter + 不同 ckpt iter_v3 + DIV-7,见 §1。)

**invalid 帧处理(已挖清 monolith 真相 + 已复现,2026-06-29):** all-frame **绝对 MPJPE 曾被 invalid 帧
污染**(WP 不处理时 416 vs 401)。挖 monolith 后厘清:① monolith 的 frame 级 invalid 判据是 **detector-based**
——`(kp[...,3]>0).sum()==0`(17 个检测 KP 全失效;单 KP 有效 = `score>=2.0 AND 落 person bbox 内`),
**没有数据侧 per-frame 判据**;② monolith **不是 mask 排除**,而是 **driver 侧对 invalid 帧的 world root 线性
插值驯服**(joints 按 root delta 平移),reducer 照样全帧计分。③ 本仓 `eval_reducer` 行为**已与 monolith 一致**
(都全帧不 mask;WorldPose 交叉验证:本仓 reducer 跑 **monolith dump** = 401.36 == 录值
→ reducer+WP GT 逐帧 bit-faithful;JTA GT parity 的残余问题仍按 §1/DIV-7 单独看)。
故修复点在 **driver(experiments),非本仓**:`eval_protocol_driver.py` 复刻了 monolith 的 detector mask +
线性插值驯服(`tame_invalid=True` 默认)。**复现结果:WP all-frame MPJPE=401.36 / T-MPJPE=59.34 == monolith
录值,完整 apples-to-apples(含绝对 MPJPE)。** T-MPJPE 因 root-relative 一直吻合(平移 garbage 被抵消)。

**交接(独立改进,未做):** monolith 的判据是 detector-dependent(依赖 rtmlib),**不可移植**。一个**数据侧**
per-frame 判据(GT 可见性 / bbox 面积 / 出界)更干净可移植,但会偏离 monolith 录值 → 属协议设计改动,user
2026-06-29 定「先复现后改进」,改进留作独立 Phase 2+ 议题。(注:v06 JTA dump 在 taming 前产出、未驯服;JTA
parity 靠 valid-frame pred-parity 3μm 成立,不受影响;JTA all-frame 非 parity 靶。)

### Phase 2 —— 保真(补到论文头 421)
1. **加 `h2p5_p75` scene filter**:把 `jta_kept_h2p5_p75.txt` 的 scene-keep 接进协议
   (本仓 testset 的 `restrict_to_scenes` 已有 hook,或 policy 扩展)。这是从 1037 档走到 421 档的**主**杠杆。
2. **核 JTA GT parity(DIV-7)**:查清 T-MPJPE 116 vs 86 的 GT 散度 —— dataset-std 22→54 remap /
   关节定义 / 单位 vs monolith `JTA_Raw_Label`。修到 nofilter 档 T-MPJPE 对齐(~86)。
3. 两件齐 → 本仓应能在 monolith dump 上复出 h2p5_p75 的 ~421,届时它成为 campaign 02 真闭合 yardstick。

## 4. 给接手 session 的合同
- 读本文 §1-§3 即可接 Phase 1。**Phase 1 代码落 experiments,不动本仓**(本仓 parity-verified,勿破)。
- 设计先行(family norm:严禁直接糊代码);driver 放置如 §3 有开放点先与 user 确认。
- 跑法:env `/home/hj/softs/miniconda3/envs/hjlib_py312/bin/python`;pyright strict 带
  `--pythonpath <那个 python>`;改 experiments 文件走 snapshot-and-diff。
- Phase 2 回本仓做(filter + GT parity),届时更新本文 §1 表 + campaign 决策日志。
