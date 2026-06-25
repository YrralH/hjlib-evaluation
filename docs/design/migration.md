# 迁移记录 —— hjlib-evaluation(从 monolith protocol_dynamic 迁出)

> 进度:**端口 Phase 1-5 done**(2026-06-24,real-data 验证绿):脚手架 + testset 层 +
> GT 层 + tester / eval_reducer / dump_reader。下表「What was ported」全部已勾。**Phase 6
> parity + behavior 已在 `hjlib-migration-tests/evaluation/` 跑绿**(2026-06-25;capstone
> end-to-end + enumeration/GT 含 jta_ext + behavior),**只差 SHA pin**(见下「Migration test
> status」Phase 2);commit/push 待 family 收敛。

## 1. Source

- 仓:monolith `lib_dynamic_hvip`(`/home/hj/Repo/dynamic_hvip`)
- 子目录:`lib_dynamic_hvip/test/protocol_dynamic/`(~18 文件,~2333 行)
- entry:`script/test/test_protocol_dynamic/run_test.py`
- 捕获日期:2026-06-24
- pin:`2bc42db41613420a8311cb9a8877b7c2de298e09`(monolith 冻结)

## 2. Destination

- lib:`hjlib-evaluation`,import `hjlib_evaluation`
- src 路径:`src/hjlib_evaluation/`
- 初始 commit:待 Phase 1 close(本脚手架 commit)

## 3. Equivalence model(boilerplate)

按 family 统一口径:每个公开 API 以 **parity**(new-vs-old 在真实数据上等价)+
**behavior**(new-only 正确性,monolith-free)验证;故意改了行为处记 **divergence**。
详见 `hjlibm/docs/hjlib_standard/migration_protocol.md`。本端口的 parity 口径 =
同 ckpt + 同 testset 下对齐 monolith 指标数字(Phase 6 / track 2)。

## 4. What was ported(file-mapping 计划)

| monolith 源 | → dest | 状态 | 备注 |
|---|---|---|---|
| `test_segment.py` | `test_segment.py` | ✅ Phase 2 | Test_Segment(verbatim) |
| `testset.py` | `testset.py` | ✅ Phase 2 | TestSet + Filter_Stats;divider = assembly `Filtered_Sub_Seq_Divider`(import);+ build recipe 字段(path_root_label/fps);Filter_Stats provenance → None(DIV-4) |
| `testset_builder_base.py` | `testset_builder_base.py` | ✅ Phase 2 | ABC;路径注入式(无 local_setting) |
| `per_dataset/testset_builder_{wp,jta}.py` | `testset_builder.py`(单一) | ✅ Phase 2 | **合并**三个近重复 builder 为一个泛型 `TestSet_Builder`(DIV-3) |
| `get_by_dataset.py`(builder 半) | `get_by_dataset.py` | ✅ Phase 2 | `get_testset_builder`(注入 dump/filter roots;wp filter token quirk 保留);gt_provider 半 → Phase 3 |
| `assembly_factory.py` | `assembly_factory.py` | ✅ Phase 2 | `build_test_assembly` 经 assembly 工厂 `get_dataset_seq_assembly(..., divider=)` 注入(DIV-1);返回 Dataset(非 `(ds_ss, assembly)` tuple,DIV-5) |
| `gt_provider_base.py` | `gt_provider_base.py` | ✅ Phase 3 | ABC;**scoped 指标 baseline**:joints/param/eval_meta abstract;camera/ground/video deferred(base 默认 raise,DIV-8) |
| `per_dataset/gt_provider_{wp,jta}.py` + `{wp,jta}_eval_meta.py` | `per_dataset/...` | ✅ Phase 3 | WP(从 assembly dump full label)/ JTA+JTA_Ext(从 dataset-std raw 22-joint,22→54 limb remap;param raise);per-dataset Eval_Meta |
| `eval_meta.py` | `eval_meta.py` | ✅ Phase 3 | Eval_Meta / Metric_Spec_3D / Metric_Spec_2D_OKS(verbatim;gt_provider 依赖,提前到 Phase 3) |
| `network_driver_base.py` | `network_driver_base.py` | ✅ Phase 4 | ABC(`infer(dict_item)->dict`);**live driver deferred**(eval-on-dumps 不需要,见 DIV-10) |
| `tester.py` | `tester.py` | ✅ Phase 5 | Tester:stage_eval(核心)+ stage_list_segments(joints-only GT pull)+ stage_inference(需 live driver);stage_vis 不迁(vis out of scope,DIV-12) |
| `eval_reducer.py` | `eval_reducer.py` | ✅ Phase 5 | `eval_dumps_against_gt`(MPJPE/T-MPJPE/Jitter 归约,verbatim 数学);`compute_jitter` inline(原 monolith util_metric);fps 走 assembly registry |
| (新增) | `dump_reader.py` | ✅ Phase 5 | qualname-routing unpickler(读 monolith 真 dump 不 import monolith;assembly legacy_pickle 模式),DIV-9 |

## 5. What was NOT ported

| 源 | 原因 |
|---|---|
| filter 生产(monolith `seq_modification.py` 等 415 行) | **复用 assembly 已 washed 的 `Filter_Modifications_Store` v1**,不重港生产(设计 SSOT §6 track 2 明确) |
| `per_dataset/{testset_builder,gt_provider}_vrv1.py` + `vrv1_eval_meta.py` | **先占位/跳过**:assembly 当前 vrv1 out-of-scope(washed 跳过),待 assembly 支持后再纳入(用户明确) |
| `script/test/test_protocol_dynamic/run_test.py` + `dispatch_*.sh` + `network_driver_ief_legacy.py` | entry / 实验编排 / 具体 IEF driver 归 **hjlib-experiments**(app 层),不进本执行层仓;本仓只留 `Network_Driver_Base` ABC + 通用 driver |

## 6. Intentional API divergences from monolith

| id | 改动 | why |
|---|---|---|
| DIV-1 | `build_test_assembly` 经 assembly 工厂 `get_dataset_seq_assembly(..., divider=testset.divider)` 构 Dataset,不再 monolith 式手建 `Dataset_Single_Seq`+`Seq_Label_Manager`+encoder+Dataset。**divider 注入而非 `Assembly_Config_Filtered_Seq`**:eval 的 testset 必须把 divider 与 scene-level Test_Segment 在同一循环里造(GT 查找按 flat index 对齐),且 `restrict_to_scenes` + 策展策略要能传到 Dataset——这些 by-parameter config 路径表达不了,故注入预建 divider(见 DIV-6 的 assembly 增项) | 工厂封装 label-manager/encoder/Dataset 接线 + 对齐安全 + 保 restrict_to_scenes |
| DIV-3 | monolith 的两个 builder 文件(`per_dataset/testset_builder_wp.py` + `testset_builder_jta.py`,后者含 `TestSet_Builder_JTA` + `TestSet_Builder_JTA_Ext` 两类)共三个 builder 类——仅差 name_dataset + filter-dir token + 一行 meta——合并为单一泛型 `TestSet_Builder`,per-dataset 事实在 `get_by_dataset` 注入 | 去重;vrv1 将来若需特例再单开(现 deferred) |
| DIV-4 | `Filter_Stats` provenance 字段(bias_config_name / bias_tag / min_bias_segment_length / n_frame_min_seq / produced_at)→ `None` | 一次性洗盘进版本化 store 时丢了 monolith `_meta.json` sidecar(store 版本 metadata 只有 created/note/producer/source);count 字段照算 |
| DIV-5 | `build_test_assembly` 返回单个 `Dataset_Single_Seq_Assembly`(monolith 返回 `(ds_ss, assembly)` tuple) | 新工厂内部持 label-manager;`Dataset_Single_Seq` 磁盘管理器不再外露 |
| DIV-6(跨仓,assembly 增项) | 给 assembly `get_dataset_seq_assembly` 加 **additive** `divider: Optional[Seq_Divider] = None` 注入参(默认 None = 原行为,现有 caller + `Assembly_Config_Filtered_Seq` smoke 零影响)+ 给 assembly 包加 `py.typed`(其本就 strict-typed,eval 是首个要其类型的 typed 下游) | DIV-1 需要;py.typed 让 eval strict pyright 拿到真类型(非 `reportMissingTypeStubs:none` workaround);均未 commit,随 assembly 同批提交(见 per_lib EVAL 节 + pyproject pin 注) |
| DIV-7(Phase 3,跨仓 dep) | GT raw 源:WP joints/param 从 **assembly dump full label**(原 monolith 用 `WorldPose_Official` raw reader);JTA/JTA_Ext raw 22-joint 从 **hjlib-dataset-std**(`get_jta_std(label='joints')`,原用 `JTA_Raw_Label`)。evaluation **新增 deps = hjlib-dataset-std + hjlib-skeleton**(用户 2026-06-24 确认;设计 SSOT dep 图相应更新) | 家族化 raw 数据访问;WP 无需 raw reader(dump 已带 GT) |
| DIV-8(Phase 3,scope) | GT_Provider 的 `get_camera_K_RT_gt` / `get_ground_param_world` / `get_scene_video_streamer` 改为 base 默认 **raise NotImplementedError**(deferred),而非 per-dataset 实装 | 用户选「聚焦指标 baseline」:这三者服务 2D OKS(stage_eval 未实现)+ reprojection vis(out of scope);待 2D OKS/vis 取用时经 dataset-std 接 |
| DIV-9(Phase 5) | 新增 `dump_reader.load_inference_dump`:custom `Unpickler.find_class` 把(monolith 或 hjlib)`Test_Segment` qualname 路由到本仓类,numpy/builtins 用真 reducer。**+ legacy 名归一(Phase 6 fix,见 §7 FIX-1)**:把 monolith dump 的 bare `name_dataset`(`worldpose`/`jta`/`jta_ext`)硬映射到具体 canonical 变体(`worldpose_smpl`/`jta_smpl_fitted`/`jta_ext_smpl_fitted`)。 | monolith 真 dump pickle 了 monolith `Test_Segment` qualname,直接 unpickle 会拉 monolith 重栈(network/yacs);路由后读 dump 零 monolith import(assembly legacy_pickle 模式)。归一是因 assembly registry 把 bare 名 deprecate 为 canonical 变体名,而 frozen dump 仍是 bare —— reducer 的 `seg==seg_on_disk` 全等会因 name_dataset 不符而炸 |
| DIV-10(Phase 4) | **live network_driver deferred**:迁移 + parity 走 monolith 真 inference dump(`stage_eval` 读 dump,与 inference 解耦),不需 live 网络;live driver(载 ckpt + dict_batch 适配)仅用于评新 ckpt,后置 | 用户 2026-06-24:用真实、和 monolith 一致的 dump 反而更顺畅 + 直接 parity-comparable;省去 ckpt/feature-cache/dict_batch 关键路径 |
| DIV-11(Phase 5) | `compute_jitter` inline 进 eval_reducer(原 monolith `evaluate.compute_metric.util_metric.compute_JITTER`);fps 走 `get_dataset_facts(name).fps`(原 `get_fps_by_dataset_name`) | 避免港整个 monolith metric util;fps 已在 assembly registry |
| DIV-12(Phase 5) | `tester.stage_vis` 不迁;`stage_list_segments` 的 GT pull 仅 joints(原含 camera K/RT) | vis out of scope;camera 在 deferred GT 面(DIV-8) |
| DIV-2(候选,live driver) | network_driver 适配 `dict_batch`(参考 experiments adapter),而非 monolith 的 `Single_Seq_Sample_Batch` 直喂 | hjlib-network 对外契约仅 `dict_batch` |

> 上表在后续(live driver / 2D OKS / vis)落地时继续补 / 修订。

## 7. Bug fixes during the port

| id | 位置 | 症状 | 修复 |
|---|---|---|---|
| FIX-1(Phase 6,parity 时发现) | `dump_reader.load_inference_dump` | eval-on-dumps(含本仓 `test/test_eval_on_dumps_with_data.py` + parity)在 reducer `_load_one_segment` 的 `assert seg_on_disk == seg` 处炸:monolith dump 的 `name_dataset='worldpose'`(旧 bare 名),新 testset 盖的是 canonical `'worldpose_smpl'`(因 assembly registry 2026-06-24 把 bare 名 deprecate),frozen dataclass 全等含 `name_dataset` 故不符。只 `name_dataset` 一个字段不同,scene/seq/person/frames 全一致;`build_segment_tag`(dump 文件名)本不含 `name_dataset`,故文件名对得上、相等判断却更严。 | `dump_reader` 读 dump 时把 bare `name_dataset` 硬映射到具体 canonical 变体(`_LEGACY_DUMP_DATASET_TO_CANONICAL`),**不丢弃**该字段(它标识 fit 变体,reducer 仍据此守卫「dump 是否对的变体」)。new lib 自产 dump 已是 canonical 名,原样通过。用户拍板:用硬映射而非放宽全等(name_dataset 有语义:指明 dump 是哪个变体)。 |

## 8. Where verification lives

`hjlib-migration-tests/evaluation/`(新 Claude 会话写 parity + behavior,见
migration_protocol.md「外部 migration test」)。**不在本仓**(本仓不依赖 monolith)。

## Migration test status

> 术语提醒:本节用 `migration_protocol.md` 的**通用 Phase 1/2/3 lifecycle**(1=code
> in new lib / 2=parity+behavior / 3=absorbed),与本仓 docs 别处的**自定义 Phase 1-6
> 工作阶段编号**(1 脚手架…5 tester+eval / **6=parity**)是两套编号。映射:本仓自定义
> 「Phase 6 parity」== 协议「Phase 2」;本仓自定义 Phase 1-5(端口)全部落在协议「Phase 1」内。

- [~] Phase 1 — code lives in new lib; pyright strict 0 + smoke + data-dependent
      tests green。**仅差 initial commit**(尚未 git init / commit,push 待用户授权);
      代码 + 测试侧已就绪,checkbox 仅卡在 commit。
- [x] Phase 2 — parity + behavior tests green in
      `hjlib-migration-tests/evaluation/`; `(monolith_sha, new_lib_sha)`
      pair pinned in that subdir's README.md;
      `grep -rn 'lib_dynamic_hvip' .../evaluation/behavior/` returns empty.
      **Verified 2026-06-25 at `(monolith 2bc42db4, hjlib-evaluation 3753ad8)`** —— 15 tests:
      capstone end-to-end(wp×2 + jta×2)+ enumeration/GT(含 jta_ext)+ behavior(8);pyright
      standard 0;grep 空。monolith working tree 带 uncommitted **vrv1-only** 改(不在 parity
      路径上,故 pin 仍准——详见 migration-tests `evaluation/README.md` SHA pair 注)。OLD 侧
      经 `py312th280cu128` subprocess 跑活的。
- [ ] Phase 3 — absorbed; `behavior/` + per-lib `conftest.py` +
      `local_setting_test.py` + readers moved into
      `hjlib-evaluation/test/`; `parity/` and `divergence/` deleted;
      `hjlib-migration-tests/evaluation/` subdir removed
      (absorb date: YYYY-MM-DD)
