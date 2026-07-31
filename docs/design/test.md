# 测试布局 —— hjlib-evaluation

family 两棵树政策(`test_smoke/` 无外部数据 / `test/` 真实数据 FAIL-not-skip)的
本仓实例化。政策本身见 `hjlib_standard/test_layout.md`。

## test_smoke/ —— 合成数据,处处可跑

评测的纯逻辑件(`Eval_Meta` 契约、`TestSet` 容器不变量 + `restrict_to_scenes` 对齐、
工厂的路径解析 + 错误分支)用合成 numpy / 直接构造的 `Filtered_Sub_Seq_Divider` 覆盖,
不依赖真实数据集 / dump。

| 文件 | 覆盖 |
|---|---|
| `test_testset.py` | `Test_Segment`(length/to_str);`TestSet`(len / get_test_segment / restrict_to_scenes 对齐 / summary);`get_testset_builder` 路径解析(wp→`wp` token 等)+ vrv1/unknown/curated-policy 错误分支 |
| `test_gt.py` | `Eval_Meta`(WP 全 SMPL-24 / JTA 12·10 limb 两 variant);`get_gt_provider` 错误分支(vrv1 / unknown / jta 缺 raw root) |
| `test_eval_pred_field.py` | `eval_dumps_against_gt` / `Tester.stage_eval` 的 `pred_joints_key` 字段选择:默认 `joints_54_world` vs raw 诊断 `joints_54_world_raw` |
| `test_testset_fixed_window.py` | fixed-window 子集的窗口范围、segment offset、统计量与错误分支 |
| `test_trajectory_residual.py` | scalar residual summary、authoritative mask、overflow/feasibility、macro/micro reduction |
| `test_all_func.py` | master runner;导入并依次跑每个 `smoke_test_*` |
| `clean_test_data.py` | `LIST_PATH_CLEAN`(当前空:无持久产物) |

每个 case 文件同时暴露 `test_*`(pytest 发现)和一个 `smoke_test_<topic>()` 入口,
master runner 用 `sys.path.insert(0, dirname)` 导入。跑法:

```bash
python test_smoke/test_all_func.py      # 或
pytest test_smoke/ -q
```

## test/ —— 真实数据(FAIL-not-skip)

需真实 washed filter store + dumped labels(+ raw 数据集 + monolith dump)才有意义的
端到端验证。`test/local_setting_test.py.example` 是 tracked placeholder-only
字段契约;复制为 `test/local_setting_test.py`(gitignored)后只改本机路径。缺 runtime
则 import 期 AssertionError(FAIL-not-skip,不 skip)。

| 文件 | 覆盖 |
|---|---|
| `test_testset_with_data.py` | build worldpose/test + jta_ext/test;divider↔segment index 对齐 + seq-local→scene-level offset + assembly len;WP GT joints/param + jta_ext GT joints(12 limb 非 NaN)+ param raise |
| `test_eval_on_dumps_with_data.py` | 自动发现一个 worldpose/full 的 monolith 真 dump 目录,跑 `stage_eval`(内部断言 = pkl 存在 / segment 匹配 / shape / 指标 index 非 NaN);完成不 raise = 通过 |
| `test_all_func_with_data.py` | master runner;跑全部 `smoke_test_*_with_data` |
| `local_setting_test.py.example` | tracked contract:`PATH_DUMP_ROOT_HJLIB` / `PATH_FILTER_STATS_BASE` / `PATH_JTA_EXT_DATA_ROOT` / `PATH_INFERENCE_DUMPS_BASE` |
| `local_setting_test.py` | per-machine runtime(gitignored);真实路径与秘密只留本机 |

本仓 `test/` 的「reader」即 `get_testset_builder` / `get_gt_provider`(per-dataset 工厂),
不另设 `reader_<dataset>.py` Protocol —— 数据访问由工厂注入根路径完成。

跑法:`python test/test_all_func_with_data.py`(或 `pytest test/ -q`,需先配 local_setting_test)。

## parity 测试不在本仓

new-vs-monolith 的 **parity**(同 dump + 同 GT,跑 monolith `stage_eval` 对齐数字)落
`hjlib-migration-tests/evaluation/`(absorb-time 不变量,见 migration_protocol.md step 13-14,
fresh session);Phase 3 absorb 时其 `behavior/` 再并入本仓 `test/`。
