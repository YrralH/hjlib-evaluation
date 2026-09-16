# 评估 normalized LSV-HR scene

当 caller 已有 `Corrected_Crowd_Sequence`、GT selection 和与全部 GT rows 对齐的 base
VISRUN labels 时，直接调用：

```python
from hjlib_evaluation import (
    evaluate_standard_evaluation_scene,
    reduce_standard_evaluation_summaries,
)

scene_summary = evaluate_standard_evaluation_scene(
    sequence,
    filtering_id,
    split_id,
    selected_gt_mask,
    base_visrun_labels,
)
result = reduce_standard_evaluation_summaries([scene_summary])
print(result.metric('ACC-JOINT-RATIO'))
```

要独立 profile scope-index 与 metric 时间时，先构造 index，再调用 indexed 入口：

```python
from hjlib_evaluation import (
    build_standard_evaluation_scope_index,
    evaluate_standard_evaluation_scene_indexed,
)

index = build_standard_evaluation_scope_index(
    sequence,
    selected_gt_mask,
    base_visrun_labels,
)
scene_summary = evaluate_standard_evaluation_scene_indexed(
    index,
    filtering_id,
    split_id,
)
```

`base_visrun_labels` 不是 selected rows 上临时生成的 run：它必须先按完整 protocol-visible
domain 标记，再应用 selection。每个 selected row 必须有非负 label。最终 14 项均要求全局
support 非空；ACC ratio 的每个 GT joint denominator 都必须为正。

| 现有输入 | 入口 |
|---|---|
| sequence + selection + base labels | `evaluate_standard_evaluation_scene(...)` |
| 已预建 index / 需要分 phase profile | `evaluate_standard_evaluation_scene_indexed(...)` |
| 多 scene raw summaries | `reduce_standard_evaluation_summaries(...)` |
| persisted JSON | `standard_evaluation_*_to_json/from_json(...)` |
