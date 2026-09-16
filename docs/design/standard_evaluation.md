# Standard evaluation 三层 scope

## Contract

`standard_evaluation_scope.py` 对一个 immutable `Corrected_Crowd_Sequence`
建立一次 direct-target row join，并共享给 TRACK、VISRUN、FRAME 三类 partition。
standard join 只复制 row index，不复制 joints、COCO17 或 bbox tensor。VISRUN 边界必须由
caller 传入 protocol-owned `base_visrun_labels`；selection hole 不切断 alignment VISRUN，
但会在同一个 VISRUN 内切断 acceleration sub-run。
Index 只保留 joined-row orders、half-open offsets 与 joined VISRUN labels，不复制
metric tensor；公开构造会从 join/labels 重推 canonical partitions，拒绝用合法 FRAME
partition 冒充 TRACK 等语义替换。

`standard_evaluation.py` 只有三个顶层 metric loop：

```text
for TRACK:  SEQ-T/SEQ-RT on TRACK
for VISRUN: SEQ-T/SEQ-RT on VISRUN + exact-subrun ACC
for FRAME:  occurrence geometry/OKS + cross-person PPDS
```

每个 `add_standard_*_metrics` 只调 `standard_evaluation_metrics.py` 中的独立 pure
leaf。leaf 只接收已经切好的 metric-native tensor，不能观察 sequence、index、frame ID、
track ID 或 scope identity，因此不能重建遍历。

## Profile 与 reduction

`LSVHR_STANDARD_METRICS_V1` 固定 14 项：MPJPE-WORLD、T-MPJPE、RT-MPJPE、
两个 VISRUN sequence alignment、两个 TRACK sequence alignment、RTE-WORLD、
ACC-ROOT、ACC-ROOT-RATIO、ACC-JOINT-RATIO、OKS-VIS、PPDS、PA-PPDS。

所有普通 metric 保存 primitive sum/count 并做跨 scene micro reduction。两个 ACC ratio
保存 numerator/denominator 到最终 reduction 才相除。`ACC-JOINT-RATIO` 先为每个 SMPL24
joint 计算全局 ratio，再对 24 个 ratio 等权平均；joint 0 的充分统计必须精确等于
`ACC-ROOT-RATIO`。GT denominator 必须逐 joint 严格大于零，不加 epsilon、不跳 joint。

scene-local projection 允许某项 support 为零，并把该 cell 写成 `null`；这不会提前否定
跨 scene reduction。最终 overall result 才要求 14 项全局 support 非空以及全部 ACC GT
denominator 为正。

## Extension

新增 metric 时先决定它属于 TRACK、VISRUN 或 FRAME，新增一个只接收 sliced tensor 的
leaf，再由对应 add 函数登记 contribution。不要新增第四次 traversal，也不要在 leaf 内按
identity group。修改 VISRUN population 时必须由 protocol caller 改 label construction，不能从
selected rows 推断完整 base run。

本次 task 的完整数学、性能门槛与 migration 记录见
[task design](tasks/lsvhr-standard-evaluation/README.md)。
