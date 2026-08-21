# 用 `Tracked_Scene` 评估地面估计

已有带 keypoint 的 `Tracked_Scene` 时，先收集满足 caller 协议的 person-frame。
下面是历史 confidence>3 + fixed-count baseline：

```python
from hjlib_evaluation import (
    collect_ground_observations,
    estimate_ground_from_observations,
    sample_ground_observations,
)

candidates = collect_ground_observations(
    tracked_scene,
    top_joint_pair=(5, 6),
    bottom_joint_pair=(15, 16),
    confidence_threshold=3.0,
)
selected = sample_ground_observations(candidates, max_count=5_000, seed=17)
result = estimate_ground_from_observations(selected, K)
```

`estimate_ground_from_observations` 默认调用 `hjlib-ground-solver` 的 RCR
top/bottom solver。返回的 `plane_camera_abcd` 是 camera-frame unit-normal
plane，`objective` 是 solver 的 dimensionless fitting objective。

若已有共同的 GT-ground ray support，可调用
`compute_same_ray_ground_errors(support, K, plane)` 得到每个 person-frame 的
3D Euclidean error，单位与 support 的 3D 坐标相同；本仓 VirtualCrowd
协议使用 metre。再用 `summarize_ground_errors` 做 person-frame-micro 归约。
函数要求全部 ray 都有 finite positive forward intersection，不会静默丢 row。

若要保留全部严格筛选 observations 并比较 density weights，可让 collection 同时
检查 bottom-pair/bbox-width ratio，再用 `hjlib-ground-solver` 的
`compute_ground_observation_kde_density` 构造 automatic-bandwidth weights。
VirtualCrowd current single-arm run 使用：

- `min(score[5], score[6], score[15], score[16]) > 4.0`；
- `norm(kp[15]-kp[16]) / bbox_width < 0.20`；
- 全部 17,992 observations，不 sampling；
- unweighted 与 exact-LOO Gaussian KDE + Scott bandwidth 两个版本。

VirtualCrowd 的完整实验入口是：

```bash
python ./script/evaluate_virtualcrowd_rcr_ground.py \
  --path-dataset-root <VirtualCrowd-root> \
  --path-tracked-scene-root <tracked-scene-root> \
  --path-ground-effect-support-root <plain-ground-effect-root> \
  --path-output-root <new-output-root>
```

Density-balanced 完整入口使用相同四个 roots：

```bash
python ./script/evaluate_virtualcrowd_density_balanced_rcr_ground.py \
  --path-dataset-root <VirtualCrowd-root> \
  --path-tracked-scene-root <tracked-scene-root> \
  --path-ground-effect-support-root <plain-ground-effect-root> \
  --path-output-root <new-output-root>
```

这两个入口都是从 repo root 执行的 dataset operation scripts，不是安装后的
`[project.scripts]` console commands。

在启动多 arm 求解前，可只读取 dataset 与 `Tracked_Scene` 来展开并核对
Cartesian matrix：

```bash
python ./script/evaluate_virtualcrowd_density_balanced_rcr_cartesian.py \
  --path-dataset-root <VirtualCrowd-root> \
  --path-tracked-scene-root <tracked-scene-root>
```

默认是 prepare-only：不会读取 ground-effect support、调用 RCR/KDE，或创建 output
root。当前 matrix 是 `confidence>{4.0,4.5,5.0}` ×
`ankle/bbox ratio<{0.15,0.20}` ×
`{filtered_unweighted,density_kde_scott_loo}`，即 6 个 observation populations、
12 个 configs。真实 dry-run population totals 为：

| confidence | ankle ratio | observations | smallest scene | configs |
|---:|---:|---:|---:|---:|
| `>4.0` | `<0.15` | 9,427 | 441 | 2 |
| `>4.0` | `<0.20` | 17,992 | 885 | 2 |
| `>4.5` | `<0.15` | 6,770 | 285 | 2 |
| `>4.5` | `<0.20` | 13,524 | 613 | 2 |
| `>5.0` | `<0.15` | 3,839 | 87 | 2 |
| `>5.0` | `<0.20` | 8,326 | 256 | 2 |

若以每 scene 约 800 observations 为经验下限，只有
`confidence>4.0, ankle/bbox<0.20` 全部 scenes 达标。执行时需显式增加
`--execute`、ground-effect support root 和一个不存在的 output root。

已完成的 12-arm global person-frame-micro same-ray means 为：

| confidence | ankle ratio | unweighted (m) | KDE (m) | KDE change | improved scenes |
|---:|---:|---:|---:|---:|---:|
| `>4.0` | `<0.15` | 21.5923 | 18.2115 | -15.66% | 5/8 |
| `>4.0` | `<0.20` | 18.8961 | 16.4576 | -12.90% | 5/8 |
| `>4.5` | `<0.15` | 25.8974 | 21.4003 | -17.37% | 6/8 |
| `>4.5` | `<0.20` | 17.4740 | 15.8484 | -9.30% | 4/8 |
| `>5.0` | `<0.15` | 21.8608 | 18.3688 | -15.97% | 7/8 |
| `>5.0` | `<0.20` | 17.1769 | 15.7277 | -8.44% | 5/8 |

`baseline001` 固定指代 `confidence>5.0, ankle/bbox<0.20, KDE`，其 global
mean 为 `15.727720 m`。该 population 的最小 scene 只有 256 observations。六组
KDE 的 combined mean 均改善，而
support-weighted normal-oracle 与 distance-only means 均变差；因此 KDE gain
包含系统性的 error cancellation，不应解释为两个 ground components 都更准。

当前 KDE single-arm 的 global mean same-ray error 为：unweighted `18.896103 m`、
KDE `16.457570 m`，即降低 `2.438532 m`（12.90%）。8 个 scenes 中 5 个改善、3 个
变差；global normal-oracle mean 从 `8.375987 m` 变为 `9.123947 m`，distance-only
mean 从 `21.030782 m` 变为 `25.014907 m`，所以 combined gain 含 error
cancellation，不能解释为 plane geometry 全面改善。历史 fixed-kNN 探索的 k=16/32/64
means 为 `15.384228/15.590497/15.900516 m`，均不是当前 automatic KDE protocol。

| 我有的数据 / 目标 | 调用 |
|---|---|
| 完整 `Tracked_Scene`，需要候选 observations | `collect_ground_observations` |
| 需要 global frame 0 diagnostic | `select_ground_observations_at_frame` |
| 需要固定 seed、至多 N 个 person-frame | `sample_ground_observations` |
| 已选 observations + `K`，需要估计 plane | `estimate_ground_from_observations` |
| 已有 GT ray support，需同射线地面误差 | `compute_same_ray_ground_errors` + `summarize_ground_errors` |
| 需要 normal / D 独立诊断 | `compute_ground_plane_diagnostics` + `compute_ground_effect_decomposition` |
| 需要完整 strict-filter density variants | `evaluate_virtualcrowd_density_balanced_rcr_ground.py` |
| 需要展开并核对 confidence × ankle ratio × density configs | `evaluate_virtualcrowd_density_balanced_rcr_cartesian.py`（默认 prepare-only） |
