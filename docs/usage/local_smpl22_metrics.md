# Local SMPL22 joint errors

Use these pure leaves for pelvis-centered T-MPJPE and rigid RT-MPJPE over SMPL
joints0..21. Inputs are world-metre joint arrays `(N,J>=22,3)` with equal positive
frame counts. Extra joints are ignored before alignment and finite checks;
the selected22 points must all be finite. RT permits planar support and rejects
degenerate point sets or cross-covariance.

```python
from hjlib_evaluation import (
    compute_local_smpl22_rt_mpjpe_values,
    compute_local_smpl22_t_mpjpe_values,
)

t_errors_m = compute_local_smpl22_t_mpjpe_values(predicted_joints, reference_joints)
rt_errors_m = compute_local_smpl22_rt_mpjpe_values(predicted_joints, reference_joints)
t_mpjpe_mm = 1000.0 * float(t_errors_m.sum()) / t_errors_m.size
rt_mpjpe_mm = 1000.0 * float(rt_errors_m.sum()) / rt_errors_m.size
```

Each result is a read-only float64 `(N,22)` array. Wrists20/21 and pelvis0
remain in the population; hands22/23 cannot influence the fit. T independently
subtracts pelvis0 from each frame. RT independently fits a proper rotation and
translation over all22 joints in each frame, without scale or reflection.

For multiple scenes, accumulate error sums and joint counts, then divide once.
Do not average scene means. The caller validates exact population coverage and
owns parameter interpretation, GT source selection and method scheduling.
Scoring local joints does not establish lookahead compliance or a complete
world-SMPL Output. See the [metric design](../design/local_smpl22_metrics.md)
for numerical rank tolerances.
