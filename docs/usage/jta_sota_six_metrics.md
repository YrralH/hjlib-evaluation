# JTA fitted-SMPL six metrics

For one paired batch, compute sufficient statistics and finalize them:

```python
from hjlib_evaluation import (
    compute_jta_sota_metric_sums,
    finalize_jta_sota_metric_sums,
)

sums = compute_jta_sota_metric_sums(
    pred_joints,
    gt_joints,
    pred_occurrence_ids,
    gt_occurrence_ids,
    pred_endpoint_xy,
    gt_endpoint_xy,
    gt_smpl24_xy,
    joint_layout='smpl_24',
    unit_world='m',
    pred_coordinate_frame='camera',
    gt_coordinate_frame='camera',
    image_coordinate_frame='jta_original_pixels',
)
result = finalize_jta_sota_metric_sums(sums)
```

For multiple batches, call `.plus(...)` before finalization. Before accepting a
split, pass its expected int64 occurrence IDs and the ordered per-batch ID
arrays to `validate_jta_sota_occurrence_partition(...)`.

| Input situation | Use |
| --- | --- |
| Paired fitted-SMPL identities | `compute_jta_sota_metric_sums(...)` |
| Several batches or workers | `JTA_SOTA_Metric_Sums.plus(...)` |
| Need reported means | `finalize_jta_sota_metric_sums(...)` |
| Need exact population coverage | `validate_jta_sota_occurrence_partition(...)` |

Inputs must be finite float32/float64 arrays. The image-space arguments use the
declared full-image coordinate frame, and `gt_smpl24_xy` defines the per-person
OKS area. This API does not match detections to GT identities.
