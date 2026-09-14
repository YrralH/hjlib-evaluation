# Compute unreduced joint errors

Use the package-root functions with equal arrays ending in `(J, 3)`:

```python
import numpy as np

from hjlib_evaluation import (
    compute_joint_height_errors,
    compute_pa_joint_position_errors,
    compute_joint_position_errors,
)

joint_axis = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
reference_pose = np.stack(
    (joint_axis, joint_axis ** 2, joint_axis ** 3), axis=-1)
reference = np.broadcast_to(reference_pose, (120, 12, 3)).copy()
predicted = 1.5 * reference + np.array([1.0, -2.0, 0.5])
ground_normal = np.array([0.0, 0.0, 2.0])

error_3d = compute_joint_position_errors(predicted, reference)
error_height = compute_joint_height_errors(
    predicted, reference, ground_normal)
error_pa = compute_pa_joint_position_errors(predicted, reference)
```

All three outputs have shape `(120, 12)` and dtype `float64`. The height function
normalizes `ground_normal`; callers must still decide joint selection, units,
validity policy, and occurrence reduction.

## Signatures

```python
compute_joint_position_errors(target_points, reference_points) -> NDArray[np.float64]
compute_pa_joint_position_errors(target_points, reference_points) -> NDArray[np.float64]
compute_joint_height_errors(target_points, reference_points, ground_normal) -> NDArray[np.float64]
```

## Which function to use

| Need | Function |
|---|---|
| Full 3D Euclidean error per joint | `compute_joint_position_errors` |
| Per-occurrence positive-scale proper-similarity error | `compute_pa_joint_position_errors` |
| Absolute error along the ground-normal direction | `compute_joint_height_errors` |
| Joint selection, unit conversion or aggregation | Keep these in the caller |
