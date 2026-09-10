# Compute unreduced joint errors

Use the package-root functions with equal arrays ending in `(J, 3)`:

```python
import numpy as np

from hjlib_evaluation import (
    compute_joint_height_errors,
    compute_joint_position_errors,
)

predicted = np.zeros((120, 12, 3), dtype=np.float32)
reference = np.ones((120, 12, 3), dtype=np.float32)
ground_normal = np.array([0.0, 0.0, 2.0])

error_3d = compute_joint_position_errors(predicted, reference)
error_height = compute_joint_height_errors(
    predicted, reference, ground_normal)
```

Both outputs have shape `(120, 12)` and dtype `float64`. The height function
normalizes `ground_normal`; callers must still decide joint selection, units,
validity policy, and occurrence reduction.

## Signatures

```python
compute_joint_position_errors(target_points, reference_points) -> NDArray[np.float64]
compute_joint_height_errors(target_points, reference_points, ground_normal) -> NDArray[np.float64]
```

## Which function to use

| Need | Function |
|---|---|
| Full 3D Euclidean error per joint | `compute_joint_position_errors` |
| Absolute error along the ground-normal direction | `compute_joint_height_errors` |
| Joint selection, unit conversion or aggregation | Keep these in the caller |
