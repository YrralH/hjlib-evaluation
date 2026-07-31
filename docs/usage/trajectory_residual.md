# Trajectory residual summaries

Use this API after the caller has constructed one scalar residual per frame and
decided the final valid population.

```python
import numpy as np

from hjlib_evaluation import (
    reduce_trajectory_residual_summaries,
    summarize_trajectory_residuals,
)


first = summarize_trajectory_residuals(
    np.array([0.1, 0.2, np.nan]),
    np.array([True, True, False]),
)
second = summarize_trajectory_residuals(
    np.array([0.4]),
    np.array([True]),
)
population = reduce_trajectory_residual_summaries([first, second])
```

The supplied boolean mask is authoritative. Masked-in residuals must be finite
and non-negative; masked-out placeholders may be non-finite. Quantiles use NumPy
linear interpolation.

The reduction exposes:

- `macro_mean` / `macro_mse`: every trajectory has equal weight;
- `micro_mean` / `micro_mse`: every valid frame has equal weight.

Pooled median and p95 are intentionally absent because a per-trajectory summary
does not retain the residual samples needed to reconstruct them.

This API is unit-neutral. A caller that supplies metre residuals may serialize
fields with metre-specific names in its own report schema.
