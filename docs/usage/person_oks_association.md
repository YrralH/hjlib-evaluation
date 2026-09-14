# Per-frame person OKS association

`Person_OKS_Association` represents a complete, disjoint matched/FN/FP
partition. `match_coco17_people(...)` is the single public dispatcher for two
named COCO17 profiles:

- `standard-matching-v1` (the default): visibility-aware OKS, inclusive `0.5`
  threshold, maximum cardinality, maximum quantized total OKS, deterministic
  fixed-order SciPy Hungarian tie resolution. A GT row without any visible
  COCO17 joint remains in the GT partition as a forced FN and is omitted from
  the solver, so it cannot perturb supported-row tie resolution.
- `crowd4d-author-greedy-v1`: all-joint author area policy,
  strict `1e-6` threshold, first-best greedy selection and collision drop with
  no second-choice retry.

Both profiles take `(G,17,3)` GT xy/visibility, `(P,17,3)` prediction joints
in camera coordinates, `(G,4)` GT xyxy boxes, and a `3x3` camera matrix. A
prediction is projection-valid only when all 17 depths are positive. Invalid
predictions remain FP. The compatibility profile is per-frame association
only; it does not include Crowd4D temporal repair or metric reduction.

The aligned `match_gt_indices`, `match_prediction_indices` and `matched_oks`
arrays have profile-defined order. Standard matching emits ascending GT-row
order. Crowd4D-author matching emits the supplied evaluator's best-distance
process order, including collision drops; FN and FP arrays remain ascending
complements. This order is preserved by result serialization and paired-color
visualization. Public association construction accepts integer index arrays
and a boolean projection-valid mask only; fractional indices and non-boolean
masks are rejected rather than coerced during immutable normalization.

Crowd4D author fidelity applies on the projection-valid domain. The supplied
evaluator consumes 2D joints directly, whereas this API deliberately leaves a
nonpositive-depth camera-space prediction unmatched. Equal best distances
inherit the supplied runtime's default `np.argsort` order; their relative order
is not promised across NumPy versions.

```python
from hjlib_evaluation import (
    CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
    match_coco17_people,
)

association = match_coco17_people(
    gt_coco17_xyv,
    prediction_coco17_camera,
    gt_bbox_xyxy,
    camera_K,
)

author_compatible = match_coco17_people(
    gt_coco17_xyv,
    prediction_coco17_camera,
    gt_bbox_xyxy,
    camera_K,
    CROWD4D_AUTHOR_GREEDY_MATCHING_PROFILE,
)
```

The standard profile performs one Hungarian solve for nonempty inputs. The
author-compatible profile performs no Hungarian solve. At approximately 200
GT people and 200 predictions, pairwise OKS construction dominates runtime;
no matcher cache or stateful object is required.

JTA's older detector protocol shares only the cardinality/quality objective
solver. It intentionally retains a different explicit row-lex third objective
and therefore does not call this dispatcher.

The cross-repository ownership and formal objective are recorded in the
[Crowd3D matching task design](../../../hjlib-experiments-results/docs/design/tasks/crowd3d_matching/README.md).
