# Selected Corrected-Crowd Population

Use this additive surface when a comparison needs an explicitly named subset
of the existing corrected crowd sequence. It does not alter the legacy
`GT_VISIBLE` or `C4D_DYCROWD_COMMON` schemas/results.

The Campaign 04 default is
`C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9`: start from the frozen old-common mask,
map the source visibility channel to COCO-17, and retain an occurrence when at
least 9 channels are greater than zero. Author-relayed `0.5` self-occlusion
values therefore count as visible for this selection.

```python
from hjlib_evaluation import (
    C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
    evaluate_corrected_crowd_selected_view,
    make_coco17_visible_ge9_common_mask,
    reduce_corrected_crowd_selected_view_summaries,
)

selected = make_coco17_visible_ge9_common_mask(
    sequence.gt_visibility_native,
    old_common_mask,
)
summary = evaluate_corrected_crowd_selected_view(
    sequence,
    C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
    selected,
)
result = reduce_corrected_crowd_selected_view_summaries([summary])
```

When the same selected population also needs GT-relative world dynamics, use
the validate-once combined entry and reduce the additive summaries separately:

```python
from hjlib_evaluation import (
    evaluate_corrected_crowd_selected_view_and_world_dynamics,
    reduce_corrected_crowd_world_dynamics_summaries,
)

summary, dynamics_summary = (
    evaluate_corrected_crowd_selected_view_and_world_dynamics(
        sequence,
        C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
        selected,
    )
)
dynamics = reduce_corrected_crowd_world_dynamics_summaries([
    dynamics_summary,
])
```

Call the standalone entry when the legacy 15-metric result is not needed:

```python
from hjlib_evaluation import evaluate_corrected_crowd_world_dynamics

dynamics_summary = evaluate_corrected_crowd_world_dynamics(
    sequence,
    C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9,
    selected,
)
```

The additive metric order is `ACC-JOINT`, `ACC-ROOT`, `JERK-JOINT`,
`JERK-ROOT`. All use unaligned SMPL-24 world coordinates and GT derivative
residuals. Root means SMPL pelvis joint 0. Acceleration requires exact matched
selected triples and reports `mm/frame^2`; jerk requires exact quadruples and
reports `mm/frame^3`. `ACC-JOINT` is exactly the legacy `ACCEL-WORLD` value.
Prediction-only jitter magnitude is not part of this result.

The selection function requires an explicit old-common mask and validates all
shapes. It is a named comparison profile, not a registry, auto-detection rule,
or safety gate. The evaluator preserves the established metric order, units,
penalties, temporal rules, and lexical scene reduction. Selected summaries and
results use their own versioned JSON schema so legacy schema-v1 artifacts remain
immutable.

For the accepted VirtualCrowd release, the frozen common set contains 167,243
occurrences and this profile contains exactly 159,405. A producer remains
responsible for constructing `Corrected_Crowd_Sequence` and for binding its
native outputs to source identities; `hjlib-evaluation` neither reads
TPA-private results nor starts a method runner.
