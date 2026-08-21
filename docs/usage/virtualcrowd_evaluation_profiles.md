# VirtualCrowd Evaluation Profiles

Use an explicit profile ID in every result, command record, and table caption.

## Default

“Test on VirtualCrowd” means:

```text
VC_HJ_DEFAULT_V1
```

It composes:

- all eight VirtualCrowd scene/view sequences;
- `VC_GT_MOT_ASSOCIATION_V1`, using direct static GT identity;
- `C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9`, where `0.5` counts as visible;
- `VC_HJ_DEFAULT_METRICS_V1`, containing 18 non-duplicate corrected spatial,
  sequence, crowd, visibility, acceleration, and jerk metrics.

The default profile displays `VISSEQ`, `ID`, and `ACC-JOINT`. Existing
schema-v1 files may still contain their exact legacy names `VISRUN`, `TRACK`,
and `ACCEL-WORLD`; they are not rewritten in place.

## Crowd4D fidelity baseline

Use:

```text
VC_CROWD4D_NATIVE_V1
```

This selects `VC_CROWD4D_NATIVE_ASSOCIATION_V1`,
`VC_CROWD4D_NATIVE_POPULATION_V1`, and the supplied evaluator's native penalty,
temporal, reduction, and twelve-cell metric semantics. It is not the HJ
default, and similarly named native and HJ metrics are not interchangeable. All
eight scenes must be requested explicitly because the supplied native default
roster contains only `scene1`.

## Reporting

- Compare methods only when their complete evaluation-profile IDs match.
- If both profiles are reported, use two labeled tables.
- Record the evaluation, metric, population, and association identities in the
  result or its bound receipt.
- Do not claim native-profile support unless the method TPA can provide all
  required semantics without guessed fields.

The complete definitions and ordered metric lists are in the
[design residence](../design/tasks/virtualcrowd-evaluation-profiles/README.md).
