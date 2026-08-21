# VirtualCrowd Evaluation Profiles

## Requirements

Define the stable names needed to answer one practical question without relying
on conversational context:

> When a result says that a method was tested on VirtualCrowd, which population,
> association policy, metrics, units, and reduction semantics were used?

The profile vocabulary is owned by `hjlib-evaluation`. Dataset readers continue
to own dataset facts, while method-specific request adapters, runtimes, and
native-result interpreters remain in their `hj-tpa-*` repositories. In
particular, the Crowd4D fidelity adapter/runtime remains owned by
`hj-tpa-crowd4d`; `hjlib-evaluation` must not import a TPA-private schema or
start its runner.

This task records two accepted metric profiles and the two complete evaluation
profiles that compose them. It does not yet authorize a Python registry or
change an existing result schema.

### Naming layers

The following nouns are deliberately separate:

- a **population profile** decides which GT occurrences are eligible;
- an **association profile** decides which prediction is paired with which GT
  identity;
- a **metric profile** fixes the ordered metrics, their mathematical semantics,
  units, penalties, and reductions;
- an **evaluation profile** composes dataset scope, population, association,
  and metric profiles.

A metric profile therefore never silently selects a visibility population or
an identity policy.

### Default phrase resolution

Within HJ documentation, the bare phrases “test on VirtualCrowd” and “在 VC
上测试” resolve to `VC_HJ_DEFAULT_V1` unless another evaluation profile is
named. Machine-readable results and receipts must still record the explicit
evaluation-profile identity; the prose default is not a substitute for
provenance.

“Crowd4D native” resolves only to `VC_CROWD4D_NATIVE_V1`. It does not mean the
HJ default with Crowd4D predictions.

## Profile Architecture

### Metric profile: `VC_HJ_DEFAULT_METRICS_V1`

This is the HJ default ordered metric set for VirtualCrowd. It contains 18
non-duplicate metrics:

```text
MPJPE-WORLD
T-MPJPE
RT-MPJPE
PA-MPJPE
SEQ-T-MPJPE-VISSEQ
SEQ-RT-MPJPE-VISSEQ
SEQ-PA-MPJPE-VISSEQ
SEQ-T-MPJPE-ID
SEQ-RT-MPJPE-ID
SEQ-PA-MPJPE-ID
PPDS
PA-PPDS
PCOD-3C-0.3m
OKS-VIS
ACC-JOINT
ACC-ROOT
JERK-JOINT
JERK-ROOT
```

Its current mathematical source is the accepted corrected-crowd selected-view
protocol plus the additive world-dynamics protocol. The profile adopts these
display-name migrations without changing the frozen schema-v1 artifacts:

| Frozen schema-v1 name | Default-profile name | Relation |
| --- | --- | --- |
| `SEQ-*-MPJPE-VISRUN` | `SEQ-*-MPJPE-VISSEQ` | Same maximal visible-run scope |
| `SEQ-*-MPJPE-TRACK` | `SEQ-*-MPJPE-ID` | Same GT identity across visibility gaps; samples remain eligible selected/matched rows |
| `ACCEL-WORLD` | `ACC-JOINT` | Mathematically and numerically identical |

`ACCEL-WORLD` is not displayed a second time. `ACC-JOINT`, `ACC-ROOT`,
`JERK-JOINT`, and `JERK-ROOT` use GT-relative world-space vector derivative
residuals on exact consecutive frames, in `mm/frame^2` or `mm/frame^3`.

The exact per-metric math, support, units, and reduction remain owned by:

- [corrected metric protocol](../virtualcrowd-corrected-metric-protocol/README.md);
- [selected population](../virtualcrowd-corrected-population-profile/README.md);
- [world dynamics](../corrected-crowd-world-dynamics/README.md).

### Metric profile: `VC_CROWD4D_NATIVE_METRICS_V1`

This profile is a fidelity name for the twelve ordered cells emitted by the
identified supplied Crowd4D evaluator:

```text
matched ratio
PPDS
PA-PPDS
PCOD
MPJPE
PA-MPJPE
WA-MPJPE
W-MPJPE
ACCEL
OKS
GMPJPE
Score
```

The names are not aliases for similarly named HJ metrics. Given the pairs,
repaired track samples, and eligible/missing population supplied by the native
association and population profiles, this metric profile retains the supplied
evaluator's formulas, penalty consumption, compact-time temporal behavior,
frame-weighted reduction, and four-decimal display semantics. Matching and
identity repair are not selected by the metric profile. Its evidence is the
machine-local package whose identity is frozen in the
[author evaluator logic analysis](../author-evaluator-logic-analysis/README.md);
official upstream provenance remains unverified.

Notable native metric semantics include:

- all-joint OKS without the HJ COCO-17 visibility selection;
- pelvis-relative, compact-sample `ACCEL`, which is not `ACC-JOINT`;
- native penalty and frame-weighted aggregation rules.

The native profile must not be approximated by renaming HJ outputs. A method is
eligible only when its TPA can provide every required native semantic without
guessing or fabricating fields.

### Complete evaluation profile: `VC_HJ_DEFAULT_V1`

| Component | Binding |
| --- | --- |
| Dataset scope | The accepted full VirtualCrowd release, all eight scene/view sequences |
| Association | `VC_GT_MOT_ASSOCIATION_V1`: direct static GT identity |
| Population | `C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9` |
| Visibility rule | At least 9 mapped COCO-17 source channels are `> 0`; `0.5` is valid |
| Metrics | `VC_HJ_DEFAULT_METRICS_V1` |

This is the default meaning of testing a method on VirtualCrowd in HJ work.
The selected population and matched prediction population must be reported
separately, and no frame gap is bridged for derivative metrics.

### Complete evaluation profile: `VC_CROWD4D_NATIVE_V1`

| Component | Binding |
| --- | --- |
| Dataset scope | All eight VirtualCrowd scene/view sequences passed explicitly |
| Association | `VC_CROWD4D_NATIVE_ASSOCIATION_V1`: supplied greedy OKS association plus Crowd4D temporal repair |
| Population | `VC_CROWD4D_NATIVE_POPULATION_V1`: supplied frame/GT population and missing-person rules |
| Metrics | `VC_CROWD4D_NATIVE_METRICS_V1` |

The explicit eight-scene binding is required because the supplied evaluator's
native default roster contains only `scene1`. This profile is an isolated
fidelity baseline, not the HJ default and not an endorsement of every native
edge behavior. The two native component IDs name only the verified behavior of
the frozen supplied evaluator; they do not introduce public Crowd4D artifact
types into `hjlib-evaluation`.

### Comparison and reporting rules

1. A table compares methods only within the same complete evaluation profile.
2. A result must record both `evaluation_profile` and `metric_profile`; the
   population and association identities must also remain recoverable from the
   result or its bound receipt.
3. A report using both profiles presents two separately labeled tables. Native
   and HJ cells with similar names are not merged.
4. Capability is profile-specific. A TPA may support `VC_HJ_DEFAULT_V1`,
   `VC_CROWD4D_NATIVE_V1`, both, or neither.
5. An interpreter emits the closest already-validated stable HJ contract. It
   must preserve unresolved scale, coordinate, or association semantics rather
   than inventing values to satisfy a profile.

## Code Architecture

No implementation is authorized in this documentation task. A later
coordinating task may add immutable profile descriptors or a registry in
`hjlib-evaluation`, but must preserve these boundaries:

- evaluation contracts and profile composition are owned here;
- TPA-private adapters, runtime bridges, and native interpreters remain in the
  corresponding `hj-tpa-*` repository;
- no common TPA base class, family registry, or family-wide private-result
  schema is implied;
- `hjlib-evaluation` never launches a TPA runner or parses a TPA-private
  artifact.

Existing corrected schema-v1 and Crowd4D fidelity artifacts remain immutable.
Adoption requires a new versioned output identity rather than in-place field
renaming.

## Smoke-Test Standard

Future implementation must at minimum prove:

1. both profile IDs resolve to exact ordered metric lists;
2. `VC_HJ_DEFAULT_V1` resolves to GT-MOT plus the GE9 population and all eight
   scenes;
3. the legacy-to-default display mapping is exact and does not duplicate
   `ACCEL-WORLD`/`ACC-JOINT`;
4. Crowd4D native `ACCEL` cannot be selected as `ACC-JOINT`;
5. result/receipt round trips preserve all four component identities;
6. a cross-profile reduction or table merge is rejected;
7. existing schema-v1 canonical bytes remain unchanged.

## Coordination Handoff

The next VirtualCrowd protocol coordinator should decide the implementation
task split, result-schema migration, profile registry surface, and which
existing method producers claim each capability. That work is intentionally
not pre-decided here.

## Modification History

- 2026-08-20: Recorded the user decision that VirtualCrowd needs at least two
  metric profiles: an HJ default and a Crowd4D-native fidelity profile. Named
  both metric profiles and their complete evaluation-profile compositions;
  froze the default phrase resolution and the no-cross-profile comparison rule.
- 2026-08-20: Consistency/contract and boundary/reuse reviews found no Critical
  issues and three overlapping Concerns. Removed association selection from the
  native metric profile, narrowed the `TRACK` to `ID` mapping to eligible rows
  across visibility gaps, and assigned stable IDs to both native components and
  the HJ GT-MOT association. Both focused re-reviews returned zero remaining
  findings.
