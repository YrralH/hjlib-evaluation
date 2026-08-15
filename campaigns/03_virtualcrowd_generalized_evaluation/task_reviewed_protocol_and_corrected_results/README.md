# Task - Reviewed Protocol And Corrected Results

## Purpose And Boundary

Define separately named HJ-reviewed result profiles after author parity. The
first authorized work package changes only identity pairing: both Crowd4D and
DyCrowd are treated as GT-MOT inputs, while the existing author metric,
penalty, temporal, aggregation, and display mathematics remain unchanged.
Preserve all T2 author-parity outputs and receipts unchanged.

This first package does not decide visibility-aware OKS, gap-aware
acceleration, presence-aware denominators, coordinate corrections, metric
renaming, or any other later protocol change.

## Status

- State: active; corrected adapter implementation
- Current layer: accepted implementation architecture
- Next action type: implement and portable-test the TPA normalization,
  common-manifest, worker, and transaction boundary
- Next authorized action: complete adapter smoke gates. Do not run corrected
  real-data evaluation until those gates pass.
- Blocker: none. All five first-work-package completion criteria remain
  satisfied.

### Accepted native-output prerequisite

The completed
[native-output and scene-semantics audit](../task_native_output_and_scene_semantics_audit/)
establishes that both methods provide usable scene/world support. It also
classifies `track_flag == False` dense state as invalid evaluation padding;
occlusion recovery is represented by valid output without method observation,
not by bypassing `track_flag`. Corrected metrics therefore use a GT-owned
population, valid matched geometry, and separately exposed completeness.

### Accepted corrected-population direction

The corrected profile will keep two distinct, immutable evaluation views.

1. **GT-visible profile.** Its GT-owned person-frame population contains the
   177,315 released GT-present person-frames with at least one positive
   (`0.5` or `1`) COCO-17 source-visibility channel. Whole-sequence identity
   assignment yields valid matched person-frames, false negatives, and false
   positives. Report person-frame precision, recall, and F1 separately from
   geometric quality. Geometric metrics consume valid matched support only and
   receive no fixed missing-pose substitution.
2. **Frozen Crowd4D/DyCrowd common-support profile.** Freeze the 167,497
   `(scene, frame_id, GT track_id)` cells in the intersection of the GT-visible
   population, valid Crowd4D support, and valid DyCrowd support. This excludes
   9,818 GT-visible person-frames (`5.54%`) and must be explicitly described as
   a method-derived paired-quality subset. It is frozen once and is never
   recomputed when another method is added.

A later method uses its own reviewed GT-MOT identity correspondence against
the same frozen common-support manifest. The manifest is not intersected again
with the new method's output support: a missing new-method output inside it is
a false negative, while geometry remains conditional on valid matched output.
GT-MOT establishes identity correspondence and does not fabricate output
existence.

For Crowd4D versus DyCrowd, the common-support profile isolates paired geometry
without a missing-case penalty because both methods exist there by definition.
The full GT-visible profile remains necessary to expose their shared blind
spots. Results from the two profiles must not be merged into one unnamed
number.

F1 is not multiplied into or divided out of the headline MPJPE-family errors.
Conditional geometric error and completeness remain separate primary columns.
A bounded higher-is-better quality multiplied by F1 may be published only as a
separately named secondary composite; an `error / F1` diagnostic likewise may
not replace its source error.

### Corrected-metric clearing order

Requirements are selected serially before Mathematical Architecture:

1. frame-level 3D joint metrics: `MPJPE`, `T-MPJPE`, `RT-MPJPE`, and
   `PA-MPJPE`;
2. whole-sequence 3D alignment metrics;
3. crowd-layout metrics: `PPDS`, `PA-PPDS`, and corrected `PCOD`;
4. visibility-aware `OKS`;
5. gap-aware `ACCEL`;
6. final reduction, naming, support reporting, and compatibility-only columns.

All six attended requirement items are accepted. This completion authorizes the
Mathematical Architecture layer, not implementation or a corrected run.

### Accepted frame-level 3D joint metrics

The corrected headline set contains `MPJPE`, `T-MPJPE`, `RT-MPJPE`, and
`PA-MPJPE` over the canonical SMPL-24 body joints. Released
`joints_3d_smpl54[:24]` is the GT representation; both method adapters produce
the same named SMPL-24 representation. The release-native H36M-17 order is not
authoritatively documented and has no direct source-visibility array, so the
author H36M-14 path remains compatibility-only rather than a corrected stable
contract.

Visibility and matching operate outside the metric:

- source visibility selects the GT person-frame population; once a
  person-frame is admitted, all 24 joints are evaluated regardless of each
  joint's visibility state;
- whole-sequence association supplies identity-paired GT and prediction
  arrays, but match ratio, F1, false positives, false negatives, missing
  penalties, and redundancy never enter a geometric metric or reducer;
- `T-MPJPE` uses the SMPL pelvis as its per-person-frame translation anchor;
  `RT-MPJPE` fits one per-person-frame rigid `SE(3)` transform without scale;
  `PA-MPJPE` fits one per-person-frame similarity transform; every fit and
  error uses all 24 joints;
- the geometry reducer accumulates the 24 joint errors over its declared
  paired person-frame support and applies one micro mean. It never frame-macro
  averages or weights by a matching statistic.

The full GT-visible view reports conditional geometry support beside its
separate completeness outputs. Crowd4D and DyCrowd use identical geometry
support in the frozen common-support view. A missing prediction has no defined
geometric error; it is represented only by the separate completeness branch.

### Accepted whole-sequence 3D alignment scopes

Keep two explicitly named alignment scopes over the same SMPL-24 all-joint
representation and sequence-constant scene/world coordinates:

1. `T/RT/PA-MPJPE-VISRUN` fits one translation, `SE(3)`, or `Sim(3)` transform
   per GT-defined visibility run. This is the primary within-observable-run
   sequence-quality view.
2. `T/RT/PA-MPJPE-TRACK` fits one corresponding transform over all admitted
   frames of one GT `track_id`, across visibility gaps. This is the stricter
   cross-occlusion/re-entry consistency diagnostic.

The visibility-run partition is GT-owned. Method `track_flag`, missing output,
or matching success must never split or merge a run, because doing so would let
a method buy additional alignments by omitting frames. Frame gaps do not alter
the TRACK fit, and prediction holes inside one GT run do not create extra
VISRUN fits. Each scope first fits its declared transform and then contributes
all resulting frame/joint errors to one micro reduction; it does not macro
average runs or tracks and receives no matching/F1 statistic.

The precise, deterministic person-visibility predicate and minimum/degenerate
run rules are deferred to Mathematical Architecture and implementation. The
leading low-freedom definition remains a maximal consecutive GT-present run in
which at least one COCO-17 source channel is `0.5` or `1`, with no smoothing or
method-dependent repair. This bounded detail does not block the accepted
two-scope requirement.

No corrected first-two-frame `W-MPJPE` is retained. The author root-relative
WA/W quantities remain compatibility-only. An unaligned sequence MPJPE is not
duplicated because, on identical support, it reduces the same raw joint errors
as frame-level absolute MPJPE.

### Accepted crowd-layout metrics

`PPDS`, `PA-PPDS`, and both explicitly named PCOD variants consume the
identity-aligned per-frame population supplied by association, but receive no
matching statistic, recall multiplier, F1 multiplier, missing-pose penalty, or
completeness penalty. Their pair population is the set of all unordered valid
matched person pairs in each admitted frame, reduced by one pair-micro mean over
the declared evaluation view. This conditional pair denominator can change
when a method misses an identity. The frozen common-support view supplies
identical person pairs to Crowd4D and DyCrowd for the current comparison.

`PPDS` retains the Crowd3D clipped relative pair-distance score. `PA-PPDS`
first fits one crowd-level `Sim(3)` transform per frame over all valid person
anchors and then evaluates the same pair score; it must not fit an independent
transform per pair. Exact epsilon and degenerate-frame support rules remain
bounded Mathematical Architecture details.

Keep two PCOD definitions as distinct result columns rather than silently
choosing one meaning for the same name:

1. `PCOD-3C-0.3m` is the corrected headline metric. It uses the same SMPL
   pelvis anchor for GT and prediction, measures camera optical-axis depth, and
   independently classifies each GT and predicted pair as closer, farther, or
   roughly equal. A pair is roughly equal when the absolute pelvis-depth
   difference is at most `0.3 m`; the result is pair-class accuracy.
2. `PCOD-2C-STRICT` reproduces the supplied evaluator's strict sign-agreement
   quantity, including its observed prediction-torso versus GT ground-projected
   HSIP anchor semantics and failed ties. It is compatibility-only, not an HJ
   corrected headline metric.

Neither PCOD variant is multiplied by person recall or F1; completeness remains
a separate output. The corrected metric uses the shared pair population and
does not emit a generic valid-support counter. An empty pair population is
not available, and any observed invalid case must receive a specific reviewed
label before it can be retained.

### Accepted visibility-aware OKS

The corrected profile has one pure conditional `OKS-VIS` metric. Map the native
VirtualCrowd visibility encoding back to the COCO keypoint-visibility roles
before applying the original OKS inclusion rule:

- native `0` maps to COCO `v=0` and is excluded;
- native `0.5` (self-occluded but labeled) maps to COCO `v=1`;
- native `1` (unoccluded) maps to COCO `v=2`.

This is an evaluation-policy mapping, not a claim that native `0` and COCO
`v=0` have identical annotation provenance. As in original OKS, every `v>0`
joint is included equally; `0.5` is an inclusion state, never a fractional
metric weight. The GT-visible population already requires at least one native
`0.5/1` COCO-17 joint per person-frame, so corrected OKS has no zero-valid-joint
case on that view.

`OKS-VIS` compares released GT COCO-17 image joints with predicted SMPL-54
COCO-17 joints projected through that method's own native `cam_int`. It uses
the conventional COCO-17 sigmas and GT `bbox_xyxy` area. It first averages the
joint similarities admitted by the GT mask within one identity-paired
person-frame, then applies one person-frame micro mean over the declared view.
Recall, F1, missing-pose substitutions, and completeness penalties do not enter
the metric; geometry remains conditional on the supplied association. A
non-finite or non-positive-depth projected joint cannot be silently removed or
replaced by a zero similarity: the generic validity checker fails with that
exact predicate.
If such a case is observed and must be retained, it receives a separately
reviewed specific invalid label and policy.

The supplied evaluator's displayed `OKS` is not a second corrected OKS metric:
it selects pairs through its matching procedure, ignores visibility for its
all-17-joint inner OKS, and multiplies the matched-pair mean by person recall.
Retain that exact quantity only in T2/T3 author-compatibility outputs as a
matching-coupled legacy result. Its inner all-joint OKS leaf is documented for
reproduction but does not receive a separate corrected result column.

### Accepted gap-aware acceleration metrics

The corrected headline is `ACCEL-WORLD`. For world-space SMPL-24 joint `j` in
three consecutive frames, define the unit-frame second difference

```text
a(t, j) = J(t + 1, j) - 2 * J(t, j) + J(t - 1, j)
```

and score the full 3D vector residual
`||a_pred(t, j) - a_gt(t, j)||_2`. Prediction and GT use the same
sequence-constant scene/world coordinate system, so root trajectory and
articulated motion both remain in the metric. This global interpretation is
the headline because the task evaluates global crowd reconstruction.

`ACCEL-POSE` may be emitted as an optional, non-headline pose-dynamics
diagnostic. It applies the same vector-residual formula after prediction and GT
each subtract their own SMPL pelvis in every frame. It intentionally removes
root translation and therefore must not be substituted for `ACCEL-WORLD`.

Both corrected variants use a GT-owned candidate set: the three frame IDs must
be strictly consecutive and belong to the same GT-defined visibility run.
Prediction holes never compact the timeline, bridge a gap, or create a new
triple. Once the person-frames are admitted, all SMPL-24 joints are used. The
reducer is one triple-joint micro mean, the evaluated triple support is
reported, and no matching/recall/F1 value or missing-pose substitution enters
the acceleration error.

Until an authoritative dataset FPS is established, both corrected values are
named and reported in `mm/frame^2`; no implicit `30^2` factor is allowed. A
later evidenced FPS may provide a deterministic physical-unit conversion, not
a silent metric redefinition.

A primary-source comparison found that the released
[GENMO](https://github.com/NVlabs/GENMO/blob/caac11010d88565f13e2f4d54e54a9604bd1fc59/gem/utils/eval_utils.py#L366-L397)
and
[GVHMR](https://github.com/zju3dv/GVHMR/blob/cb702cd2184ba2c51e041beade6b5625432738d3/hmr4d/utils/eval/eval_utils.py#L260-L291)
code uses pelvis-relative vector acceleration error, while the public
[CRISP utility](https://github.com/Z1hanW/CRISP-Real2Sim/blob/4d32c3b857b43dd9f4cd1032423feab684a7bdec/MotionTracking/smpllib/smpllib/smpllib/smpl_eval.py#L182-L222)
uses vector acceleration error without an internal root subtraction, although
its binding to the paper table is not published. All three support vector
residuals rather than the supplied evaluator's magnitude-only quantity.

The author quantity compares scalar acceleration magnitudes on compacted,
pelvis-relative H36M-14 samples and track-macro averages the results. Preserve
it only in author-compatibility outputs under an explicitly legacy identity
such as `ACCEL-MAG-COMPACT-AUTHOR`; it is not a corrected ACCEL variant.

### Accepted reduction, completeness, validity, and naming partition

Completeness is set-derived rather than a generic support taxonomy. For the
full `GT_VISIBLE` view, let `G` be its GT person-frame instances, `P` the
prediction instances in the frozen scene/frame domain, and `M` the one-to-one
identity-consistent association relation. A prediction specifically mapped to
a GT-present all-zero-visibility key is excluded with that GT case; unmapped,
absent-GT, wrong-ID, and duplicate occurrences inside the frame domain remain
eligible excess predictions. Correctly covered instances are `M`; missing GT
instances are
`G \\ projection_GT(M)`; excess predictions are
`P \\ projection_prediction(M)`. Precision, recall, and F1 derive only from
those three disjoint outcomes. An additional prediction targeting an already
matched GT naturally remains excess; it receives a duplicate-specific label
only when that distinction is materially present and reviewed.

The frozen common-support view restricts only matched geometry and publishes no
second completeness/precision/recall/F1 partition.

Do not publish generic `candidate_support`, `evaluated_support`, or
`unsupported_or_degenerate_count` columns. Metric reducers consume the
identity-paired geometry selected by `M` and never receive completeness values.
Frame/joint, pair, person-frame OKS, and exact-consecutive triple/joint values
use their already accepted micro reductions. Any metric-specific population or
failure fact that is actually needed receives a precise name such as an exact
consecutive-triple count; no speculative catch-all field is created.

A shared validity checker enforces the normalized contract before metric
reduction. It does not invent invalid categories. If a concrete non-finite,
shape, coordinate, projection, or alignment failure appears, the operation
fails or records evidence first; a specifically named invalid label and its
policy are then reviewed before accepting results. Invalid data is never
silently removed to improve a metric denominator.

Corrected stable names distinguish absolute world, alignment scope, visibility,
and temporal meaning. The leading set is `MPJPE-WORLD`, `T-MPJPE`,
`RT-MPJPE`, `PA-MPJPE`, the `SEQ-{T,RT,PA}-MPJPE-{VISRUN,TRACK}` family,
`PPDS`, `PA-PPDS`, `PCOD-3C-0.3m`, `OKS-VIS`, and `ACCEL-WORLD`;
`ACCEL-POSE` is optional diagnostic. Author `MPJPE`, `WA/W`, strict PCOD,
magnitude-only ACCEL, recall-adjusted OKS, GMPJPE, matched ratio, and Score stay
in a separate author-compatibility namespace whose run metadata explicitly
states author-greedy versus GT-MOT association. No new corrected composite
Score is defined.

## Corrected-Protocol Detailed Residence

- [VirtualCrowd Corrected Metric Protocol](../../../docs/design/tasks/virtualcrowd-corrected-metric-protocol/README.md)

## Second Work Package - Metric Semantics Audit

Purpose:

1. reconstruct each retained metric from the independent adapter implementation
   and frozen author evidence;
2. compare its mathematical quantity, units, coordinate frame, visibility,
   missing-person treatment, temporal support, denominator, and reduction with
   the intended HJ evaluation semantics;
3. classify each metric as retain, clarify/name, parameterize, or correct, with
   the expected consequence of any later change.

Boundary:

- this work package is read-only with respect to metric formulas and result
  generation;
- the completed T2 and GT-MOT result sets remain historical evidence;
- no recommendation becomes an accepted protocol change until the user reviews
  and explicitly selects it;
- because this is a one-time conclusion that later protocol decisions will
  depend on, it skips a new Layered Design residence and receives a dedicated
  mathematical review after the audit is recorded.

### Metric-Semantics Audit

Evidence basis:

- the public [Crowd4D paper](https://arxiv.org/html/2607.19517v1) defines the
  reported metric set and describes WA/W as global-coordinate sequence metrics;
- the public [Crowd3D paper](https://arxiv.org/abs/2301.09376) owns the original
  PPDS/PA-PPDS definition, while the public
  [SMAP paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600545.pdf)
  defines cited PCOD as three-class root-depth accuracy with a 30 cm
  roughly-equal band;
- the observed implementation is frozen in the accepted
  [author-evaluator analysis](../../../docs/design/tasks/author-evaluator-logic-analysis/README.md)
  and the independent
  [T2 parity architecture](../../../docs/design/tasks/hj-composed-author-parity-reproduction/README.md);
- person-presence and visibility counts below are direct reductions of the
  eight content-addressed VirtualCrowd labels in the accepted GT-MOT receipt:
  actual slots are `sum(len(frame.instances))`; a COCO-17 GT joint is visible
  iff its native third channel is positive.

Scope distinction:

- the Crowd4D paper reports eight Table-2 metrics: `PPDS`, `PA-PPDS`, `PCOD`,
  `MPJPE`, `PA-MPJPE`, `WA-MPJPE`, `W-MPJPE`, and `ACCEL`;
- the supplied evaluator additionally emits `matched ratio`, `OKS`, `GMPJPE`,
  and `Score`; these four are evaluator diagnostics rather than Table-2 result
  columns;
- `hjlib-evaluation` currently owns method-neutral unreduced joint-position
  error and OKS leaves plus its earlier dense MPJPE/T-MPJPE/Jitter reducer. It
  does not yet own stable generalized definitions for PPDS, PCOD, WA/W, ACCEL,
  missing-person penalties, or the author composite score. Exact T2/T3
  reproduction therefore proves compatibility, not HJ protocol acceptance.

Cross-cutting findings:

1. Direct GT identity pairing is accepted and needs no further association
   correction. Every later metric must consume only same-frame GT presence for
   the mapped identity.
2. The fixed `150 mm` replacement conflates coverage with geometric error and
   can make dropping a case with true error above 150 mm improve a lower-is-
   better metric. HJ results should report support/recall separately and reduce
   geometric error only over its declared valid population. The author penalty
   remains available only in the named compatibility profile.
3. Temporal missing slots use `legacy_person_count * frame_count`, not actual
   GT presence. In the first 200 frames the four view-2 scenes contain
   `11,522/15,535/11,119/33,562` actual person-frames versus
   `17,000/30,000/14,400/35,800` nominal slots: excess denominators are
   `47.54%/93.11%/29.51%/6.67%` relative to actual presence. This is material,
   not an edge case.
4. Frame metrics are averaged after per-frame reduction. The existing HJ leaves
   intentionally own no generalized reduction policy, so the alternative is
   still a protocol choice rather than an accepted contract. Candidate
   population reductions accumulate numerator and support count before one
   division: person-frame weighting for person metrics, pair weighting for pair
   metrics, and valid-window weighting for finite differences. A separately
   named scene/frame-macro view is another valid choice; every published value
   must name which population it represents.
5. Finite-difference metrics must not treat a source-frame gap as one unit time
   step. Exact-consecutive runs are the candidate support for ACCEL/Jitter.
   WA/W alignment scope is a separate decision: gaps may exclude individual
   samples, split alignment fits, or leave one whole-track fit over observed
   samples. W's initialization window likewise needs an explicit observed-
   sample versus consecutive-frame rule; this audit does not select one.
6. Every stable result name must bind joint set, coordinate frame, unit,
   alignment, visibility, missing-data, and reduction semantics. Degenerate
   alignment inputs must have explicit invalid/support behavior rather than
   inherited non-finite propagation.

Per-column verdict:

| Column | Observed author-compatible quantity | HJ verdict | Later protocol action |
| --- | --- | --- | --- |
| `matched ratio` | Per-frame `matched GT / present GT`, then mean of frames | Retain with clarification | Rename or define as GT recall. Candidate population reduction is `sum matched / sum present`; frame-macro recall may remain separately named. Report prediction precision/redundancy separately if needed. |
| `PPDS` | Clipped relative error of all matched torso-center pair distances, multiplied by person recall | Retain mathematical core; parameterize support | Preserve the Crowd3D pair score, but expose conditional matched-pair PPDS and an explicitly coverage-adjusted variant. Do not silently use person recall as a substitute for pair support; reduce by pair counts. |
| `PA-PPDS` | Per-frame crowd-point Sim(3) alignment followed by PPDS, multiplied by person recall | Retain mathematical core; parameterize support | Keep per-frame crowd-layout alignment, name the alignment scope, and apply the same conditional/coverage and pair-weighted rules as PPDS. |
| `PCOD` | Strict two-class sign agreement between predicted torso depth and GT ground-projected HSIP depth; ties fail | Current quantity is non-contract; replacement is provisional | The externally sourced SMAP definition classifies each root-depth difference as closer, farther, or roughly equal when its magnitude is at most `0.3 m`, then reports pair-class accuracy. That is the leading replacement candidate, but anchor type, tolerance, and pair reduction require user selection. Any accepted variant must use the same declared anchor type on prediction and GT. The current torso/HSIP asymmetry remains compatibility-only. |
| `MPJPE` | H36M-14 error after subtracting each person's pelvis, plus unmatched-GT penalty | Rename and correct reduction | Its geometric core is HJ `T-MPJPE`/root-relative MPJPE, not absolute MPJPE. Report matched-support root-relative error without a fixed missing penalty; add a separately named absolute metric when required. |
| `PA-MPJPE` | Per-person, per-frame H36M-14 Sim(3)-aligned pose error, plus unmatched-GT penalty | Retain mathematical core | Keep as PA-MPJPE with joint set/dtype/alignment declared; remove the fixed missing penalty and population-average by valid person-joint support. |
| `WA-MPJPE` | One Sim(3) fit over each compact track's pelvis-relative H36M-14 poses | Current input contradicts the stated global-coordinate meaning | The replacement needs one coordinate frame constant over the sequence: declared world coordinates, or a fixed-camera frame only when its fixed rigid relation is evidenced. Per-frame camera coordinates are not interchangeable with world coordinates. Whole-track versus per-run alignment and gap support remain separate choices. The current root-relative input discards trajectory translation and stays compatibility-only. |
| `W-MPJPE` | Sim(3) fit from the first two compact samples of each pelvis-relative track, applied to the track | Current input contradicts the stated global-coordinate meaning | Use the same selected sequence-constant coordinate as WA. Separately select whether the fit uses the first two observed samples, the first two consecutive frames, or another declared initialization window. Current compact, root-relative input is compatibility-only. |
| `ACCEL` | Difference of scalar acceleration magnitudes, normed across joints, on compact pelvis-relative samples; no frame-gap or FPS factor | Parameterize or replace | Gap-aware support and explicit coordinates/units are required. A vector acceleration-error candidate computes the 3D acceleration difference before its norm on exact-consecutive triples; this is a different metric, not a parity-preserving fix, and requires selection. The legacy magnitude-difference variant can remain explicitly named. Neither is HJ Jitter, which is absolute world-space jerk. |
| `OKS` | Direct-identity paired COCO-17 OKS using every joint and GT bbox area, multiplied by person recall | Visibility/support decision required; retain leaf | Use the GT visibility channel as the candidate OKS valid-joint mask and declare native-2D versus reprojected-GT source. In the selected data, `427,016 / 3,041,946 = 14.04%` of COCO-17 GT joint labels are non-visible and `113,684 / 178,938 = 63.53%` of person-frames contain at least one such joint, so all-joint OKS is materially different. The candidate default excludes a person-frame with zero valid joints from OKS support and reports that unsupported count; treating it as zero or failing are alternatives that must not be implicit. Recall and reduction population remain separately selected. |
| `GMPJPE` | Unaligned absolute camera-coordinate H36M-14 track error with nominal missing penalty | Retain only as explicitly named diagnostic | Rename to camera-coordinate absolute MPJPE, use actual presence and valid person-frame support, and do not imply world coordinates unless a common world transform is part of the contract. It is not a Crowd4D Table-2 column. |
| `Score` | Weighted frame composite of PA-PPDS, hidden scale score, PCOD, OKS, and hidden redundancy penalty | Legacy only | Do not use as an HJ headline score. It is absent from the paper table, hides two unreported components, mixes incompatible support semantics, and has no evidenced weight calibration. A future composite would need visible components and a separately reviewed objective. |

Provisional correction order, if later selected by the user:

1. select actual-presence populations, reduction weighting, and conditional
   versus coverage-adjusted outputs;
2. select metric identities and coordinate names (`MPJPE`, `WA/W`, `GMPJPE`);
3. select a PCOD definition, visibility/zero-support OKS policy, and gap-aware
   ACCEL definition;
4. publish a separately named HJ-reviewed profile beside, never over, the T2
   author-parity and T3 GT-MOT compatibility profiles;
5. keep `Score` compatibility-only unless a new composite is independently
   justified.

This audit does not select one of those corrections for implementation and does
not authorize a result rerun.

Review history:

- 2026-08-14: Dedicated Mathematical Architecture review found no Critical and
  five Concerns. All were accepted: proposed weighting is now explicitly
  provisional; WA/W coordinate and gap/alignment choices are separated; vector
  ACCEL is a candidate metric rather than an assumed correction; SMAP PCOD is
  marked external/provisional; and zero-valid-joint OKS alternatives are
  explicit. Focused re-review accepted the revised audit with no remaining
  Critical or Concern.

## First Work Package

Purpose:

1. recover a provenance-bound per-scene mapping from each DyCrowd prediction
   column to the VirtualCrowd GT `track_id`; the user relayed the author's
   confirmation that the DyCrowd predictions were produced from GT-MOT input;
   this task owns the reviewed whole-track 2D recovery because the artifact does
   not serialize or define the prediction-column mapping;
2. consume Crowd4D's native stored GT identities and the recovered DyCrowd
   identities as direct frame-pairing contracts;
3. compute both eight-scene workflows with the existing twelve author metrics,
   missing penalties, temporal formulas, aggregation, and display rounding;
4. publish a separately named GT-MOT identity result set and a complete delta
   against T2 author parity.

Boundary:

- whole-track geometry may recover DyCrowd's lost identity serialization once;
  it is not a per-frame evaluation association rule;
- the frozen supplied evaluator and T2 profile remain unchanged;
- Crowd4D/DyCrowd inference is not rerun;
- Crowd4D-native schemas, recovery, orchestration, and receipts remain owned by
  `hj-tpa-crowd4d`; stable metrics remain owned by `hjlib-evaluation`;
- no later candidate protocol change is implied by this activation.

## Detailed Residence

- [GT-MOT Identity Baseline](../../../docs/design/tasks/gt-mot-identity-baseline/README.md)

## First-Work-Package Completion Criteria

1. Every active DyCrowd prediction column in all eight scenes has an injective,
   evidence-backed GT-ID mapping or the operation fails with explicit ambiguity
   evidence; mapping and diagnostics are frozen in a receipt-bound sidecar.
2. Crowd4D direct IDs and DyCrowd recovered IDs pair predictions only with the
   same GT identity when that identity is present in the frame.
3. The existing twelve metric formulas, penalties, temporal formulas,
   frame/scene/global reductions, and four-decimal rendering are unchanged and
   covered by regression tests.
4. Both methods complete all eight scenes into a new output residence without
   modifying supplied inputs, T2 evidence, or historical tables.
5. The receipt binds inputs, mapping algorithm and mapping evidence, code,
   runtime, outputs, and every T2-versus-GT-MOT metric delta.

## Activation Gate

The original activation gate required:

1. completed T2 author-parity tables and receipts;
2. a user-reviewed list of protocol semantics to retain, parameterize, or
   replace;
3. a new task-specific Layered Design residence and review before implementation.

Items 1 and 2 were satisfied by completed T2 evidence and the user's bounded
2026-08-13 decision. Item 3 passed dedicated Mathematical and Code Architecture
review before implementation began.

Only direct GT identity is accepted for the first work package. Presence-aware
temporal denominators, visibility-aware OKS, gap-aware acceleration, and
coordinate/name corrections remain deferred evidence prompts.

## Handoff

The first T3 work package is complete. Its accepted artifacts are mirrored at
[`hj-tpa-crowd4d` GT-MOT identity evidence](../../../../hj-tpa-crowd4d/campaigns/03_hj_composed_author_parity/evidence/gt_mot_identity/).
The mapping audit covers 909 edges; the final result contains two complete
`9 x 12` grids and T2 deltas. T2 summaries reproduced byte-identically, strict
Pyright reports zero errors, and 19 portable smoke tests pass. Return to the
campaign for user review; do not infer authorization for deferred changes.
