# VirtualCrowd Corrected Metric Protocol

## Requirements

### Purpose and boundary

Define the method-neutral corrected evaluation contract selected in Campaign 03
T3. The stable mathematics and normalized contracts reside in
`hjlib-evaluation`; Crowd4D/DyCrowd artifact parsing, scene normalization,
GT-MOT identity recovery, orchestration, and receipts reside in
`hj-tpa-crowd4d`. Dependency direction remains:

```text
hj-tpa-crowd4d -> hjlib-evaluation
```

The corrected profile is independent of, and never overwrites, the frozen T2
author-parity and T3 GT-MOT author-compatible profiles. It does not reproduce
inference, invent missing poses, multiply geometry by completeness, or define a
new composite score.

### Evaluation views

The protocol has two immutable views:

1. `GT_VISIBLE`: the 177,315 released GT-present person-frame keys whose
   COCO-17 source visibility contains at least one `0.5` or `1`.
2. `C4D_DYCROWD_COMMON`: the frozen 167,497-key intersection of `GT_VISIBLE`,
   valid Crowd4D output support, and valid DyCrowd output support. Its identity
   is frozen by manifest and is never recomputed for later methods.

`GT_VISIBLE` owns completeness headlines. `C4D_DYCROWD_COMMON` is an explicitly
method-derived paired-quality view. A later method is checked against the same
manifest; its missing keys do not shrink the view.

### Corrected result identities

Headline geometry:

```text
MPJPE-WORLD
T-MPJPE
RT-MPJPE
PA-MPJPE
SEQ-T-MPJPE-VISRUN
SEQ-RT-MPJPE-VISRUN
SEQ-PA-MPJPE-VISRUN
SEQ-T-MPJPE-TRACK
SEQ-RT-MPJPE-TRACK
SEQ-PA-MPJPE-TRACK
PPDS
PA-PPDS
PCOD-3C-0.3m
OKS-VIS
ACCEL-WORLD
```

`ACCEL-POSE` is an optional pose-dynamics diagnostic. `PCOD-2C-STRICT` and the
supplied evaluator's MPJPE, WA/W, ACCEL, OKS, GMPJPE, matched ratio, and Score
remain author compatibility only.

### Reduction and validity requirements

- Completeness and geometry are separate outputs.
- Geometry is conditional on identity-paired valid predictions and receives no
  fixed missing value, recall/F1 factor, or method-controlled view definition.
- Joint metrics reduce joint samples once; pair metrics reduce unordered person
  pairs once; OKS reduces person-frame scores once; acceleration reduces exact
  consecutive triple/joint samples once. No frame, scene, run, or track macro
  headline is introduced.
- A shared normalized-input checker validates shape, dtype, finiteness, units,
  coordinate identity, key uniqueness, and required calibration before metric
  reduction. It is fail-fast and does not silently filter invalid payloads.
- There is no universal invalid taxonomy. A concrete observed failure must be
  evidenced and assigned a specific reviewed label and policy before a result
  containing it is accepted.

## Mathematical Architecture

### 1. Domains, keys, and normalized inputs

A GT person-frame key is:

```text
u = (scene_id, integer frame_id, positive GT track_id)
```

For one view `V`, let `G_V` be its finite set of GT keys. Every GT key carries:

```text
J_gt_world[u]        float64 (24, 3), metres
pelvis_gt_world[u]   J_gt_world[u, 0]
J_gt_coco2d[u]       float64 (17, 2), pixels
visibility_native[u] float64 (17,), values in {0, 0.5, 1}
bbox_gt_xyxy[u]      float64 (4,), pixels
pelvis_gt_cam_z[u]   float64 scalar, metres
```

The adapter supplies normalized prediction occurrences `p` with a stable
method-local identity and, when association accepts them, the same-frame target
GT key. A prediction used by corrected geometry carries:

```text
J_pred_world[p]      float64 (24, 3), metres
pelvis_pred_world[p] J_pred_world[p, 0]
J_pred_coco2d[p]     float64 (17, 2), pixels
J_pred_coco_cam_z[p] float64 (17,), metres
pelvis_pred_cam_z[p] float64 scalar, metres
```

The adapter may construct `J_pred_coco2d` by projecting its SMPL-54 COCO-17
joints through that method's native intrinsic matrix. It supplies the same
camera-space joints' pre-projection depths as `J_pred_coco_cam_z`; the metric
layer never tries to infer projection validity from finite pixel coordinates.
`pelvis_gt_cam_z` and `pelvis_pred_cam_z` use one shared capture-camera
coordinate convention: the optical-axis coordinate is positive away from the
camera. The adapter receipt must state the source transform and prove that both
depths are in that convention. The normalized contract also states the
sequence-constant scene/world frame. The metric layer does not parse native
transforms or infer their provenance.

All corrected arithmetic uses `float64`. Unit conversion to display units
occurs only after the declared metric value is reduced.

### 2. Association and the no-duplicate/no-omission partition

Association is an input contract, not a metric leaf. Let `G_full` be
`GT_VISIBLE`. Its manifest also freezes the eight-scene frame domain `F_eval`.
Let `G_invisible` be the GT-present keys in `F_eval` whose 17 native visibility
values are all zero. For every valid method output occurrence `p` in `F_eval`,
the reviewed static identity map returns either its same-frame GT key `h(p)` or
`UNMAPPED`. Define the prediction universe as:

```text
P_m = {p : frame(p) in F_eval and h(p) not in G_invisible}
```

Thus an output for a specifically identified visibility-excluded GT
person-frame is out of scope on both sides. An occurrence in `F_eval` whose
identity is unmapped, maps to no GT-present key at that frame, or duplicates
another output remains in `P_m` and can become FP. Outputs outside the frozen
scene/frame domain are not evaluation occurrences. Let:

```text
A_m subset G_full x P_m
```

be the supplied one-to-one association relation. It must satisfy:

- each GT key and prediction occurrence appears in at most one pair;
- every pair has equal scene and frame and the prediction's reviewed static
  identity correspondence targets that GT `track_id`;
- a method-side missing frame cannot be repaired by copying or interpolating a
  prediction;
- association provenance is frozen by the adapter receipt.

The corrected completeness partition is defined exactly once, on
`GT_VISIBLE`:

```text
matched_GT = projection_GT(A_m)
matched_prediction = projection_prediction(A_m)
missing_GT = G_full - matched_GT
excess_prediction = P_m - matched_prediction
```

with:

```text
TP = |A_m|
FN = |missing_GT|
FP = |excess_prediction|
recall = TP / |G_full|
precision = TP / |P_m| when |P_m| > 0, else 0
F1 = 2 * precision * recall / (precision + recall)
     when precision + recall > 0, else 0
```

An extra prediction aimed at an already matched GT remains in
`excess_prediction`. A duplicate-specific counter is not part of the base
contract; it is added only if concrete results require that diagnosis.

For geometry on view `V`, define only the restricted matched relation:

```text
M_m,V = {(u,p) in A_m : u in G_V}
```

`C4D_DYCROWD_COMMON` has no separate completeness, precision, recall, or F1.
Unmapped, wrong-ID, duplicate, and out-of-view predictions are already owned
by the single full `GT_VISIBLE` partition; the common view cannot redefine or
hide that global FP population.

### 3. GT visibility and temporal partitions

Map the native COCO-17 visibility encoding to COCO roles:

```text
0   -> v=0
0.5 -> v=1
1   -> v=2
```

This is a protocol mapping, not a claim of identical annotation provenance.
Person-frame visibility is:

```text
visible(u) = any(mapped_visibility[u, k] > 0 for k in COCO-17)
```

For each GT track, a `VISRUN` is one maximal run of visible keys with frame IDs
increasing by exactly one. There is no smoothing. Prediction presence,
association success, or method track state cannot split, join, or extend a
VISRUN. A `TRACK` is all visible keys of one GT track in frame order, including
across gaps.

### 4. Frame-level joint metrics

For one matched pair, write predicted and GT SMPL-24 joints as
`X, Y in R^(24 x 3)` in the normalized world frame.

`MPJPE-WORLD` uses:

```text
e_world[j] = ||X[j] - Y[j]||_2
```

`T-MPJPE` subtracts each skeleton's SMPL pelvis:

```text
X_T[j] = X[j] - X[0]
Y_T[j] = Y[j] - Y[0]
e_T[j] = ||X_T[j] - Y_T[j]||_2
```

`RT-MPJPE` chooses one reflection-disabled rigid transform per person-frame:

```text
(R*, t*) = argmin over R in SO(3), t in R^3
           sum_j ||R X[j] + t - Y[j]||_2^2
e_RT[j] = ||R* X[j] + t* - Y[j]||_2
```

`PA-MPJPE` chooses one positive-scale reflection-disabled similarity transform
per person-frame:

```text
(s*, R*, t*) = argmin over s > 0, R in SO(3), t in R^3
               sum_j ||s R X[j] + t - Y[j]||_2^2
e_PA[j] = ||s* R* X[j] + t* - Y[j]||_2
```

Every result is the single mean of all 24-joint errors over all pairs in
`M_m,V`, converted from metres to millimetres after reduction. Per-joint source
visibility does not mask these 3D errors once the person-frame belongs to the
view.

### 5. Sequence-level aligned joint metrics

For a GT-defined VISRUN or TRACK scope `S`, collect only matched keys in that
scope and flatten their SMPL-24 points into paired sets `X_S,Y_S`. The
sequence-level translation is a least-squares point-set fit, not the
per-person-frame pelvis subtraction used by `T-MPJPE`:

```text
t*_S = mean over matched (u,j) in S of (Y[u,j] - X[u,j])
```

`SEQ-T` applies that one `t*_S` to every matched joint in `S`. `SEQ-RT` and
`SEQ-PA` fit exactly one rigid or positive-scale similarity transform to the
same flattened paired sets using the Section 4 objectives, then apply it to
every matched point in `S`.

Missing prediction frames remain missing: they neither create new fits nor
split a VISRUN. A TRACK fit may use matched samples on both sides of a GT gap;
the frame gap does not alter its single fit. Across scopes, accumulate all
resulting matched frame/joint errors and take one micro mean. Scope count is not
a weight.

Stable names prefix the sequence meaning:

```text
SEQ-{T,RT,PA}-MPJPE-VISRUN
SEQ-{T,RT,PA}-MPJPE-TRACK
```

No corrected first-two-frame W metric and no duplicate unaligned sequence
MPJPE are defined.

### 6. Crowd-layout pair population and PPDS

For one frame, let `I` be the GT identities present in the frame's association
relation and use the SMPL pelvis as the symmetric prediction/GT crowd anchor.
The pair population contains each unordered pair `{i,j}` from `I` exactly once.

For GT and predicted pelvis distances `d_gt(i,j)` and `d_pred(i,j)` in metres:

```text
PPDS(i,j) = max(
    0,
    1 - abs(d_pred(i,j) - d_gt(i,j)) / d_gt(i,j),
)
```

The normalized-input checker requires every consumed `d_gt` to be strictly
positive. A concrete coincident-anchor case must fail with evidence and receive
a reviewed specific policy; it is not silently dropped by a generic epsilon.
The result is one micro mean across all admitted pairs in all frames. This is
conditional matched geometry: its pair denominator can change when a method
misses an identity, but it is never multiplied by, divided by, or otherwise
penalized with completeness.

For `PA-PPDS`, solve the reflection-disabled least-squares `Sim(3)` fit from all
predicted pelvis anchors in the frame to all GT pelvis anchors. Let `R*` be an
optimal `SO(3)` rotation, `x_bar/y_bar` the two centroids, and:

```text
V_x = sum_i ||x_i - x_bar||_2^2
C = sum_i (y_i - y_bar)^T R* (x_i - x_bar)
s* = C / V_x
```

The admitted population requires `V_x > 0` and `C > 0`. These predicates make
the positive scale unique even when a two-person rotation is non-unique. Since
rotation and translation preserve pair distances, PA-PPDS then uses
`s* * d_pred(i,j)` directly in the PPDS equation; it does not depend on which
optimal rotation is returned. It never fits one scale or transform per pair.
Frames with fewer than two matched people have no pair contribution. Failure
of either predicate, or any non-finite fit quantity, fails the validity gate
with its exact predicate rather than returning a fabricated score.

### 7. PCOD

For ordered representation of unordered pair `{i,j}` with `i<j`, define camera
optical-axis pelvis-depth differences in metres:

```text
delta_gt = pelvis_gt_cam_z(i) - pelvis_gt_cam_z(j)
delta_pred = pelvis_pred_cam_z(i) - pelvis_pred_cam_z(j)
```

Classify each independently with tolerance `tau=0.3 m`:

```text
class(delta) = EQUAL   if abs(delta) <= tau
               CLOSER  if delta < -tau
               FARTHER if delta > tau
```

`PCOD-3C-0.3m` is the pair-micro accuracy of
`class(delta_pred) == class(delta_gt)`. Prediction and GT use the same SMPL
pelvis anchor and camera-depth convention. `PCOD-2C-STRICT` preserves the
author's asymmetric strict-sign behavior only in the compatibility profile.

### 8. Visibility-aware OKS

For one matched person-frame, released GT bbox coordinates have half-open
continuous-image semantics and:

```text
A = (x_max - x_min) * (y_max - y_min)
```

The checker requires `x_max > x_min`, `y_max > y_min`, and finite coordinates.
Freeze the COCO-17 order and sigma vector as:

```text
[nose, left_eye, right_eye, left_ear, right_ear,
 left_shoulder, right_shoulder, left_elbow, right_elbow,
 left_wrist, right_wrist, left_hip, right_hip,
 left_knee, right_knee, left_ankle, right_ankle]

sigma = [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
         0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89] / 10
```

For mapped visibility `v[k] > 0`:

```text
e[k] = ||x_pred[k] - x_gt[k]||_2^2
       / ((2 * sigma[k])^2 * A * 2)
similarity[k] = exp(-e[k])
OKS_VIS(person_frame) = mean(similarity[k] over v[k] > 0)
```

Native `0.5` and `1` are both included equally; `0.5` is never a fractional
weight. The view definition guarantees at least one included joint. For every
included joint, `J_pred_coco2d` and `J_pred_coco_cam_z` must be finite and the
source depth must be strictly positive. Violation fails the generic validity
gate with the exact predicate; no joint is silently removed or replaced by a
zero similarity. A retained invalid class would require a later reviewed,
specifically named policy. Final `OKS-VIS` is the person-frame micro mean.
Recall and matching do not enter it.

### 9. Gap-aware acceleration

For world-space SMPL-24 joint `j` on three exact-consecutive frames of one
GT-defined VISRUN:

```text
a_X(t,j) = X(t+1,j) - 2 X(t,j) + X(t-1,j)
a_Y(t,j) = Y(t+1,j) - 2 Y(t,j) + Y(t-1,j)
e_ACCEL(t,j) = ||a_X(t,j) - a_Y(t,j)||_2
```

A triple contributes only when all three person-frames are in the association
relation. Missing frames do not compact the sequence or bridge a gap.
`ACCEL-WORLD` is the micro mean over all contributing triples and all 24 joints,
then converted to `mm/frame^2`.

Optional `ACCEL-POSE` first defines:

```text
q_X(t,j) = X(t,j) - X(t,0)
q_Y(t,j) = Y(t,j) - Y(t,0)
```

and applies the same vector-residual equation to `q_X/q_Y`. It removes world
root translation and is not a global headline. No FPS factor is applied until
an authoritative dataset FPS is evidenced; a future SI-unit representation is
a deterministic conversion, not a different support rule.

### 10. Empty and invalid behavior

An empty matched geometry, pair population, aligned scope collection, or
consecutive-triple population has no numeric metric value; it is represented as
not available rather than zero, infinity, or a missing-pose penalty.
Completeness remains fully defined from the association sets.

The generic checker owns only pass/fail contract validation. It must not
silently remove active prediction payloads, replace non-finite geometry, apply
an undocumented epsilon, or emit a catch-all invalid count. When a real case
fails, evidence must identify the exact predicate; the protocol then adds a
specific label and policy only if that result needs to be retained.

### 11. Compatibility boundary

Corrected output names never overwrite author names. Compatibility results
retain the supplied evaluator's formulas, penalties, reductions, rounding, and
hidden composite behavior under an author namespace, with explicit run metadata:

```text
association_mode = AUTHOR_GREEDY | GT_MOT
```

The corrected profile defines no Score and never combines F1 with a geometry
metric. T2 and the completed GT-MOT compatibility receipts remain immutable
evidence.

## Code Architecture

### Ownership and residence

The implementation is split at the native-artifact boundary:

```text
hj-tpa-crowd4d
  native .pt / VirtualCrowd JSON / reviewed identity sidecars
  -> normalized per-scene corrected input
  -> hjlib-evaluation corrected protocol
  -> per-scene sufficient statistics
  -> TPA transaction, receipt, and result rendering
```

`hjlib-geometry` owns reusable stateless point-registration fit/apply.
`hjlib-evaluation` owns normalized schemas, validation, metric residuals,
pelvis-anchor policy, GT-owned temporal grouping, micro reduction, and
cross-scene reduction.
`hj-tpa-crowd4d` owns trusted native loading, SMPL forward, fixed-camera/world
coordinate evidence, GT-MOT identity maps, common-view manifest production,
short-lived workers, output transactions, and artifact receipts. Stable code
never imports the TPA.

Extend the existing `hjlib-geometry.registration` owner rather than creating a
second solver. Reuse its public `fit_mean_translation` /
`apply_mean_translation` for sequence `T`; add rigid and similarity fit/apply
beside it. `hjlib-evaluation` consequently adds a direct, pinned
`hjlib-geometry` dependency. Geometry remains independent of joint names,
coordinate identity, metric naming, support grouping, and reduction.
`PA-PPDS` consumes the same similarity fit as PA-MPJPE and reads its scale; it
does not own another solver. The `hjlib-camera-solver` Kabsch routine remains
unrelated because it estimates camera rotation from rays.

Add under `hjlib-geometry/src/hjlib_geometry/registration/`:

```text
rigid.py
    Rigid_Registration_Fit
    fit_rigid_registration(target_points, reference_points, pair_valid_mask)
    apply_rigid_registration(points, fit)

similarity.py
    Similarity_Registration_Fit
    fit_similarity_registration(target_points, reference_points, pair_valid_mask)
    apply_similarity_registration(points, fit)
```

Both fit APIs consume paired `(N,3)` arrays plus a boolean `(N,)` mask and fit
target to reference by the reflection-disabled least-squares objectives in the
Mathematical Architecture. Their frozen result arrays use owned immutable
bytes-backed storage, matching `Mean_Translation_Fit`. Rigid state contains
`rotation_target_to_reference (3,3)`, `translation_target_to_reference (3,)`,
`fit_mask (N,)`, and `fit_point_count`. Similarity adds the finite strictly
positive scalar `scale_target_to_reference`. Fit/apply separation permits one
scope fit to be applied to all associated joints. Similarity requires the
Section 6 `V_x>0,C>0` predicates; a non-unique optimal rotation is allowed when
the scale and transformed residual objective are deterministic.

### Stable-library file and API map

Add the following files under `src/hjlib_evaluation/`:

```text
crowd_layout.py
    compute_ppds_scores(predicted_anchors, reference_anchors, scale=1.0)
    compute_pcod_3class_matches(predicted_depths, reference_depths, tolerance_m)

joint_acceleration.py
    compute_joint_acceleration_errors(predicted_joints, reference_joints)

corrected_crowd_data.py
    Corrected_Crowd_Sequence
    Corrected_Crowd_Sequence_Summary
    Corrected_Crowd_Result
    validate_corrected_crowd_sequence(...)

corrected_crowd_protocol.py
    evaluate_corrected_crowd_sequence(
        sequence: Corrected_Crowd_Sequence,
    ) -> Corrected_Crowd_Sequence_Summary
    reduce_corrected_crowd_summaries(
        summaries: Sequence[Corrected_Crowd_Sequence_Summary],
    ) -> Corrected_Crowd_Result
```

`crowd_layout.py` and `joint_acceleration.py` are stateless, unreduced
mathematical leaves. `compute_ppds_scores` consumes matched `(N,3)` anchors and
returns one `float64 (N*(N-1)/2,)` array in canonical `i<j` order; `scale` is a
finite positive scalar obtained from the shared similarity fit for PA-PPDS.
`compute_pcod_3class_matches` consumes matched `(N,)` camera depths and returns
one `bool (N*(N-1)/2,)` array in the same order.
`compute_joint_acceleration_errors` consumes equal `(T,24,3)` arrays and returns
unreduced `float64 (T-2,24)` vector residuals; exact-frame admission remains
with the protocol orchestrator.

Freeze these identities:

```text
CORRECTED_CROWD_SCHEMA_VERSION = 1
CORRECTED_CROWD_VIEWS = ('GT_VISIBLE', 'C4D_DYCROWD_COMMON')
CORRECTED_CROWD_METRICS = (
    'MPJPE-WORLD', 'T-MPJPE', 'RT-MPJPE', 'PA-MPJPE',
    'SEQ-T-MPJPE-VISRUN', 'SEQ-RT-MPJPE-VISRUN',
    'SEQ-PA-MPJPE-VISRUN', 'SEQ-T-MPJPE-TRACK',
    'SEQ-RT-MPJPE-TRACK', 'SEQ-PA-MPJPE-TRACK',
    'PPDS', 'PA-PPDS', 'PCOD-3C-0.3m', 'OKS-VIS', 'ACCEL-WORLD',
)
```

`Corrected_Crowd_Sequence` is one scene's exact columnar input schema:

```text
schema_version                         exact int == 1
scene_id                              non-empty str
frame_domain                          int64 (F,), strictly increasing
gt_frame_ids                          int64 (G,)
gt_track_ids                          int64 (G,), positive
gt_joints_world_m                     float64 (G,24,3)
gt_coco17_xy_px                       float64 (G,17,2)
gt_visibility_native                  float64 (G,17), values {0,0.5,1}
gt_bbox_xyxy_px                       float64 (G,4)
gt_pelvis_camera_depth_m              float64 (G,)
prediction_frame_ids                  int64 (P,)
prediction_local_track_ids            int64 (P,), non-negative column IDs
prediction_joints_world_m             float64 (P,24,3)
prediction_coco17_xy_px               float64 (P,17,2)
prediction_coco17_camera_depth_m       float64 (P,17)
prediction_pelvis_camera_depth_m       float64 (P,)
prediction_identity_target_gt_rows     int64 (P,), -1 or [0,G)
matched_gt_rows                        int64 (Q,), unique [0,G)
matched_prediction_rows                int64 (Q,), unique [0,P)
common_gt_mask                         bool (G,)
coordinate_frame                      'FIXED_CAMERA_WORLD_EQUIVALENT'
length_unit                           'metre'
camera_depth_axis                     'POSITIVE_Z_AWAY_FROM_CAMERA'
smpl_joint_order                      'SMPL_24'
coco_joint_order                      'COCO_17'
```

`(gt_frame_ids,gt_track_ids)` and
`(prediction_frame_ids,prediction_local_track_ids)` are unique within
`scene_id`; together they are the stable GT and method-local occurrence keys.
`-1` is the only identity-target sentinel and means no GT-present row at that
frame. A matched pair must equal the prediction's non-sentinel identity target.
GT camera intrinsics are absent because no corrected leaf consumes them; the
TPA receipt owns projection provenance.

The input retains GT-present all-zero-visibility rows so the library can derive
both `G_full` and `G_invisible` and validate `P_m` itself. Inactive native
padding is never represented as a prediction occurrence. Identity targets and
accepted association are separate fields, so duplicate/unmatched predictions
remain countable without creating a second match.

`validate_corrected_crowd_sequence` is the one shared fail-fast gate. Every
array in its returned value is an owned bytes-backed read-only copy; frozen
dataclasses alone are not treated as immutable. It checks
all shapes, exact integer/bool fields, unique keys, row bounds, one-to-one
association, same-scene/same-frame identity consistency, fixed literals,
finiteness, positive included projection depth, positive bboxes, common-mask
subset, positive consumed GT pair distance, and the Section 6 alignment
predicates. It does not filter rows or create invalid categories.

`evaluate_corrected_crowd_sequence` validates once, derives the full and common
matched relations, and emits one immutable scene summary. It reuses
`compute_joint_position_errors` and `compute_keypoint_oks_matrix`; their frozen
existing behavior is not changed. It groups frame metrics by matched rows,
pair metrics by frame, sequence metrics by the GT-owned VISRUN/TRACK labels,
and acceleration by exact frame-ID triples. No metric leaf receives F1.

Freeze the immutable scene-summary schema:

```text
Corrected_Crowd_Sequence_Summary
    schema_version                     exact int == 1
    scene_id                           str
    tp, fn, fp                         exact non-negative ints
    metric_sample_sums                 float64 read-only (2,15)
    metric_sample_counts               int64 read-only (2,15)
    accel_exact_consecutive_triple_count int64 read-only (2,)
```

Rows/columns follow the two constant tuples above. Joint-position sums are in
metres, `ACCEL-WORLD` sums in metres/frame², and PPDS/PA-PPDS/PCOD/OKS sums are
dimensionless. Counts are an internal cross-scene reduction contract; only the
specifically approved `accel_exact_consecutive_triple_count` is published as a
named diagnostic, not as a generic support taxonomy.

`reduce_corrected_crowd_summaries` checks unique scene identities, adds counts,
combines sums in lexical `scene_id` order, derives the one global
precision/recall/F1 partition, and never averages scene means. It returns:

```text
Corrected_Crowd_Result
    schema_version                     exact int == 1
    tp, fn, fp                         exact non-negative ints
    precision, recall, f1              finite float
    metric_values                      tuple[2][15] of float | None
    accel_exact_consecutive_triple_count tuple[2] of int
```

`metric_values` is already in display units: joint positions are millimetres,
`ACCEL-WORLD` is millimetres/frame², and dimensionless metrics remain `[0,1]`.
An entry is `None` exactly when its global sample count is zero. Tuple storage
makes the public result immutable without a NaN/magic-value placeholder.

The stable owner also provides `corrected_crowd_summary_to_json`,
`corrected_crowd_summary_from_json`, and `corrected_crowd_result_to_json`.
Their JSON objects carry `schema_version`, explicit view/metric name arrays,
units by metric, and decimal integer counts; readers reject missing, extra,
reordered, non-finite, or version-mismatched content. The TPA worker and parent
must round-trip through these functions and cannot invent a parallel schema.

Frame `T-MPJPE` is computed directly from per-person pelvis-relative joints and
never calls a fitted transform. `SEQ-T` alone flattens the whole declared scope
and calls geometry's mean-translation fit. Frame/sequence RT and PA call the
same geometry fit APIs with different populations. This keeps the two `T`
semantics explicit at the call boundary.

All new public types, constants, leaves, serializers, and protocol functions
are re-exported from `hjlib_evaluation`. Individual functions stay below
roughly two screens; state belongs only in immutable snapshots, while grouping
and arithmetic remain ordinary functions.

### Dataset adapter and operation map

Add the following TPA-owned files:

```text
scene_input.py
    load_scene_inputs(...) moved from gt_mot_scene_worker without behavior change

corrected_adapter.py
    Corrected_Crowd4D_Adapter
    build_corrected_scene_input(...)
    build_common_view_scene_keys(...)

corrected_scene_worker.py
    one method/scene normalization and evaluation worker

corrected_operation.py
    freeze_corrected_common_view(...)
    run_corrected_evaluation(...)
```

Extend the existing `hj-tpa-crowd4d-gt-mot` Typer app with two flat commands:

```text
freeze-corrected-common-view
evaluate-corrected
```

The first command deterministically intersects GT-visible keys with valid
Crowd4D and DyCrowd occurrences under the accepted identity maps, writes the
167,497-key manifest in canonical `(scene, frame_id, track_id)` order, and
binds all source identities. The second requires that immutable manifest; it
does not recompute or shrink it.

`Corrected_Crowd4D_Adapter` composes the existing trusted loader,
`Crowd4D_SMPL_Adapter`, Crowd4D native `idxs`, and accepted DyCrowd mapping.
It reuses the existing frame normalizers rather than reimplementing SMPL or
projection. GT and prediction SMPL-24 are the first 24 named SMPL joints from
the already normalized SMPL-54 arrays. COCO pixels use the existing named
SMPL-54-to-COCO-17 mapping; source projection depth comes from those same
camera-space joints before division.

For VirtualCrowd, the adapter declares the fixed capture-camera coordinate as
the normalized sequence world frame. The audited camera extrinsic is
sequence-constant, so this frame differs from scene world by one fixed rigid
transform and preserves all declared Euclidean/vector quantities. Pelvis depth
is its `+z` coordinate. The receipt records this provenance and the bounded
extrinsic-constancy check; a violation fails before metrics rather than
silently switching coordinate definitions.

Each method/scene runs in a short-lived subprocess so SMPL/native heaps are
released. Workers write only normalized summary JSON inside a transaction
staging root. The parent validates the complete `2 methods x 8 scenes x 2
views` grid, reduces through `hjlib-evaluation`, writes machine-readable
results plus one compact comparison table, publishes the two named
`accel_exact_consecutive_triple_count` values per method, binds the common
manifest and accepted identity evidence, and promotes atomically. Other
`sample_count` values remain internal serialized reduction state rather than
result columns. Author tables and T2/T3 compatibility evidence are read-only
and never overwritten.

### Performance and failure boundaries

SMPL forward remains batched per active frame as in the current adapter. Metric
leaves vectorize joints and person pairs; sequence grouping sorts integer row
indices once per scene. No property/getter performs I/O or scans a directory.
Native loading, hashing, process launch, and result writing remain explicitly
named TPA operations.

Validation precedes reduction. Both corrected commands require a disjoint
explicit `path_failure_root`. Worker launch captures stdout/stderr and one
structured record containing method, scene, exception type, and exact message.
If any worker fails, the success staging tree is deleted by the existing
transaction, then the parent atomically promotes only `failure.json` plus the
captured worker log under `path_failure_root` and re-raises. Failure evidence
has its own schema/status and can never satisfy or occupy `path_output_root`.
A successful operation requires the failure root to remain absent. Smoke tests
verify both no-success-promotion and exact-predicate recovery. There is no
best-effort partial table, implicit epsilon, invalid-row filtering, or automatic
fallback to author semantics.

## Smoke-Test Standard

### `hjlib-evaluation/test_smoke`

Add portable synthetic coverage for:

1. geometry-owner translation, reflection-disabled rigid, and positive-scale similarity fits,
   including a reflected cloud and zero-spread rejection;
2. frame SMPL-24 micro reductions and the equality of unaligned sequence/frame
   populations on identical support;
3. GT-owned VISRUN versus TRACK grouping, prediction holes that do not split a
   run, and exact-consecutive acceleration that never bridges a frame gap;
4. vector ACCEL residuals that differ from magnitude-only residuals;
5. unordered pair enumeration, PPDS clipping, PA-PPDS two-person rotation
   non-uniqueness with unique scale, `V_x/C` failures, and no-pair `None`;
6. PCOD classes at `-0.3`, `+0.3`, and just outside both boundaries;
7. native `0/0.5/1` visibility mapping, frozen COCO sigma/order, half-open bbox
   area, and equality of `0.5/1` inclusion weights;
8. non-finite points, non-positive included projection depth, non-positive bbox,
   duplicate keys, cross-frame matches, and non-injective association all
   fail-fast without filtering;
9. the full `G/P/M` partition: visibility-excluded mapped predictions are out
   of scope, while unmapped, absent-GT, and duplicate occurrences remain FP;
10. common view emits geometry only and cannot change global completeness;
11. cross-scene sufficient-statistic reduction equals a direct concatenated
    micro mean and returns `None` only for an empty metric population;
12. all existing joint-error, OKS, trajectory-residual, and harness smoke tests
    remain unchanged and passing.

New topic files expose `smoke_test_corrected_crowd_*()` entry functions and are
explicitly invoked by `test_smoke/test_all_func.py`; pytest discovery alone is
not the master-runner contract.

### `hj-tpa-crowd4d/test_smoke`

Add portable fixtures for both native identity modes and cover:

1. extraction of SMPL-24, COCO-17 pixels/source depth, pelvis depth, native
   visibility, and fixed frame IDs into the stable normalized schema;
2. inactive `track_flag=False` buffers never becoming prediction occurrences;
3. deterministic common-manifest key order/digest and the frozen count check;
4. altered/missing common manifest, accepted DyCrowd mapping, prediction,
   label, or coordinate-provenance evidence rejecting the run;
5. worker failure preventing promotion and a complete synthetic transaction
   producing the exact method/scene/view grid;
6. CLI help for both new flat commands.

New TPA topic files likewise expose `smoke_test_corrected_*()` and are added to
that repository's explicit `test_smoke/test_all_func.py` call list.

Portable smoke does not claim real artifact correctness. After implementation,
the existing explicit-path real-data gate first freezes and verifies the common
manifest, then runs the corrected operation once. This is the only stage that
may produce corrected numeric results.

## Migration Plan

1. Extend `hjlib-geometry.registration` with rigid/similarity fit/apply, update
   its registration design/usage/public exports, and pass its smoke/strict
   typing gates.
2. Add and verify the two evaluation metric leaves plus normalized data/result
   dataclasses in `hjlib-evaluation`; add the direct geometry pin without
   altering existing leaf behavior.
3. Add the per-scene evaluator and cross-scene reducer, then re-export the new
   public surface and update usage/design docs.
4. Move `load_scene_inputs` to the TPA noun-owned module and prove both existing
   GT-MOT worker behavior and smoke remain unchanged.
5. Add the corrected adapter, common-manifest operation, workers, CLI commands,
   transaction receipts, and portable tests.
6. Run strict Pyright and portable smoke in leaf-first dependency order:
   `hjlib-geometry`, then `hjlib-evaluation`, then `hj-tpa-crowd4d`.
7. Freeze the real common-view manifest and check its expected 167,497 unique
   keys before any corrected metric reduction.
8. Run the corrected two-method evaluation once, reconcile schema/counts and
   named invalid predicates, and only then promote results into Campaign 03
   evidence. Any newly observed invalid class returns to protocol review rather
   than being silently handled.

Rollback is additive: before promoted evidence, remove only the new corrected
modules/commands and restore the small `scene_input` move. Frozen T2 author
parity and T3 GT-MOT compatibility paths, APIs, results, and receipts are never
migrated in place.

## Modification History

- 2026-08-15: Created the task-specific Layered Design residence from the
  completed attended requirements freeze and drafted the Mathematical
  Architecture. Dedicated review is pending; implementation is not authorized.
- 2026-08-15: Accepted the first Mathematical Architecture review. Closed its
  two Critical findings by making camera depth/projection depth explicit and
  making invalid OKS projection fail-fast. Closed the common-view completeness,
  conditional pair-denominator, PA-PPDS degeneracy, and OKS area/sigma findings.
  The focused re-review confirmed those closures and identified one remaining
  prediction-universe boundary.
- 2026-08-15: Closed the focused-review concern by freezing the full
  scene/frame domain and excluding only predictions specifically mapped to
  visibility-excluded GT keys. Unmapped, absent-GT, wrong-ID, and duplicate
  occurrences inside that domain remain eligible FP. The final focused review
  accepted the Mathematical Architecture with zero Critical and zero Concern.
- 2026-08-15: Drafted Code Architecture, Smoke-Test Standard, and Migration
  Plan with the stable-library/TPA boundary, frozen common-manifest operation,
  fail-fast validation, and short-lived scene workers. Dedicated Code
  Architecture review is pending; implementation has not started.
- 2026-08-15: Accepted the first Code Architecture review. Moved generic
  registration to the existing `hjlib-geometry` owner; froze all normalized,
  summary, result, serialization, unit, and immutable-array contracts; added a
  separate failure-evidence transaction; exposed only the approved exact
  consecutive-triple count; and bound new smoke topics into both explicit
  master runners. Focused math review accepted the frame/sequence translation
  split, and focused Code Architecture review accepted all dispositions with
  zero Critical and zero Concern. Its non-blocking deterministic-order note is
  resolved by lexical `scene_id` reduction.
- 2026-08-15: Implemented the leaf-first generic facilities. `hjlib-geometry`
  rigid/similarity registration passed 100 portable tests, master smoke, and
  strict Pyright and was committed as `b4594ad`. `hjlib-evaluation` now owns
  the immutable versioned schema, pair/ACCEL leaves, per-scene evaluator,
  cross-scene reducer, JSON round trip, public exports, and 37-test portable
  gate. No corrected real-data evaluation has run.
