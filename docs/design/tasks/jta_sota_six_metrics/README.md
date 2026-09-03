# JTA SOTA Six-Metric Reducer Task Design

## Requirements

Provide one method-neutral semantic owner for the six metrics required by the
JTA SOTA Finetune Rule: Root Error, MPJPE, T-MPJPE, RT-MPJPE, PA-MPJPE, and
fitted-target OKS. The reducer consumes paired occurrence identities and
predicted/GT joints; it owns no model, training loop, dataset sampling, file
format, checkpoint policy, or artifact residence.

The primary profile uses the canonical endpoint order
`[L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle]`, with
SMPL indices `[16,17,18,19,20,21,1,2,4,5,7,8]` and sigma vector
`[.079,.079,.072,.072,.062,.062,.107,.107,.087,.087,.089,.089]` in that exact
order. The left/right native SMPL hip midpoint is root. Three-dimensional
inputs are complete SMPL-24 or SMPL-54 arrays; endpoint XY inputs are already
selected in the canonical twelve-joint order, while target area input is the
complete SMPL-24 fitted XY array. All arrays share the same occurrence row axis
and paired IDs. Inputs may be metres or millimetres and must declare their
coordinate frame. Two-dimensional arrays must share one per-occurrence
coordinate system. No hash, digest, association, or detection semantics enter
this reducer.

## Mathematical Architecture

For paired finite endpoint arrays `X,Y in R^(N x J x 3)`, roots `x,y` are the
mean left/right hip. Let `R,t` be the reflection-disabled least-squares rigid
fit from each prediction to GT, and `s,R,t` the reflection-disabled,
strictly-positive-scale similarity fit. The accumulated distance sums are:

```text
root_sum  = sum_i ||x_i - y_i||
mpjpe_sum = sum_i,j ||X_i,j - Y_i,j||
t_sum     = sum_i,j ||(X_i,j-x_i) - (Y_i,j-y_i)||
rt_sum    = sum_i,j ||R_i X_i,j + t_i - Y_i,j||
pa_sum    = sum_i,j ||s_i R_i X_i,j + t_i - Y_i,j||
```

Root uses denominator `N`; the four joint metrics use denominator `N*12`.
Every call validates unique paired occurrence IDs within its batch. Distance
sums are accumulated in float64 and normalized to millimetres before a sums
value is constructed, so metre and millimetre inputs cannot be mixed during
addition. Empty, non-finite, identity-mismatched, within-batch duplicate-ID, or
zero-spread occurrences are terminal. Zero spread means either prediction or
GT selected endpoints have exactly non-positive float64 centered squared norm;
failure is per occurrence and rejects the batch. Rigid and similarity fits
reuse `hjlib-geometry`; reflection is not allowed and similarity scale must be
positive.

Additive sums carry the exact metric profile, coordinate frame, fixed joint
count, millimetre-unit tag, occurrence count, and scalar denominators. Addition
requires the profile, frame, joint count and unit tag to match; scalar sums,
occurrence count and denominators are added. Every value enforces
`root_denominator=N`, `joint_denominator=12N`, and `oks_denominator=N` before
and after addition. It does not claim to detect an occurrence repeated across
separate batches; the caller must bind the ordered occurrence population and
prove disjoint/exhaustive batches before epoch finalization.

For endpoint XY prediction `u` and fitted target `v`, let `A_i` be the strictly
positive axis-aligned envelope area of all 24 fitted target XY joints. With the
reviewed twelve JTA endpoint sigmas, paired OKS is:

```text
oks_i = mean_j exp(-||u_i,j-v_i,j||^2 / (8 sigma_j^2 A_i))
```

The OKS sum uses denominator `N`. Because numerator distances and area scale
quadratically, a shared positive uniform scale plus translation does not change
the value; planar rotation is not part of this invariance claim because the
reference envelope is axis-aligned.
The profile name is `jta12_fitted_all_valid_v1`; it is not raw-JTA visibility
OKS and performs no person association.

## Code Architecture

- `src/hjlib_evaluation/jta_sota_metric_reducer.py` owns immutable
  `JTA_SOTA_Metric_Sums` and `JTA_SOTA_Metric_Result`, input validation,
  endpoint/root selection, per-occurrence registration, linear paired OKS,
  additive batch accumulation, and finalization.
- The public kernel is `compute_jta_sota_metric_sums(pred_joints, gt_joints,
  pred_occurrence_ids, gt_occurrence_ids, pred_endpoint_xy,
  gt_endpoint_xy, gt_smpl24_xy, *, joint_layout, unit_world,
  pred_coordinate_frame, gt_coordinate_frame, image_coordinate_frame)`.
  `JTA_SOTA_Metric_Sums.plus(other)` adds sufficient statistics only, and
  `finalize_jta_sota_metric_sums(sums)` returns the six averages. The public
  `validate_jta_sota_occurrence_partition(expected_ids,
  batch_occurrence_ids)` owns the generic ordered disjoint/exhaustive caller
  gate. The second argument is an ordered sequence of one-dimensional int64 ID
  vectors, one per optimizer step, and the concatenation must equal
  `expected_ids` exactly. Native workers record those vectors; the shared
  Python 3.12 coordinator invokes this one public gate at epoch finalization,
  so TPAs do not implement a second population rule.
- Existing `validate_occurrence_inputs()` remains the full-layout/identity gate.
  The reducer directly reuses `SMPL54_ENDPOINT_INDICES`, `JTA_ENDPOINT_NAMES`,
  and `JTA_ENDPOINT_OKS_SIGMAS` from `jta_person_detection_data.py`; it does not
  copy a second canonical profile. Existing rigid/similarity fit-and-apply
  functions remain the registration owner and their math is not copied.
- `keypoint_oks.py` adds public `compute_paired_keypoint_oks(...)`, the linear
  `O(N*J)` identity-paired counterpart to its association matrix leaf. The
  SOTA reducer calls that leaf and never allocates an `N x N` matrix.
- Public exports added to `hjlib_evaluation.__init__` are
  `JTA_SOTA_METRIC_PROFILE`, `JTA_SOTA_Metric_Sums`,
  `JTA_SOTA_Metric_Result`, `compute_jta_sota_metric_sums`,
  `compute_paired_keypoint_oks`, `finalize_jta_sota_metric_sums`, and
  `validate_jta_sota_occurrence_partition`. TPAs may use the NumPy reducer for
  epoch validation/test in the Python 3.12 coordinator.
- A TPA native environment may implement an equivalent detached torch batch
  reducer for per-step logging. Such a reducer is method-runtime code; no
  shared TPA base class is introduced.
- Native parity is one campaign-owned cross-environment gate at
  `hjlib-experiments/campaigns/10_jta_sota_grouprec_comhr/task_six_metric_evaluation/verify_native_metric_parity.py`.
  The Python 3.12 coordinator creates one deterministic, pickle-free `.npz`
  fixture, runs the shared reducer, invokes each TPA's data-free native CLI on
  that same fixture, and compares floating scalar sums at `rtol=1e-5`,
  `atol=1e-6`; integer counts and denominators must be exactly equal. Native
  outputs are pickle-free `.npz`; the fixture
  and outputs are task-local transient evidence and use no hash.

## Smoke-Test Standard

`test_smoke/test_jta_sota_metric_reducer.py` exposes
`smoke_test_jta_sota_metric_reducer()` and is called by
`test_smoke/test_all_func.py`. Synthetic gates verify exact endpoint/index/sigma pairing and full-3D,
endpoint-XY and full-target-XY shapes, then distinguish identical,
translation-only, rotation-only,
positive-scale, reflected, and degenerate inputs. They verify exact occurrence
and joint denominators, additive short-batch accumulation, unit conversion,
metadata mismatch, within-batch identity/order failures, caller-side
cross-batch duplicate detection, perfect/displaced OKS, positive-uniform-scale
plus translation invariance and non-invariance to a rotated axis-aligned
envelope, nonpositive area failure, and parity of RT/PA with the reviewed
geometry leaves.

## Migration Plan

The existing `SMPL_Joint_Occurrence_Result` and MPJPE/T-MPJPE APIs are not
changed. The new reducer is additive. Existing receipts are never recomputed or
rewritten merely because more metrics become available.

## Modification History

- 2026-09-01: Initial task design frozen before implementation.
- 2026-09-01: Mathematical review fixed endpoint/sigma order, full input
  shapes, accumulation metadata/unit semantics, caller-owned cross-batch
  uniqueness, exact zero-spread failure, and the limited OKS invariance claim.
- 2026-09-01: Mathematical re-review clarified that identity metadata matches
  while sums/counts/denominators add, with denominator equations enforced.
- 2026-09-01: Final mathematical re-review found no remaining finding.
- 2026-09-01: Code Architecture review froze the public kernel/addition/
  finalization and population-gate APIs, reused the existing JTA constants and
  a paired OKS leaf, named the smoke/master-runner seam, and assigned one
  campaign-owned cross-environment native parity gate.
- 2026-09-01: Code Architecture re-review made counts/denominators exact in
  parity, froze the ordered batch-ID sequence and coordinator-owned population
  gate, and added the paired OKS leaf to the public export list.
- 2026-09-01: Final Code Architecture re-review found no remaining finding.
