# VirtualCrowd Density-Balanced RCR Ground

## Status

- State: KDE single-arm and Cartesian evaluation complete.
  All 12 configs, six real-input observation populations, and 167,243-row
  evaluations were written and independently reconstructed from source inputs.
  The earlier kNN implementation and real operation are retained as historical
  exploratory evidence; they are not Cartesian density alternatives.
- Task owner: `hjlib-evaluation`.
- Generic algorithm owner: `hjlib-ground-solver`.
- Execution: Sequential Task Level 3, seventeen ordered steps.
- Scope: one task; this is not a new campaign.

The completed task's current result, the post-completion Crowd4D/GeoCalib
investigation, and the unactivated next-task boundary are summarized in
[GeoCalib handoff status](geocalib_handoff_status.md). No GeoCalib TPA or
inference operation has been activated by that report.

## KDE Single-Arm and Prepared Cartesian Extension

The user rejected fixed `k=16/32/64` as the formal density axis and selected a
standard Gaussian KDE family instead. Before any Cartesian expansion, this
extension implements and evaluates exactly one selection arm:

```text
confidence > 4.0
ankle-distance / bbox-width < 0.20
density mode in {filtered_unweighted, density_kde_scott_loo}
H_prior = 1.35 m
```

Both density modes use the same reviewed 17,992 ordered observations and the
same frozen 167,243-person-frame ground-effect support. The KDE result was
shown to and accepted by the user before confidence or ankle thresholds were
crossed.

The completed single arm gives a global mean same-ray error of `16.457570 m`
versus `18.896103 m` unweighted on 167,243 rows, an improvement of 12.90%.
Five scenes improve and three worsen. Normal-oracle and distance-only global
means both worsen (`8.375987 -> 9.123947 m` and
`21.030782 -> 25.014907 m`), so the combined gain contains error cancellation.
This numerical result is the accepted checkpoint used to prepare the Cartesian
grid.

The prepared grid is the exact product
`confidence>{4.0,4.5,5.0}` × `ankle/bbox ratio<{0.15,0.20}` ×
`{filtered_unweighted,density_kde_scott_loo}`. It contains six observation
populations and 12 configs. The real dry-run totals are `9,427`, `17,992`,
`6,770`, `13,524`, `3,839`, and `8,326` in confidence-major, ratio-minor
order. Preparation reads only dataset labels and tracked detections; it creates
no result root and calls neither KDE nor RCR. The later explicitly authorized
execution completed all 12 configs.

Global person-frame-micro same-ray means are:

| Confidence | Ankle ratio | Unweighted (m) | KDE (m) | KDE change | KDE improved scenes |
|---:|---:|---:|---:|---:|---:|
| `>4.0` | `<0.15` | 21.5923 | 18.2115 | -15.66% | 5/8 |
| `>4.0` | `<0.20` | 18.8961 | 16.4576 | -12.90% | 5/8 |
| `>4.5` | `<0.15` | 25.8974 | 21.4003 | -17.37% | 6/8 |
| `>4.5` | `<0.20` | 17.4740 | 15.8484 | -9.30% | 4/8 |
| `>5.0` | `<0.15` | 21.8608 | 18.3688 | -15.97% | 7/8 |
| `>5.0` | `<0.20` | 17.1769 | 15.7277 | -8.44% | 5/8 |

The raw best config is `confidence>5.0, ankle/bbox<0.20, KDE`, but its
smallest scene has only 256 observations. Across all six populations KDE lowers
the combined mean while both support-weighted normal-oracle and distance-only
means worsen. The result therefore retains explicit low-support and systematic
error-cancellation limitations; it is not evidence that both plane components
improved.

## Named Baseline Registry

### `baseline001`

`baseline001` is an immutable alias for the exact frozen result below. It does
not mean "the latest best". Any later change to selection, density, height,
detections, solver, or evaluation support requires a new baseline identifier.

```text
underlying config:
  conf_gt_5p0__ankle_lt_0p20__density_kde_scott_loo
dataset:
  released VirtualCrowd, eight canonical scenes
detections:
  existing GT-MOT RTMLib Tracked_Scene, no detection rerun
observation joints:
  shoulder midpoint (5,6) -> ankle midpoint (15,16)
selection:
  min shoulder/ankle score > 5.0
  ankle distance / bbox width < 0.20
  no sampling
density:
  exact leave-one-out Gaussian KDE on provisional-ground 2D coordinates
  Scott bandwidth; clipped inverse-density weights [0.25,4.0], mean-normalized
scale prior:
  H_prior = 1.35 m, the shoulder-to-ankle proxy for an approximately 1.7 m person
solver observations:
  8,326 total; scene counts 256,810,1099,761,1104,674,560,3062
evaluation:
  same frozen 167,243 GT-MOT person-frame rays
  same-ray GT-ground versus estimated-ground 3D intersection error, metre
frozen global mean:
  15.727719674759953 m
artifact config directory:
  result_kde_cartesian/conf_gt_5p0__ankle_lt_0p20__density_kde_scott_loo
```

Limitations are part of the baseline meaning: its smallest scene contains only
256 solver observations, and its combined gain coexists with worse
support-weighted normal-oracle and distance-only means, indicating error
cancellation. The alias must not be quoted without these qualifications.

### Post-completion GT-K/GT-normal height diagnostic

A later oracle diagnostic fixed both VirtualCrowd GT intrinsics and the GT
ground normal, retained the exact `baseline001` observations, KDE weights,
distance grid, and 167,243-row evaluation support, and changed only `H_prior`.
Among `1.200, 1.225, 1.250, 1.275, 1.300, 1.325 m`, the lowest global
person-frame-micro mean occurred at `H_prior=1.250 m`:

```text
H_prior = 1.250 m
mean     = 1.8723156219237906 m
median   = 1.8245129783826275 m
std      = 0.899090688322 m
```

At `1.250 m`, all eight scene distance ratios were between `0.973` and
`1.013`. This value is an evaluation-set oracle diagnostic for the effective
shoulder-midpoint-to-ankle-midpoint proxy, not an accepted anthropometric prior
and not a change to immutable `baseline001`, which remains `H_prior=1.35 m`.
The external sweep residence is
`tmp/2026-08-19/task/virtualcrowd-gt-normal-h-sweep/summary.json`.

Per-scene `baseline001` plane diagnostics use sign-aligned unit-normal plane
coefficients `n^T X + D = 0`. Here `Delta D = D_pred-D_gt`; positive means the
estimated plane is farther from the camera origin. `Normal-only` uses the
predicted normal with its support-derived oracle D. `D-only` uses the GT normal
with predicted D. Both last columns are person-frame-micro same-ray means and
are diagnostic counterfactuals, not additive components.

| Scene | Solver N | Normal angle (deg) | GT D (m) | Pred D (m) | Delta D (m) | D ratio | Normal-only (m) | D-only (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scene1` | 256 | 0.363 | 33.325 | 34.500 | +1.175 | 1.035 | 0.493 | 4.324 |
| `scene1_view2` | 810 | 2.223 | 5.959 | 9.000 | +3.041 | 1.510 | 5.981 | 41.961 |
| `scene2` | 1,099 | 1.245 | 29.621 | 29.400 | -0.221 | 0.993 | 3.304 | 1.104 |
| `scene2_view2` | 761 | 4.132 | 6.141 | 12.800 | +6.659 | 2.084 | 13.368 | 99.094 |
| `scene3` | 1,104 | 0.395 | 16.750 | 18.400 | +1.650 | 1.099 | 0.835 | 11.873 |
| `scene3_view2` | 674 | 1.918 | 9.578 | 8.000 | -1.578 | 0.835 | 6.447 | 11.753 |
| `scene4` | 560 | 2.938 | 54.682 | 49.500 | -5.182 | 0.905 | 9.458 | 18.213 |
| `scene4_view2` | 3,062 | 0.952 | 8.573 | 6.900 | -1.673 | 0.805 | 14.920 | 24.077 |

The active KDE Mathematical Architecture, Code Architecture, Smoke-Test
Standard, and cost contract live in [kde_density.md](kde_density.md). The older
[density_intermediate.md](density_intermediate.md) remains a historical record
of the completed kNN experiment. Its kNN variants are sensitivity evidence,
not formal KDE alternatives.

## Requirements

This task tests whether spatially repeated human detections dominate the RCR
ground estimate and adds a generic inverse-density weighting option to the
ground solver. The VirtualCrowd operation uses the existing GT-MOT
`Tracked_Scene` detections and the existing same-ray ground-effect evaluation.
It does not rerun detection, alter the frozen comparison support, or sample a
fixed number of observations in its primary variants.

For every scene, form the complete canonical person-frame population satisfying
both conditions:

1. `min(score[j] for j in (5, 6, 15, 16)) > 4.0`;
2. `||kp[15,xy] - kp[16,xy]||_2 / bbox_width < 0.20`.

`bbox_width` is the raw tracked-box right-minus-left extent. The inequality is
strict. Missing detections are outside the candidate population. Nonfinite
values, nonpositive bbox width, and degenerate top-to-bottom observations fail
the scene rather than being silently removed after selection. The reviewed
real-data population is 17,992 observations across eight scenes, with counts
`885, 1397, 3370, 1486, 1638, 1220, 3046, 4950` in canonical scene order.

The density path first obtains an unweighted provisional ground normal. It then
maps every retained ankle-midpoint ray to a common provisional ground plane,
constructs an inspectable 2D ground-coordinate density intermediate, and uses
inverse-density weights in the normal fit. Ground truth is prohibited from the
density calculation and from solver fitting. The same filtered observations
must feed every compared solver variant; only the weighting rule and its
explicit parameter may differ.

Required real variants are:

- filtered unweighted;
- inverse-density kNN with `k=16`;
- inverse-density kNN with `k=32`, the candidate headline;
- inverse-density kNN with `k=64`.

All variants retain `H_prior=1.35 m` and the same RCR distance search contract.
Each result reports the estimated camera-frame plane, angular error against the
GT ground normal, estimated-to-GT plane-distance ratio, actual RCR objective,
density diagnostics where applicable, and the existing same-ray ground-effect
distribution in metres on the exact frozen 167,243-person-frame support.

## Owner and Boundary

`hjlib-ground-solver` owns the method-neutral numerical capability:

- provisional-plane ground-coordinate construction from image observations;
- kNN density and normalized inverse-density weights;
- the immutable density intermediate representation;
- weighted RCR normal fitting and its numerical validation.

`hjlib-evaluation` owns the concrete experiment:

- VirtualCrowd joint/bbox selection and canonical identities;
- construction of the named variants;
- scene orchestration and plain result files;
- GT-only diagnostic comparison and same-ray metre evaluation.

The dependency remains `hjlib-evaluation -> hjlib-ground-solver`. Dataset names,
GT-MOT assumptions, joint indices, thresholds, scene paths, and evaluation GT
must not enter the generic solver package. The density intermediate contains
only numerical observations and diagnostics; it is not an evaluation schema or
artifact-signing layer.

## Layered Design Sequence

1. Establish owner, boundary, and tracked residence.
2. Freeze and independently review the density intermediate representation.
3. Implement and validate that generic density intermediate.
4. Freeze and review weighted RCR and evaluation architecture.
5. Implement weighted solving and the named evaluation variants.
6. Run all real variants and the corrected same-ray evaluation.
7. Land usage/design documentation, complete final reviews, and close.

The density Mathematical and Code Architecture live in
[density_intermediate.md](density_intermediate.md).

## Weighted RCR Mathematical Architecture

Let each image top-bottom observation define the existing unnormalized
homogeneous image-line row

```text
l_i = cross(xb_i, xt_i).
```

The legacy two-pass angular trim is run unweighted, with `prop_filter=0.24`
and `times_filter=2`, and produces one original-row Boolean mask `m`. Ties at
the threshold remain included. The mask is identical for unweighted and every
density variant. This deliberately separates robust angular support from
spatial-density contribution; density weights cannot cause a row to enter or
leave the trim.

For an unweighted fit, the vertical vanishing point is the right singular
vector associated with the smallest singular value of rows `l_i` where `m_i`
is true. For a weighted variant, it is instead the nullspace fit of

```text
A_weighted[i,:] = sqrt(w_i) * l_i, for rows where m_i is true.
```

All weights are finite and strictly positive. Multiplying every weight by one
positive scalar must not change the fitted vanishing point. The existing
unnormalized `l_i` magnitude remains: density balancing is layered on top of
the inherited image-line-length influence. Row normalization, track balancing,
normal/density fixed-point iteration, and random sampling are explicitly not
part of this task.

After `ground_normal=normalize(solve(K, vanishing_point))`, the existing RCR
distance search evaluates the two per-observation dimensionless residuals
`losses_mod_i` and `losses_pixel_i`. The unweighted branch preserves the exact
arithmetic path `mean(losses)`. A weighted branch uses

```text
loss_mod   = sum_i(w_i * losses_mod_i)   / sum_i(w_i)
loss_pixel = sum_i(w_i * losses_pixel_i) / sum_i(w_i)
objective  = loss_mod + loss_pixel.
```

The full protocol-selected population participates in D search; the angular
mask applies only to normal fitting, matching inherited RCR behavior. Thus the
same density weight governs each observation's contribution to both normal and
distance estimation while preserving the legacy distinction between robust
normal support and all-row scale support. The returned objective is the actual
weighted objective for weighted variants and the actual unweighted objective
for baseline; it is diagnostic, not a cross-method metric.

The one-pass solve order is exact:

1. solve the filtered-unweighted ground once;
2. use its unit normal as the density provisional normal;
3. compute k=16, 32, and 64 density records independently on all selected
   bottom observations;
4. re-solve RCR once per k with that record's normalized weights.

All four variants use identical ordered observations and `H_prior=1.35 m`.

## Evaluation Mathematical Architecture

The concrete observation filter is applied before any solver call. For a
present detection row, define

```text
quality = min(score[5], score[6], score[15], score[16])
ankle_bbox_ratio = ||kp[15,xy] - kp[16,xy]||_2 / (bbox_right - bbox_left).
```

Retain exactly `quality > 4.0` and `ankle_bbox_ratio < 0.20`. Bbox width must be
finite and positive. Selected rows remain canonical `(frame_index, person_id)`
and are never sampled. The per-scene reviewed counts are a real-run gate.

For GT and predicted planes, normalize complete homogeneous coefficients and
then align the complete predicted plane sign:

```text
(n_gt,d_gt) = (n_gt,d_gt) / ||n_gt||
(n_pred,d_pred) = (n_pred,d_pred) / ||n_pred||
if n_pred^T n_gt < 0:
    (n_pred,d_pred) = -(n_pred,d_pred)
```

All coefficients must be finite with nonzero normals and `|d_gt|>1e-12 m`.
Every following diagnostic uses only these normalized/sign-aligned
coefficients. Report

```text
normal_angle_deg = acos(clip(n_pred^T n_gt, -1, 1)) * 180/pi
distance_ratio = abs(d_pred) / abs(d_gt).
```

The headline evaluation remains the existing complete-support same-ray error
in metres. Before decomposition, each stored GT intersection must satisfy the
normalized GT plane with `|n_gt^T X_gt_i+d_gt| <= 1e-8 m`, in addition to the
existing pixel/K ray roundtrip. To separate normal and distance failure, for
every frozen support ray `r_i=solve(K,[u_i,v_i,1]^T)` and stored GT intersection
`X_gt_i`, define the fixed-normal per-ray distance coefficient

```text
d_i = -n_pred^T X_gt_i
a_i = ||r_i||_2 / |n_pred^T r_i|.
```

Require finite positive ray norms, normalized cosine
`|n_pred^T r_i|/||r_i||_2 > 1e-10`, finite `d_i`, and finite positive geometric coefficients
`a_i`. Any violation rejects the complete scene/variant. These geometric coefficients are unrelated to the solver's density
weights. The `oracle_distance_m` is the deterministic lower weighted median of
`d_i` under `a_i`: lexically sort by `(d_i, canonical row position)` and choose
the first value whose inclusive cumulative weight is greater than or equal to
`0.5*sum(a)`. This minimizes
`sum_i a_i |d-d_i|`, exactly the summed same-ray Euclidean error for fixed
`n_pred`. The same-ray error distribution from plane
`(n_pred, oracle_distance_m)` is the normal-only residual floor. Conversely,
the error distribution from `(n_gt,d_pred)` is the distance-only diagnostic.
Both diagnostic planes must produce finite positive forward intersections on
the complete support or the scene/variant fails. Neither oracle diagnostic
enters the solver or density weights.

All error reductions are person-frame micro over the exact 167,243 ordered
support rows, use metres, population standard deviation, and NumPy linear
quantiles. Any invalid ray/intersection rejects the complete scene/variant; no
row is dropped.

## Weighted RCR Code Architecture

`hjlib-ground-solver` extends existing RCR APIs by appending every new optional
argument after all existing positional parameters:

- `get_KN(xb, xt, flag_ret_A=False, observation_weights=None)` applies
  `sqrt(weight)` only to SVD rows and preserves the raw returned line matrix;
- `get_KN_with_filter(xb, xt, prop_filter=0.2, times_filter=2,
  flag_ret_filtered_result=False, observation_weights=None)` always derives
  trim membership unweighted, then applies weights only in the final fit;
- `get_projection_loss(xb_gt, xt_gt, xt_pred,
  flag_ret_filter_mask=False, ratio_filter_keep=0.9,
  observation_weights=None)` accepts a torch `[N]` tensor on the same
  dtype/device as the losses;
- `solve_D_search(..., H_prior=1.5, D_init=10.0,
  flag_ret_filter_mask=False, ratio_filter_keep=0.9,
  device=torch.device('cpu'), *, distance_min=-5.0, distance_max=80.0,
  distance_step=0.1, observation_weights=None)` preserves historical positional
  slots, accepts NumPy float64 `[N] | None`, validates it,
  and converts it exactly once to torch float32 on `device`; and
- `solve_ground_param_by_top_bottom_given_K(..., H_prior=1.35, D_init=10.0,
  device_solve=cpu, flag_opt=False, *, distance_min=-5.0,
  distance_max=80.0, distance_step=0.1, observation_weights=None)` preserves
  historical positional binding, validates NumPy float64 `[N] | None`, and
  forwards it to normal and D fitting.

Each still-public low-level entry independently rejects non-array/wrong-dtype,
wrong-shape, nonfinite, and nonpositive weights. During each unweighted angular
trim pass, `xb_tmp`, `xt_tmp`, and `weights_tmp` are shortened by the same
current-row Boolean mask; the final weighted SVD therefore uses weights aligned
to retained original rows. D search separately receives the untouched complete
original `[N]` vector. A sentinel-weight smoke freezes this alignment.

No new stateful solver class is introduced. Existing public calls without
weights retain their signatures' meaning and unweighted arithmetic branch.

`hjlib-evaluation/src/hjlib_evaluation/ground_estimation_protocol.py` appends
`bottom_pair_bbox_width_ratio: float64[N] | None = None` to the public immutable
`Ground_Observation_Set`, preserving existing five-field positional
construction. `take/select/sample` propagate it exactly. Collection appends
`maximum_bottom_pair_bbox_width_ratio: float | None = None`: when absent,
confidence-only and KP-only callers retain old behavior; when present, bbox
modality and finite positive width are required and the strict generic
bottom-pair ratio is applied. “Ankle” remains only a VirtualCrowd script name.

The existing three-argument `Ground_Estimator` and
`estimate_ground_from_observations` injection seam remain unchanged. The new
operation uses `functools.partial` or an explicit closure to capture one weight
array for the top-level solver; it does not infer callable capability from a
caught `TypeError`.

The same generic evaluation module owns and package-root re-exports:

- frozen `Ground_Plane_Diagnostics(normalized_pred_plane_camera_abcd,
  normalized_gt_plane_camera_abcd, normal_angle_deg, distance_ratio)`;
- frozen `Ground_Effect_Decomposition(oracle_distance_m,
  normal_oracle_error_m, distance_only_error_m)` with owned read-only arrays;
- pure `compute_ground_plane_diagnostics(pred_plane, gt_plane)`;
- pure `lower_weighted_median(values, positive_weights)`; and
- pure `compute_ground_effect_decomposition(support, K, pred_plane, gt_plane)`.

The decomposition reuses `compute_same_ray_ground_errors` for both diagnostic
planes and `summarize_ground_errors` for reductions; ray intersection and
summary logic are not copied into the dataset script.
`Ground_Plane_Diagnostics.__post_init__` validates, owns, and write-protects
both float64 `[4]` plane arrays and validates finite scalar ranges. The
decomposition record applies the same owned read-only array rule. The public
weighted median rejects nonfinite values and nonfinite/nonpositive weights.

The new standalone operation is
`script/evaluate_virtualcrowd_density_balanced_rcr_ground.py`. It accepts the
same four explicit roots as the earlier RCR operation, constructs exactly the
four named variants

```text
filtered_unweighted
density_knn_k16
density_knn_k32
density_knn_k64
```

and writes one plain `summary.json` plus one numeric NPZ per scene/variant.
This is intentionally a repo-local dataset operation under `script/`, matching
the user-selected boundary and the earlier RCR operation. It is not a reusable
package CLI and is therefore not registered in `[project.scripts]`; reusable
selection, diagnostic, density, and solver APIs remain in the two `src/`
packages. Portable help coverage executes the actual repo-local script.
The output root must not exist; the operation never overwrites, merges, or
deletes a prior result. The exact common NPZ keys are:

```text
selected_frame_index                    int64[N]
selected_person_id                      int64[N]
top_xy_px                               float64[N,2]
bottom_xy_px                            float64[N,2]
quality                                 float64[N]
bottom_pair_bbox_width_ratio            float64[N]
plane_camera_abcd                       float64[4]
rcr_objective                           float64 scalar
effect_frame_id                         int64[M]
effect_gt_track_id                      int64[M]
ground_effect_error_m                   float64[M]
normal_oracle_error_m                   float64[M]
distance_only_error_m                   float64[M]
normalized_pred_plane_camera_abcd       float64[4]
normalized_gt_plane_camera_abcd         float64[4]
normal_angle_deg                        float64 scalar
distance_ratio                          float64 scalar
oracle_distance_m                       float64 scalar
```

Density variants additionally use the density record's exact public field
names as NPZ keys, with integer/scalar fields stored as numeric zero-dimensional
arrays. The unweighted variant has none of those density keys. `summary.json`
contains every scalar above plus the standard summary of each of the three raw
error arrays, per scene and globally. This is sufficient for one independent
reload/re-reduction without inventing a schema registry, receipt, or hashes.

The operation is scene-major. It loads one tracked scene and one support set,
builds at most three small kNN records, and runs four CPU RCR solves. At the
reviewed maximum `N=4,950`, density is `O(N log N + Nk)` and each D solve remains
the inherited fixed-grid `O(850N)` computation. Each oracle diagnostic adds one
`O(M log M)` weighted-median sort, and the operation writes 32 NPZ files with
raw support/error arrays. No multi-process/GPU orchestration is expected for
this workload.

## Smoke-Test Standard

In addition to the separately reviewed density tests, portable coverage must
include:

- exact strict confidence and ankle/bbox ratio filtering, retained ratio, and
  canonical identity order;
- invalid/nonpositive bbox width and nonfinite ratio inputs;
- weighted nullspace hand oracle, positive global weight-scale invariance, and
  fixed unweighted trim membership; `None` must retain the corrected unweighted
  implementation in this task, while explicit all-one weighted arithmetic needs
  only float-tolerance plane/objective parity rather than bitwise equality;
- weighted D objective hand reduction and proof that weights reach both normal
  fit and D search;
- unweighted top-level solver regression plus density k=16/32/64 wiring on one
  synthetic population;
- sign-invariant normal angle/distance ratio, deterministic weighted median,
  oracle-normal and distance-only same-ray decompositions;
- exact identical support/observation identities across all variants;
- raw-value re-reduction to per-scene and global summaries; and
- script help, dry-run reviewed 17,992 counts, exact output file set, and one
  compact synthetic operation.
- old positional low-level calls, package-root exports, low-level invalid-weight
  rejection, and nonuniform sentinel-weight alignment through both trim passes.

## Completion Criteria

The task is complete only when:

- the density intermediate has separate Mathematical and Code Architecture
  reviews with no remaining Critical or Concern;
- portable tests cover density, weighting, invalid inputs, deterministic
  variants, and exact support-preserving evaluation;
- all eight scenes run for all required variants using the 17,992 fixed
  observations without sampling;
- raw density diagnostics and ground-effect errors can reproduce reported
  summaries;
- the k=16/32/64 and unweighted results are compared with units and support
  stated explicitly; and
- final repository-level review finds no remaining Critical or Concern in this
  task's delta.

## Modification History

- 2026-08-18: User selected the strict `>4.0` four-joint confidence gate and
  ankle-distance/bbox-width `<0.20`, required all retained observations rather
  than sampling, and requested density-derived contribution balancing.
- 2026-08-18: User required density to be a separately reviewed intermediate
  and authorized Layered Design plus Sequential Task Level 3 through real
  ground estimates and evaluation, including multiple density variants.
- 2026-08-18: After density implementation review closed, froze weighted RCR
  across both the normal nullspace fit and D objective. Angular trimming remains
  unweighted and fixed across variants. Also froze strict protocol selection,
  the four real variants, normal/distance decomposition, plain output, and
  portable gates for dedicated review.
- 2026-08-18: Weighted/Evaluation Mathematical review found one Critical and
  three Concerns; Code Architecture review found seven Concerns. Accepted all:
  complete plane coefficients now normalize/sign-align together; oracle
  geometry, lower weighted median, GT-plane closure, and invalid policy are
  exact; additive signatures and trim-weight alignment are frozen; generic
  optional ratio and estimator compatibility are preserved; pure diagnostic
  APIs and exact plain output keys are specified; output root is fail-new.
- 2026-08-18: First re-review left two small contract Concerns. Closed them by
  requiring finite oracle `d_i`/median inputs and explicit validate-copy-readonly
  behavior for both diagnostic records' array fields.
- 2026-08-18: Implemented the generic density IR and weighted RCR in
  `hjlib-ground-solver`, and strict VirtualCrowd collection, decomposition, and
  plain result operation in `hjlib-evaluation`. Mathematical and Code
  Architecture implementation reviews closed at 0 Critical / 0 Concern.
- 2026-08-18: Completed all 32 real scene/variant solves and exact reload on
  167,243 common support rows. Global mean same-ray errors were 18.896103 m
  unweighted, 15.384228 m at k=16, 15.590497 m at k=32, and 15.900516 m at
  k=64. k=16 improved the matched strict-filter unweighted mean by 18.59%, but
  remained 9.54% worse than the earlier confidence>3 sampled-5000 baseline.
  Normal-oracle and distance-only diagnostics did not improve together, so the
  combined gain is explicitly recorded as containing error cancellation.
- 2026-08-18: Final delta review identified that the pre-task RCR trim applied a
  sorted-position mask to original-order columns and dropped exact threshold
  ties. This task intentionally retains the corrected original-order,
  inclusive-threshold trim; `filtered_unweighted` is therefore the corrected
  unweighted control, not byte parity with pre-task HEAD. Restored the actual
  historical positional `D_init/device/flag` slots and made the new distance
  bounds plus observation weights keyword-only.
- 2026-08-18: Final review found and closed complete-readback, multi-duplicate-
  cluster radius-floor, SciPy typing, repo-local script-boundary, stale geometry
  pin, and documentation findings. All task-delta implementation reviews now
  have zero remaining Critical/Concern. The sole landing blocker is mechanical:
  commit `hjlib-ground-solver`, then bump the direct evaluation pin and commit
  evaluation; commits were explicitly outside this Level 3 task without new
  authorization.
- 2026-08-18: Replaced the active fixed-k density axis with exact LOO Gaussian
  KDE using Scott bandwidth, centered Cholesky whitening, stable chunked direct
  distances, and clipped inverse-density weights. The real single arm completed
  on all 17,992 observations and 167,243 evaluation rows. Global mean improved
  from 18.896103 m to 16.457570 m, while decomposition diagnostics exposed
  error cancellation. At that checkpoint, Cartesian expansion remained blocked
  on user confirmation.
- 2026-08-18: The user accepted the KDE single-arm checkpoint and authorized
  Cartesian preparation only. The exact 12 configs and six nested real-input
  population counts are now implemented, dry-run validated, and documented;
  real Cartesian solve/evaluation remains not run.
- 2026-08-19: Recorded the post-completion GT-K/GT-normal D-only height sweep.
  `H_prior=1.250 m` was the best of the six requested values with global mean
  `1.8723156219237906 m`, but remains an evaluation-set oracle diagnostic and
  does not mutate `baseline001(H_prior=1.35 m)`.
- 2026-08-19: Subsequent user authorization extended the earlier prepare-only
  checkpoint to real Cartesian execution. All 12 configurations completed,
  their plain results passed independent source reconstruction/readback, and
  the task headline/status now records the completed matrix rather than the
  superseded preparation boundary.
