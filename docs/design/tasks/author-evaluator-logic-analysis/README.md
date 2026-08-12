# Author Evaluator Logic Analysis

## Requirements

Establish a verified description of the supplied Crowd4D/DyCrowd evaluator
before designing an HJ-owned generalized evaluation path. The description must
be precise enough that a later task can decide what behavior to reproduce,
reject, parameterize, or isolate without importing Crowd4D-private schemas into
the public `hjlib-evaluation` contract.

The analysis covers:

1. GT and prediction loading, schema normalization, units, coordinate spaces,
   frame selection, and identity lifecycle;
2. Crowd4D/DyCrowd branch selection and tracking/association behavior;
3. 2D projection, joint selection/mapping, alignment, and each per-frame or
   temporal metric;
4. invalid, missing, unmatched, and empty-case behavior;
5. per-person, per-frame, per-scene, and global reduction, including weights
   and denominators;
6. formatting/scaling between internal values, result tables, and paper rows;
7. cross-checks against the frozen Campaign 02 fresh results.

Third-party material remains read-only and outside Git. Cite source paths and
symbols without copying implementation. Label every conclusion as directly
observed, empirically verified, inferred, or unresolved. Official upstream
provenance remains unverified and must not be implied by behavioral analysis.

## Mathematical Architecture

### Evidence identity and confidence

The analyzed implementation is the identified machine-local evaluation package,
not a verified official checkout. Campaign 02 froze its normalized residence
digest as
`9cd6a6e3140a0111538814eeea3dce86cd4e22dada1aaee5190f9d3dd0ee8350`.
The principal source identities are:

| Source | SHA-256 |
| --- | --- |
| `scripts/run_eval_py14.py` | `733721dd01a292b92b7664956acfa9264dab22180e73487d9b11b4d4b78c3d4d` |
| `crowd4d/eval/eval_utils.py` | `495dd3873a56902d26dedda0f36af5cd07a709d44ddd7366ea3ff68d50070215` |
| `crowd4d/eval/smpl.py` | `62243356ed4a2db1279322e5c1c923091bc6ff3d784f3dc66bb34e3b4305b6fa` |
| `crowd4d/utils/projection.py` | `712c353fdc45a63f65ce1c7ef9f60e07646ef1981c40430e3ac42a5b0b67d9fa` |
| `crowd4d/constants/eval_vis_constants.py` | `8a5cd821a1897c4c28fc4e0608b728a9d848c9dca76e64fef22cd7d162d2d45f` |
| `crowd4d/constants/optim_constants.py` | `0671db3ae0fada4bfd55222d13e0cd16244e3fc14e00c4bfed4f6eca2602161c` |

“Observed” below means direct control/data flow in those identified sources.
“Verified” means the behavior is also supported by frozen Campaign 02 runs or a
read-only structure probe. “Inferred” and “unresolved” are stated explicitly.

### Execution and reduction graph

The native path is:

```text
run_eval_py14.main
  -> EvalHandler.__init__
       -> load neutral SMPL + joint regressors
       -> load one prediction artifact and one GT label per scene
  -> EvalHandler.handle
       -> for each scene
            -> for each selected frame
                 -> normalize GT and prediction
                 -> 2D OKS matching
                 -> frame metrics
            -> repair track-to-GT assignments
            -> temporal metrics
       -> collect each scene's frame rows
  -> format_results
       -> mean each scene's frame rows
       -> concatenate every scene's frame rows and take the global mean
       -> round displayed cells to four decimals
```

For scene `s`, the evaluator selects
`F_s = min(200, prediction_frame_count, GT_entry_count)` GT dictionary entries
in insertion order. Zero selected entries raise `ValueError`. Prediction frame
`t` is paired with the `t`-th selected GT entry; GT frame numbers are used only
to construct display names. There is no sort, timestamp alignment, frame-ID
equality check, or use of the declared frame rate. The native `virtualcrowd`
default roster contains only `scene1`; Campaign 02 obtained eight-scene coverage
by explicitly passing all eight names through `--scenes`.

### Native data model

For each scene, a prediction artifact provides:

- `track_flag`: `(T, N_p)`, defining whether prediction column `p` is active at
  frame index `t`;
- `thetas`: `(T, N_p, 72)`, SMPL axis-angle pose parameters;
- `trans`: `(T, N_p, 3)`;
- `betas`: accepted as `(T, N_p, ...)`, `(1, N_p, ...)`, or `(N_p, ...)`;
- `xscale_factor`: the same three leading-shape alternatives;
- `cam_int`: `[fx, fy, cx, cy]` or a `3 x 3` intrinsic matrix;
- `ground_plane`: four plane coefficients.

The frozen artifacts use 10 betas, 72 pose values, and 200 frames. Prediction
column IDs are zero-based array positions and are not asserted to be GT IDs.
For each active person, neutral SMPL produces 54 joints and an H36M-17
regression. Shape is scaled around the model origin by
`q_t,p = 1.1 ** xscale_factor_t,p`, then camera translation is added. The frozen
artifacts use track-static scale and beta values, while the loader also accepts
frame-varying values:

```text
J_pred_camera(t,p) = q_t,p * SMPL54(theta(t,p), beta(t,p)) + trans(t,p)
H_pred_camera(t,p) = q_t,p * H36M17(theta(t,p), beta(t,p)) + trans(t,p)
```

GT loading prefers `<scene>.json` and falls back to `<scene>.pkl`. JSON requires
exact schema version `virtualcrowd_label_json_v1`. The legacy pickle path does
not perform an equivalent schema check; it mutates each loaded frame dictionary
by aliasing `hvip2d`/`hvip3d` into missing `hsip2d`/`hsip3d` fields and derives
`max_track_id` from frame contents.

The GT JSON schema contains, per instance, `track_id`, `bbox_xyxy`,
`joints_2d_smpl54` with shape `(54, 3)`, `joints_3d_smpl54` with shape
`(54, 3)`, and `joints_3d_h36m17` with shape `(17, 3)`. GT IDs are used as
direct array indices; the supplied labels use positive IDs and reserve index
zero. JSON `person_num` is absent in the frozen labels, so the evaluator falls
back to legacy per-scene nominal counts.

The read-only structure probe established the following current domains:

| Scene | GT instances/frame | GT ID domain | Crowd4D columns/active | DyCrowd columns/active |
| --- | ---: | ---: | ---: | ---: |
| `scene1` | 85 | 1–85 | 83 / 83 | 85 / 78–84 |
| `scene2` | 200 | 1–200 | 199 / 199 | 200 / 184–200 |
| `scene3` | 72 | 1–72 | 72 / 72 | 72 / 66–72 |
| `scene4` | 179 | 1–179 | 170 / 170 | 174 / 155–172 |
| `scene1_view2` | 56–59 | 1–83, 61 unique | 60 / 55–58 | 60 / 48–58 |
| `scene2_view2` | 74–80 | 1–149, 100 unique | 97 / 74–80 | 94 / 68–78 |
| `scene3_view2` | 53–57 | 1–72, 60 unique | 57 / 52–56 | 58 / 44–57 |
| `scene4_view2` | 165–171 | 1–179, 174 unique | 145 / 142–144 | 166 / 135–161 |

The first four scenes therefore have constant GT presence, while view-2 labels
have sparse IDs and changing presence. This distinction matters to the temporal
missing-person calculation below.

The evaluator treats SMPL/GT coordinates as metres before multiplying joint
errors by 1000. Metre input is inferred from SMPL convention and verified by
the exact millimetre-scale paper parity; the JSON schema itself does not declare
a unit. `cam_extrinsics`, `scene_shape`, and `frame_rate` are loaded but do not
participate in metric computation.

### Derived joint and anchor representations

Matching uses the COCO-17 subset selected by names from SMPL-54. Only the first
two coordinates participate; the third GT 2D channel and all visibility flags
are ignored.

Pose metrics use the first 14 joints of the package's reordered H36M-17 output.
The H36M joint at index 14 is the pelvis root:

```text
H14_local = H36M17[:14] - H36M17[14]
H14_camera = H36M17[:14]
```

The crowd torso center is the mean of SMPL-54 indices `[16, 17, 45, 46]`
(left/right shoulders and right/left extra hips). GT HSIP is rebuilt by
orthogonally projecting that GT torso center onto the GT ground plane. The
prediction path also computes a ground-projected HSIP using the predicted
ground plane, but that value is not consumed by any metric.

Perspective projection uses only intrinsics:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

No depth-validity guard or camera-extrinsic transform is applied.

### Frame association

For GT `g`, prediction `p`, COCO joint `k`, and GT bounding-box area `A_g`, the
pairwise OKS is:

```text
e(g,p,k) = ||x_pred(p,k) - x_gt(g,k)||^2
           / ((2*sigma_k)^2 * max(A_g, spacing(1)) * 2)
OKS(g,p) = mean_k exp(-e(g,p,k))
```

The 17 COCO sigmas are the conventional constants embedded in
`_compute_oks_matrix`. Every joint is included regardless of its visibility
channel. Matching clamps bbox area to `max(area, spacing(1))`. The separately
reported matched-pair OKS clamps first and then adds `spacing(1)` again in its
denominator, so a zero or negative raw area uses `2*spacing(1)` there rather
than the matching path's `spacing(1)`.

Association is a custom asymmetric greedy procedure, not Hungarian matching:

1. each GT independently selects its minimum-distance prediction, where
   distance is `1 - OKS`;
2. GTs are processed in ascending order of their selected distance;
3. a pair is accepted when `OKS > 1e-6` and the selected prediction has not
   already been consumed;
4. when two GTs select the same prediction, the later GT is dropped rather than
   trying its second-best prediction.

The threshold is therefore permissive, but the greedy collision behavior can
still leave matchable GT unmatched. Equal-distance predictions select the
lowest prediction column through the first `np.where` result. Equal GT minima
are ordered by `np.argsort`; its tie stability is not established as a semantic
contract. The frame match ratio is `r = M / max(G,1)`.

### Frame metrics

Let the matched torso-center point sets be `P = {p_i}` and `Q = {q_i}`.

`PPDS` evaluates every unordered pair with nonzero GT distance:

```text
relative_error_ij = abs(||p_i-p_j|| - ||q_i-q_j||) / ||q_i-q_j||
pair_score_ij = max(1 - relative_error_ij, 0)
PPDS_matched = mean(pair_score_ij over valid GT pairs)
PPDS = r * PPDS_matched
```

Fewer than two matched people, or no nonzero GT pair distance, yields zero.

`PA-PPDS` first estimates one reflection-disabled similarity transform from all
matched predicted torso centers to their GT counterparts, then applies the same
PPDS equation and match-ratio multiplier. The transform scale `a` also produces
an internal scale score. It is `SS = a` when `a <= 1`, otherwise
`SS = 1 / (a + 1e-9)`; unlike PA-PPDS, `SS` is not multiplied by the match
ratio.

`PCOD` evaluates the sign of z-order for each unordered matched pair:

```text
correct_ij = ((p_j.z - p_i.z) * (h_j.z - h_i.z)) > 0
PCOD = r * mean(correct_ij)
```

Here `p` is the unprojected predicted torso center, while `h` is the
ground-projected GT HSIP. A tie is incorrect. This predicted-torso versus
GT-HSIP asymmetry is observed behavior; the computed predicted HSIP is unused.

The reported `OKS` is the mean OKS of the accepted pairs, multiplied by `r`.
It repeats the same all-17-joint, no-visibility semantics used for matching.

`MPJPE` is the mean Euclidean error over matched people and their 14
pelvis-relative H36M joints, multiplied by 1000. `PA-MPJPE` estimates a separate
similarity transform for each matched person's 14-joint pose before the same
error. Each unmatched GT person contributes a fixed 150 mm penalty:

```text
metric_frame = (metric_matched * M + 150 * (G-M)) / G
```

Redundant prediction punishment is
`RP = clip(P/G - 1.02, 0, 1)`. The internal composite score is:

```text
Score = 0.3*PA-PPDS + 0.2*SS + 0.1*PCOD + 0.4*OKS - 0.5*RP
```

`SS` and `RP` are not emitted as result columns. If a frame has GT but no
matches, MPJPE and PA-MPJPE are 150, positive metrics are zero, and Score is
only the negative redundancy term. A later frame-metric branch would emit
matched ratio one and zero metrics for no GT, but normalized JSON represents
empty instance fields as rank-one empty arrays and `_gt_process` indexes them
as rank-three arrays first. Therefore an empty JSON frame raises before that
branch; the frozen labels contain no empty frame.

PA-PPDS delegates degeneracy handling to `trimesh.registration.procrustes` with
no local finite/zero-spread guard. PA-MPJPE and the temporal similarity metrics
divide by predicted centered variance with no zero-variance or finite-value
guard. Degenerate point sets may therefore raise or propagate non-finite
values; no fallback metric value is defined.

### Track repair and temporal metrics

Per-frame matching first creates `Matching_pred[p,t]`, containing a GT ID or
`-1`. The two supplied workflows use different repair modes:

- Crowd4D omits `--use-gt-mot`. For each prediction column, the evaluator takes
  the most frequent non-`-1` GT ID that occurs more than once, fills every frame
  between its first and last occurrence with that ID, and clears frames outside
  the interval. If no GT ID repeats, the original assignments remain. Frequency
  ties inherit reversed `np.argsort(counts)` ordering and are treated as
  version-sensitive implementation behavior rather than a semantic tie rule.
- DyCrowd passes `--use-gt-mot`. For each prediction column, the modal GT ID
  across all frames is selected after giving `-1` a zero count, and that ID is
  broadcast across the entire timeline. Ties follow sorted `np.unique` order,
  so the smallest tied ID wins. Prediction `track_flag` subsequently removes
  inactive frames.

Neither repair mode checks GT presence after filling/broadcasting. A repaired
assignment on a frame where that GT ID is absent reads the zero-initialized GT
tracking buffer. This is a possible path in the implementation; whether the
current artifacts trigger it materially remains unresolved.

Every active, repaired assignment marks its `(GT ID, frame)` cell as covered,
including assignments from prediction tracks with only one or two samples.
Only prediction tracks with more than two such samples enter temporal metric
numerators. Let a numerator-valid track supply `L_p` samples:

- `WA-MPJPE`: one similarity transform fitted over all `L_p * 14` pelvis-local
  joint points, then MPJPE over the full track;
- `W-MPJPE`: one similarity transform fitted only from the first two matched
  samples (`2 * 14` points), then applied to the full track. The two samples are
  not required to be consecutive in original frame time;
- `GMPJPE`: unaligned camera-coordinate H36M-14 MPJPE;
- `ACCEL`: inputs are the same pelvis-relative H36M-14 joints used by
  WA-MPJPE/W-MPJPE, so per-frame root translation is absent. Finite differences
  operate on the compact matched-sample list, not original frame indices or
  seconds. Inactive and unmatched frames are removed, so samples separated in
  source time become adjacent unit steps. For each joint, acceleration is
  `J[i+2] - 2*J[i+1] + J[i]`; the implementation compares the scalar
  acceleration magnitudes, takes an L2 norm across 14 joints per compact time
  step, averages over compact time, and multiplies by 1000. It is not the mean
  norm of the 3D acceleration-error vector and does not account for original
  frame gaps or multiply by `fps^2`.

WA-MPJPE and W-MPJPE operate on pelvis-relative joints despite their
world/global-oriented names; they do not retain per-frame root translation.
GMPJPE alone consumes absolute camera-coordinate H36M-14 joints. WA-MPJPE,
W-MPJPE, and GMPJPE are weighted by `L_p` across numerator-valid prediction
tracks. ACCEL is an unweighted mean of per-track ACCEL values, so short and long
valid tracks have equal weight.

The missing-person penalty does not use the populated `Gt_presence_flags`.
Instead:

```text
nominal_slots = legacy_person_num * selected_frame_count
covered_slots = count of unique (GT ID, frame) cells marked by every active,
                repaired assignment, including tracks with L_p <= 2
missing_slots = nominal_slots - covered_slots
metric = (matched_metric * sum_p L_p + 150 * missing_slots)
         / (sum_p L_p + missing_slots), where the sum includes only L_p > 2
```

Consequently, view-2 frames penalize nominal scene identities that are absent
from that camera view. A short track can reduce missing slots without adding a
matched numerator sample. Multiple numerator-valid prediction tracks mapped to
one GT/frame add multiple matched samples but cover only one missing slot.
These are verified consequences of the array domains and reduction code, not
proposed HJ semantics.

The equation applies only when at least one numerator-valid track exists. If no
track has more than two samples, WA-MPJPE, W-MPJPE, and GMPJPE are set directly
to 150 and ACCEL to zero, regardless of cells covered by shorter tracks.

Each scene's four temporal scalars are copied into every frame row. This makes
the later frame mean reproduce the same scene scalar.

### Scene and global aggregation

Every frame row contains twelve metrics in this order:

```text
matched ratio, PPDS, PA-PPDS, PCOD, MPJPE, PA-MPJPE,
WA-MPJPE, W-MPJPE, ACCEL, OKS, GMPJPE, Score
```

A scene row is the unweighted mean of its selected frame rows. The `mean` row is
the unweighted mean after concatenating all selected frame rows from all scenes;
it is frame-weighted, not scene-, person-, pair-, or track-weighted. All frozen
VirtualCrowd scenes contain 200 selected frames, so the current mean is also an
equal mean of the eight scene rows before display rounding. Display uses Python
rounding to four decimals.

### Frozen-result cross-check

Campaign 02 executed both complete eight-scene paths in `hjlib_py312` with zero
writes to supplied material. The fresh tables reproduce 215 of 216 bundled
numeric cells exactly; the remaining DyCrowd scene cell differs by `0.0001` and
is equal at three decimals. The mean rows are:

| Path | PPDS | PA-PPDS | PCOD | MPJPE | PA-MPJPE | WA-MPJPE | W-MPJPE | ACCEL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Crowd4D | 0.9143 | 0.9238 | 0.9557 | 59.3523 | 44.3563 | 63.9806 | 73.3160 | 12.9303 |
| DyCrowd + GT-MOT repair | 0.8466 | 0.9123 | 0.9538 | 68.8128 | 45.3351 | 65.9120 | 80.3377 | 15.5329 |

After multiplying PPDS, PA-PPDS, and PCOD by 100 and rounding paper cells to two
decimals, all 16 corresponding Table 2 cells match arXiv
`2607.19517v1`. This verifies execution and aggregation parity for the supplied
artifacts, but not that every legacy edge behavior is desirable or official.
Paper Table 4's Crowd4D PCOD `96.57` remains inconsistent with Table 2 and the
fresh/bundled value `95.57`; no evaluator-logic explanation is evidenced.

### GT-MOT protocol interpretation addendum

The user's phrase “不重不漏” refers to the prediction artifacts stored on disk
and presented to the evaluator, not to the evaluator's matched metric pairs.
All active stored prediction columns enter the supplied evaluator; that fact
does not imply that every GT or prediction contributes to every metric.

The user also reports the author's intended interface as GT-MOT input, for which
there should theoretically be no identity-matching step. That stated intent is
not the supplied implementation's behavior. Both workflows still perform the
per-frame greedy OKS association described above. `--use-gt-mot` only changes
the later temporal identity repair; it does not bypass frame association.

A read-only artifact probe verified that, for every active Crowd4D prediction
across all eight scenes, the stored `idxs` value belongs to the current GT
`track_id` domain and is stable per prediction column. In sampled first/last
frames, 1,695 accepted OKS pairs agreed with the stored identity in 1,685 cases
and disagreed in 10. The supplied evaluator ignores `idxs`, so those conflicts
show that geometry matching is not merely a redundant re-expression of the
stored Crowd4D identity. DyCrowd's `idxs` convention is not yet established
well enough to use as a direct GT-ID contract.

This addendum changes protocol interpretation, not the documented execution
baseline: the author-parity task must first reproduce the supplied matching and
repair behavior exactly. Any direct-ID or otherwise corrected GT-MOT protocol
belongs to a separately reviewed result profile and must retain the author
baseline alongside it.

### Observed non-contract behavior and unresolved intent

The following behavior must not silently become an HJ public contract:

- nominal-person rather than actual-presence temporal missing penalties;
- repaired IDs being consumed without a GT-presence guard;
- PCOD comparing predicted torso depth with ground-projected GT HSIP depth;
- greedy nearest-only OKS matching with no second-best retry;
- all-joint OKS ignoring the GT visibility channel;
- compacted-sample ACCEL with discarded frame gaps and
  magnitude-before-error reduction;
- WA-MPJPE/W-MPJPE names suggesting global coordinates although their inputs
  are pelvis-relative joints;
- first-200/insertion-order frame pairing without identity validation;
- `--invisible-only`, `only_compute_unvis`, and
  `visible_body_joint_threshold` having no computational effect;
- computed prediction HSIP, GT presence flags, camera extrinsics, frame rate,
  and scene shape being unused;
- `SS` and `RP` affecting Score while remaining absent from result columns.

These facts do not by themselves classify the behavior as a defect. The next
campaign decision must separately choose which semantics to reproduce for
author parity, expose as an explicit legacy profile, correct in a stable HJ
contract, or test as unresolved.

## Code Architecture

Deferred. This task does not define code residence, public APIs, adapters, or
implementation decomposition. Those decisions require the accepted
Mathematical Architecture and a separately activated task.

## Smoke-Test Standard

No reusable test facility is authorized in this analysis task. Internal
cross-checks may use disposable read-only probes, but durable claims must be
checked against the frozen Campaign 02 eight-scene results and record exact
evidence identity. Any later implementation test standard belongs to its own
task.

## Modification History

- 2026-08-11: Created when the user activated Campaign 03 and selected analysis
  of the author evaluator's logic as the first task. Requirements and analysis
  boundary were fixed; Mathematical Architecture remains active.
- 2026-08-11: Initial dedicated Mathematical Architecture review found two
  Critical fidelity errors, four Concerns, and three Notes. Corrected
  short-track coverage semantics and compact-time ACCEL, then accepted and
  addressed the empty-GT, scene-selection, degenerate-alignment, legacy-GT,
  exact-scale, OKS-area, and tie-behavior findings. The layer remains active
  pending focused re-review.
- 2026-08-11: Focused re-review confirmed both Critical fixes and every prior
  lower-level finding. Its sole new Concern was that ACCEL's pelvis-relative
  input remained implicit; made that input explicit. No Critical or other open
  finding remains, and the Mathematical Architecture is accepted as the
  supplied-evaluator fidelity baseline.
- 2026-08-12: Clarified the user-defined distinction between complete stored
  prediction output and matched metric pairs. Added the user-reported GT-MOT
  intent, verified Crowd4D `idxs` evidence, and preserved geometric association
  as author-parity behavior rather than accepting it as the reviewed protocol.
