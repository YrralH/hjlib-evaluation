# VirtualCrowd RCR Ground Evaluation

## Status

- State: implementation and first real quantitative run complete.
- Owner: `hjlib-evaluation`.
- Algorithm owner: `hjlib-ground-solver`.
- Dataset provider: `VirtualCrowd_Std` from `hjlib-dataset-std`.
- Visualization is explicitly deferred.

## Requirements

This task evaluates the existing RCR ground solver on VirtualCrowd using the
already-produced GT-MOT RTMLib detections. It is one evaluation task with two
code layers, not two campaign tasks.

The numerical optimization consumes only one generic `Tracked_Scene` and one
camera intrinsic matrix. The concrete script uses `VirtualCrowd_Std` to list
scenes and obtain camera intrinsics. It must not decode images, forward SMPL-X,
or depend on a VirtualCrowd raw-label dictionary during optimization.

The retained RTMW-133 layout uses COCO joints 5 and 6 as the shoulder pair and
15 and 16 as the ankle pair. A person-frame is high-confidence exactly when all
four scores are strictly greater than `3.0`. Its top and bottom observations are
the corresponding pairwise 2D midpoints. A row with `keypoints_mask=False` is
an absent detection and is outside the candidate population. For a present row,
nonfinite values and a zero-length
top-to-bottom segment are invalid and fail the scene operation rather than being
silently dropped after the confidence decision.

Two named evaluation strategies are required:

1. `first_frame_high_confidence`: retain every high-confidence observation at
   global frame index zero. This is a single-image diagnostic, not a headline.
2. `all_frames_high_confidence_sampled_5000`: form the complete high-confidence
   person-frame population, report its count, then uniformly sample 5,000 rows
   without replacement. If a future scene contains fewer than 5,000 candidates,
   retain all candidates and report the smaller selected count.

Sampling is over person-frames, not tracks. There is no per-track cap, track
balancing, straight-pose filter, duplicate-keypoint filter, or extra geometric
quality ranking. A fixed caller-supplied seed determines the sample; selected
rows are restored to canonical `(frame_index, person_id)` order before solving.

For quantitative comparison, the script requires the prior plain
ground-effect result as a support input. It must verify that the Crowd4D and
DyCrowd files expose the same ordered `(frame_id, gt_track_id)`, `image_xy_px`,
and GT-ground intersection arrays, then evaluate each new scene plane on those
same rays. This preserves the previously reviewed comparison population without
importing `hj-tpa-crowd4d` code. The error is the 3D Euclidean distance, in
metres, between the stored GT-ground intersection and the new plane's forward
intersection on the same camera ray.

The output is deliberately plain: one JSON summary and one numeric NPZ per
strategy and scene. There is no schema registry, receipt, source hashing, or
artifact-signing mechanism. The JSON records candidate count, selected count,
RCR dimensionless objective, unit-normal camera-frame plane, and ground-effect
summary. Each NPZ records the selected observation identities/coordinates,
ground-effect validity, and raw ground-effect errors when valid so the reported
values can be recomputed. A first-frame diagnostic whose plane does not give a
forward intersection for the complete scene support is retained as a plane
estimate but its complete scene metric is marked invalid; no support row is
dropped and no partial diagnostic reduction is reported. The all-frame headline
remains fail-closed: any invalid support ray fails the operation.

## Mathematical Architecture

For each accepted person-frame `i`, let

```text
top_i    = (kp_i[5,xy]  + kp_i[6,xy])  / 2
bottom_i = (kp_i[15,xy] + kp_i[16,xy]) / 2
quality_i = min(kp_i[[5,6,15,16], score])
```

and retain it iff `quality_i > 3.0`. The ordered population is sorted by
`(frame_index, person_id)`. Uniform sampling uses NumPy `Generator.choice`
without replacement over this ordered population. A separate deterministic
seed is derived as `base_seed + scene_rank`, where `scene_rank` is the position
in `VirtualCrowd_Std.get_list_scene_names()` after requiring that list to be
lexically sorted and unique. Both strategies in a scene use the same derived
seed, although the first-frame strategy performs no random draw.

The selected `(top, bottom, K)` arrays are passed unchanged to
`solve_ground_param_by_top_bottom_given_K`, with `H_prior=1.35`, distance grid
`[-5.0,80.0)` metres at `0.1 m`, CPU device, and `flag_opt=False`. The corrected
solver uses float64 inclusive angular-bias trimming, reduced SVD, rejects fewer
than three/rank-degenerate inputs and finite vanishing points at infinity, and
fails if the best distance lies on the search boundary. Its returned plane has
a unit Euclidean normal. The second return is the dimensionless
`relative-length + normalized-pixel` objective used to select the distance.

This is named an `RCR fixed-height baseline under the HJ high-confidence
protocol`, not author-pipeline parity. `H_prior` supplies absolute scale. The
inherited unnormalized homogeneous-line rows give larger pixel-height people
more influence inside the solver, and D fitting uses all selected rows without
an additional robust trim; both remain explicit legacy algorithm choices, not
silent protocol filters. Exact horizontal-camera infinite-VP support is outside
the current solver contract. The all-frame strategy is the headline; the
first-frame strategy is diagnostic only.

Given pixel `u=(x,y)`, define camera ray direction

```text
r = K^-1 [x, y, 1]^T.
```

For normalized method plane `n^T X + d = 0`, its forward same-ray intersection
is

```text
lambda = -d / (n^T r),
X_method = lambda r,
```

requiring normalized direction/normal absolute cosine `> 1e-10` and
`lambda > 0`. With the stored GT
intersection `X_gt`, the per-person-frame error is

```text
error_m = ||X_method - X_gt||_2.
```

Reduction is person-frame micro: count, mean, population standard deviation,
median, p90, p95, p99, minimum, and maximum over raw `error_m`. All geometry is
computed in float64 on the evaluation side; the retained RCR distance search
internally uses torch float32 and its returned plane is promoted to float64 only
after solving. Percentiles use NumPy `method='linear'`. Any invalid
selected observation, solver output, ray, or
intersection fails the complete scene/strategy operation; it is not removed
from a denominator. Because `first_frame_high_confidence` is explicitly a
diagnostic rather than a headline, that strategy records an invalid whole-scene
metric and continues to later scenes; its global metric is invalid if any scene
is invalid. The headline strategy does not use this continuation policy.

## Code Architecture

`src/hjlib_evaluation/ground_estimation_protocol.py` owns the reusable layer.
It defines:

- immutable `Ground_Observation_Set(frame_indices int64[N], person_ids
  int64[N], top_xy_px float64[N,2], bottom_xy_px float64[N,2], quality
  float64[N])`, with owned read-only arrays in canonical order;
- immutable `Ground_Effect_Support(frame_ids int64[M], gt_track_ids int64[M],
  image_xy_px float64[M,2], gt_intersections_camera_m float64[M,3])`;
- immutable `Ground_Estimation_Result(observations, plane_camera_abcd
  float64[4], objective: float)`;
- pure `collect_ground_observations(tracked_scene, top_joint_pair,
  bottom_joint_pair, confidence_threshold)`;
- pure `select_ground_observations_at_frame(observations, frame_index)` and
  `sample_ground_observations(observations, max_count, seed)`;
- `estimate_ground_from_observations(observations, K, estimator)`, where the
  injectable estimator callable accepts `(top_xy_px, bottom_xy_px, K)` and
  returns `(plane[4], objective scalar)`;
- `compute_same_ray_ground_errors(support, K, plane)` and
  `summarize_ground_errors(error_m)`.

The module requires direct pinned dependencies on `hjlib-detection` and
`hjlib-ground-solver`, and reuses the existing direct `hjlib-geometry`
dependency for `intersect_rays_with_planes`; it does not duplicate plane
normalization/intersection. It must not import VirtualCrowd, TPA modules,
filesystem paths, CLI state, or image readers.

`script/evaluate_virtualcrowd_rcr_ground.py` owns the concrete operation and a
standalone `python ./script/...` Typer entry function. It is intentionally not a
registered package CLI because it is a dataset-specific experiment operation,
while reusable contracts remain in `src`. It accepts explicit dataset,
tracked-scene, prior
ground-effect-support, and new output roots. It instantiates
`VirtualCrowd_Std`, loads the eight `Tracked_Scene` files, applies the two fixed
strategies, calls the reusable layer, validates the prior ray support, and
writes the plain result. Dataset-specific scene/file naming and the fixed joint
layout/threshold/sample count live here.

The script loader alone knows the prior TPA-origin NPZ layout. After verifying
the Crowd4D and DyCrowd support arrays are exactly equal, it constructs the
method-neutral immutable `Ground_Effect_Support`; raw file keys never enter
`src`. It also requires every stored GT intersection to have positive camera
depth and verifies `project_K(gt_intersection)` equals stored `image_xy_px` with
`rtol=0`, `atol=1e-8`, preventing two mutually equal old files from being
silently evaluated under a different `K`. For each scene it requires camera
batch length equal to
`Tracked_Scene.num_frame`, all camera rows valid, and all `K` rows exactly
equal before selecting one fixed `K`. The accepted detection artifact was
produced from the same native-size VirtualCrowd streamer; because
`Tracked_Scene` does not store image-size provenance, equality of detection
pixel space and the native camera pixel space remains an explicit input
precondition rather than a claim inferred from the payload.

The operation is scene-major and holds at most one tracked scene, its complete
10k--37k high-confidence observation population, one selected 5,000-row set,
and one scene's ground-effect support in memory. The
solver is CPU numerical code; the earlier three-GPU split belongs only to the
completed RTMLib detection production and is not reproduced here.

## Smoke-Test Standard

Portable tests must cover:

- sparse multi-track `Tracked_Scene` flattening and canonical ordering;
- strict four-joint confidence threshold and midpoint construction;
- first-frame selection;
- deterministic 5,000-cap sampling without replacement and without track
  balancing;
- fewer-than-cap behavior;
- nonfinite and degenerate observation rejection;
- a fake estimator seam for wiring tests plus one existing real RCR smoke-sized
  invocation;
- hand-solvable same-ray error, parallel/behind-camera rejection, and metre
  summary reduction;
- prior support mismatch rejection; and
- support-to-current-K projection round-trip rejection;
- concrete script help plus a small synthetic two-scene operation.

The real-data check must first reproduce the reviewed candidate counts
`156,421` total and per-scene counts recorded by the operation, then run both
strategies for all eight scenes. Visualization is not a completion criterion.

## Modification History

- 2026-08-18: User selected `hjlib-evaluation` as the task owner, required the
  executable function head in `script/`, allowed only reusable connection code
  in `src/`, corrected sampling from best-per-track to high-confidence
  person-frame sampling capped near 5,000 per scene, and deferred visualization.
- 2026-08-18: Dedicated Mathematical Architecture review found the inherited
  order-dependent RCR filter bug and required numerical/coordinate/support
  closure. User authorized fixing the solver. Three independent correctness,
  reasonableness, and efficiency reviews then separated clear implementation
  bugs from retained modeling choices. The solver leaf was corrected for
  original-order/tied bias filtering, reduced SVD, explicit search/objective,
  device forwarding, and input/rank gates. This design now names all-frame as
  the fixed-height headline and first-frame as diagnostic, requires prior ray
  support, and freezes the remaining numerical/API boundaries.
- 2026-08-18: Dedicated Code Architecture review reported no Critical and six
  Concerns. The design now requires direct dependencies, geometry reuse, exact
  immutable array contracts and estimator seam, a script-only TPA-shaped
  support loader, fixed-camera validation, an explicit standalone CLI decision,
  and honest full-population memory accounting.
- 2026-08-18: Mathematical re-review closed the prior Critical and left three
  implementation-readiness Concerns. The design now validates prior support
  against the current K, states the retained float32 RCR-search boundary, and
  requires direct rank-degenerate and infinite-VP solver regressions.
- 2026-08-18: The first real quantitative attempt showed that five first-frame
  planes put part of the complete scene support behind the camera, while all
  eight all-frame planes support every ray. The diagnostic now records these as
  whole-scene invalid results without dropping rows or producing a partial
  reduction; the all-frame headline remains fail-closed.
- 2026-08-18: The final real run evaluated all 167,243 shared person-frame rays.
  The all-frame/5,000 headline is valid for all eight scenes and reports mean
  14.044917 m, median 10.466237 m, p90 32.641947 m, and population standard
  deviation 11.257760 m. The first-frame diagnostic is globally invalid because
  five scenes do not support every forward ray. Portable smoke is 43 passed and
  the new module/script/test targeted Pyright check is clean.
