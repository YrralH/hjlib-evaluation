# Stable Leave-One-Out Gaussian KDE Density

## Status

Design ready for dedicated Mathematical and Code Architecture review. No
implementation may begin until both layers close without a remaining Critical
or Concern.

## Requirements

This layer replaces fixed-k nearest-neighbor density as the proposed formal
spatial-density algorithm. It estimates a continuous empirical observation
density on the provisional ground plane and returns finite positive
inverse-density weights. It does not select observations, use GT, infer scene
support, solve a ground plane, or define evaluation membership.

The generic owner is `hjlib-ground-solver`; the current VirtualCrowd single-arm
operation remains in `hjlib-evaluation`. The historical kNN API and artifacts
may remain inspectable during this decision point, but the active operation
must construct only `unweighted` and `density_kde_scott_loo`. The latter is the
single canonical token used as density-mode value, output directory, summary
key, validator key, and test identifier; there is no second alias.

## Mathematical Architecture

### Inputs and unit-plane coordinates

Inputs are finite float64 `bottom_xy_px[N,2]`, nonsingular float64
`camera_K[3,3]`, a finite nonzero provisional camera normal `[3]`, `N >= 3`,
and two Python-float clip bounds satisfying
`0 < minimum_pre_normalization_weight <= 1 <=
maximum_pre_normalization_weight < inf`. Their generic API defaults and the
single arm are both exactly `0.25` and `4.0`. Reuse the existing deterministic
unit-plane projection contract:
normalize the provisional normal, choose the sign-equivalent forward plane at
absolute offset one, intersect every camera ray, and express intersections in
the deterministic orthonormal 2D tangent basis. Every row is retained in exact
input order. Normal sign/input scale and tangent-basis orientation cannot
change pairwise density or final weights.

Let the resulting coordinates be `q_i in R^2`. Define the unbiased sample
covariance `S` (`ddof=1`) and require it to be finite symmetric positive
definite. Collinear or collapsed populations fail the whole call; no implicit
ridge changes the metric. For dimension `d=2`, use the standard Scott factor

```text
f = N^(-1/(d+4)) = N^(-1/6)
H = f^2 S.
```

`H` is the common full-covariance Gaussian kernel covariance. In two
dimensions SciPy's Scott and Silverman factors are identical, so Silverman is
not a second variant. Likelihood bandwidth CV is explicitly deferred: it adds
another search axis, is expensive under stable LOO evaluation, and is not
needed for the user-requested single generic arm.

### Stable exact leave-one-out density

For every training observation, exclude its diagonal self-kernel from the
start:

```text
log_kernel_ij =
    -0.5 * (q_i-q_j)^T H^-1 (q_i-q_j)
    + log_kernel_normalizer

log_density_loo_i =
    logsumexp({log_kernel_ij | j != i}) - log(N-1).
```

Using the lower Cholesky factor `L` defined below, compute the normalizer without
forming a determinant:

```text
log_det_H = 2 * sum(log(diag(L)))
log_kernel_normalizer = -0.5 * (d*log(2*pi) + log_det_H).
```

Both values and every Cholesky diagonal must be finite and strictly positive
where applicable; otherwise the complete call fails.

Directly computing `N * gaussian_kde(q_i) - self_kernel` is prohibited because
isolated observations cause catastrophic cancellation. `log_density_loo` must
be finite; very negative values remain valid and must not be exponentiated
before clipping.

Define inverse density entirely in log space:

```text
log_relative_inverse_i = median(log_density_loo) - log_density_loo_i
log_clipped_i = clip(
    log_relative_inverse_i,
    log(minimum_pre_normalization_weight),
    log(maximum_pre_normalization_weight),
)
clipped_i = exp(log_clipped_i)
w_i = clipped_i / mean(clipped).
```

Thus every final weight is finite and strictly positive, `mean(w)=1`, and the
generic pre-normalization dynamic range is bounded by
`maximum_pre_normalization_weight / minimum_pre_normalization_weight`; it is 16
for the single arm's fixed `0.25/4.0`. Dense observations receive
smaller weights and sparse observations receive larger weights. Final weights
are invariant to coordinate translation, orthonormal basis change, and common
positive coordinate scale.

The effective sample size is

```text
ESS = (sum_i w_i)^2 / sum_i(w_i^2) = N^2 / sum_i(w_i^2),
```

where the second equality uses `mean(w)=1`. Require finite
`1 - 1e-12 <= ESS <= N + 1e-12`.

This is kernel-smoothed empirical observation-density balancing, not a proof of
uniform physical ground-area contribution. The algorithm has no known support
polygon and performs no boundary correction. A uniform finite square can
therefore overweight boundary rows; this limitation must be reported with the
result. The inherited RCR homogeneous-line magnitude also remains in final
normal-fit leverage.

### Solver and single-arm semantics

Run the corrected unweighted RCR once on the selected observations. Use its
normal only to construct the KDE coordinates and weights, then run weighted RCR
once with the exact same top/bottom rows. The same weight vector enters final
normal SVD and the complete-population D objective, while angular-trim
membership remains unweighted as previously reviewed.

The single-arm comparison contains exactly:

```text
filtered_unweighted
density_kde_scott_loo
```

No kNN variant enters the active comparison. Both variants report normal angle,
distance ratio, combined same-ray error, normal-oracle error, and distance-only
error on identical support.

## Code Architecture

Extend `hjlib-ground-solver/estimate_ground/observation_density.py` with a new
immutable `Ground_Observation_KDE_Density` and pure public function:

```text
compute_ground_observation_kde_density(
    bottom_xy_px: NDArray[float64],
    camera_K: NDArray[float64],
    provisional_normal_camera: NDArray[float64],
    *,
    minimum_pre_normalization_weight: float = 0.25,
    maximum_pre_normalization_weight: float = 4.0,
) -> Ground_Observation_KDE_Density
```

The clip bounds are keyword-only; the fixed internal chunk size is not an API
parameter. Reuse one internal unit-plane
projection operation shared with the historical kNN producer; do not copy
camera/plane math. The new record owns read-only float64 arrays:

```text
provisional_unit_plane_xy[N,2]
kernel_covariance_unit_plane[2,2]
loo_log_density_per_unit_area[N]
log_relative_inverse_density[N]
clipped_relative_inverse_density[N]
normalized_observation_weights[N]
scott_bandwidth_factor scalar
minimum_pre_normalization_weight scalar
maximum_pre_normalization_weight scalar
weight_normalization_factor scalar
effective_sample_size scalar
```

The producer and record independently validate all cheap algebraic closure.
The record does not rebuild the quadratic KDE. Public exports come from
`hjlib_ground_solver.estimate_ground` and package root. No KDE type enters
`hjlib-evaluation/src`.

Compute the lower Cholesky factor `L` such that `L @ L.T = H`; a failed or
nonfinite factor rejects the call. Center before whitening:

```text
z = solve(L, (q - mean(q, axis=0)).T).T.
```

For query chunks of 256 rows, compute squared Euclidean distances to all
whitened rows with `scipy.spatial.distance.cdist(..., metric='sqeuclidean')`.
Require the returned block to be finite and nonnegative before replacing the
matching diagonal by positive infinity and applying
`scipy.special.logsumexp`. Direct coordinate differences are required here:
the norm/BLAS identity suffers catastrophic cancellation for a sufficiently
isolated observation even after centering. This is exact apart from ordinary
float64 arithmetic and allocates at most one `[256,N]` distance block, not a
persistent `[N,N]` matrix. The chunk size is an internal performance constant
and cannot change results.

Update the existing repo-local VirtualCrowd density operation to write only the
two active variants and KDE record fields. Keep output plain (`summary.json`
plus numeric NPZ), fail-new, and independently reload/recompute observations,
KDE weights, plane/objective, support identities, and summaries. The independent
validator has the exact boundary

```text
validate_written_results(
    output_root: Path,
    path_dataset_root: Path,
    path_tracked_scene_root: Path,
    path_ground_effect_support_root: Path,
) -> dict[str, Any]
```

It reloads `summary.json`, reinstantiates `VirtualCrowd_Std`, reloads every
tracked scene and support, reconstructs `K`, repeats selection, then repeats the
unweighted -> KDE -> weighted sequence before comparing every persisted numeric
field and re-reducing summaries. It does not accept an in-memory producer
summary as authority. Do not add schemas, receipts, hashes, multiprocessing, or
a package CLI.

At reviewed sizes, exact KDE is `O(N^2)` time and `O(256N)` temporary working
set, with no persistent `[N,N]` array. The pre-design correctness probe measured
the same asymptotic method through a slower explicit-difference/einsum path; its
timings are only scale evidence, not a claim for the centered Cholesky +
chunked-cdist implementation. The real producer must be benchmarked after
implementation for wall time and peak working set. One KDE per scene in the
authorized single arm is expected to remain small relative to the inherited
fixed-grid RCR search; Cartesian runtime is intentionally not authorized or
claimed here.

The post-implementation real-input benchmark measured the KDE call alone from
the exact persisted observations and provisional normals. Across all eight
scenes its summed wall time was `1.509 s`; the largest scene (`N=4,950`) took
`0.643 s`. Its largest materialized distance block is exactly
`256 * 4,950 * 8 = 10,137,600` bytes (`9.67 MiB`), with no persistent full
pairwise array. Process-wide peak RSS was dominated by the surrounding dataset /
solver import and input stack, so it is not presented as KDE-owned memory.

## Smoke-Test Standard

Portable tests must cover:

- a hand/brute-force exact LOO log-density oracle with the diagonal excluded;
- an isolated observation proving no self-subtraction cancellation;
- uniform translation, rotation/reflection, normal sign/scale, and coordinate
  common-scale invariance of final weights;
- deterministic equal-area separated square lattices with 400 dense and 100
  sparse rows; require the weighted region-total ratio to be strictly closer to
  one in absolute log-ratio than the unweighted ratio `4.0`;
- a deterministic `21 x 21` finite-square lattice boundary diagnostic; define
  boundary as the outermost lattice ring and center as the central `9 x 9`
  rows, require boundary mean weight to exceed center mean weight, and retain
  both means as explicit test evidence rather than a false uniformity claim;
- duplicate observations with full-rank overall covariance;
- collapsed/collinear covariance, singular K, mixed/parallel rays, malformed
  dtype/shape, nonfinite inputs, invalid bounds, and insufficient N;
- one near-collinear but positive-definite population, proving either finite
  deterministic output or the exact Cholesky fail-closed boundary without
  silent regularization;
- exact mean-one weights, clip closure, ESS, owned read-only fields, and row
  preservation;
- a seam proving chunked evaluation excludes every global diagonal and does not
  allocate a persistent full pairwise matrix;
- single-arm operation wiring, identical observation/support identities,
  independent KDE/plane/objective reconstruction, raw summary re-reduction,
  exact two-variant output set, dry-run reviewed count, and help.

## Completion Criteria

This extension stops after:

1. both design layers and implementation close without remaining Critical or
   Concern;
2. portable tests and strict Pyright pass in both affected repositories;
3. all eight scenes produce unweighted and KDE results for the exact current
   selection arm and frozen evaluation support;
4. the numerical comparison and boundary/normal/distance limitations are given
   to the user for confirmation.

Cartesian expansion is a separate user decision after criterion 4.

## Cartesian Extension

The user authorized preparation, but not execution, of this exact product:

```text
confidence strict-greater-than in {4.0, 4.5, 5.0}
ankle-distance / bbox-width strict-less-than in {0.15, 0.20}
density mode in {filtered_unweighted, density_kde_scott_loo}
```

This yields six observation populations and 12 solver/evaluation configs. The
canonical order is confidence ascending, ankle ratio ascending, then density
mode in the order shown. Names are exact and path-safe:

```text
conf_gt_4p0__ankle_lt_0p15__filtered_unweighted
conf_gt_4p0__ankle_lt_0p15__density_kde_scott_loo
conf_gt_4p0__ankle_lt_0p20__filtered_unweighted
conf_gt_4p0__ankle_lt_0p20__density_kde_scott_loo
conf_gt_4p5__ankle_lt_0p15__filtered_unweighted
conf_gt_4p5__ankle_lt_0p15__density_kde_scott_loo
conf_gt_4p5__ankle_lt_0p20__filtered_unweighted
conf_gt_4p5__ankle_lt_0p20__density_kde_scott_loo
conf_gt_5p0__ankle_lt_0p15__filtered_unweighted
conf_gt_5p0__ankle_lt_0p15__density_kde_scott_loo
conf_gt_5p0__ankle_lt_0p20__filtered_unweighted
conf_gt_5p0__ankle_lt_0p20__density_kde_scott_loo
```

All 12 retain `H_prior=1.35 m`, the same GT-MOT detections, and the same frozen
167,243-row ordered **evaluation denominator**. Their ground-effect summaries
are therefore directly comparable. Their **solver observation populations**
vary with confidence/ankle policy and must not be intersected, padded, or
otherwise forced to share observations. Within each population, unweighted and
KDE share exact ordered observation identities, so that pair isolates the
density effect. Cross-confidence/ankle differences contain a selection effect
and must report the population observation count. The KDE uses that
population's own unweighted provisional normal and Scott bandwidth.

Strict-threshold nesting is mandatory per scene:

```text
at fixed ankle threshold: conf>5.0 subset conf>4.5 subset conf>4.0
at fixed confidence threshold: ankle<0.15 subset ankle<0.20
```

Both portable sentinel rows exactly equal to `4.0/4.5/5.0` and
`0.15/0.20`, and real-data count monotonicity, must prove the strict directions.
Every persisted config must carry `effect_frame_id/gt_track_id` exactly equal
to the same frozen support identities.

`VirtualCrowd_RCR_Cartesian_Config` is an operation-local frozen value record
with fields `name: str`, `population_name: str`,
`confidence_threshold_strict_gt: float`,
`maximum_ankle_bbox_width_ratio_strict_lt: float`, and `density_mode: str`.
The density mode is exactly one existing canonical token,
`filtered_unweighted` or `density_kde_scott_loo`; config-name suffixes use those
tokens verbatim. Its post-init reconstructs and validates both exact names and
axis membership. `virtualcrowd_rcr_cartesian_configs() ->
tuple[VirtualCrowd_RCR_Cartesian_Config, ...]` returns canonical order and
validates 12 unique names, six population keys, and exactly two canonical modes
per population. It is not promoted into `hjlib-evaluation/src` because these
axes and names are VirtualCrowd experiment policy.

The existing operation script extracts `write_plain_result(...)` and
`validate_plain_result(...)`. Both receive ordered config names, an explicit
config-to-key-set mapping, the summary, and `(config_name,scene)->payload`
mapping. The writer owns only plain I/O; the comparator owns exact file
inventory, summary, key-set, and array comparison. Single-arm and Cartesian
runners independently reconstruct their own expected source-derived artifacts
and then share these two functions. The Cartesian runner reuses the pure
`evaluate_scene` and collects each of the six populations once per scene before
producing its two density modes.

A sibling repo-local Typer script lives at
`script/evaluate_virtualcrowd_density_balanced_rcr_cartesian.py`. Its
`prepare_virtualcrowd_density_balanced_rcr_cartesian(*,
path_dataset_root: Path, path_tracked_scene_root: Path) -> dict[str,Any]` reads
only dataset metadata and tracked scenes. The CLI defaults to preparation and
requires only these two roots; optional support/output roots are untouched.
The returned JSON is exactly

```text
mode, configuration_count, population_count,
configs: ordered list of the five config fields,
populations: ordered list of population_name, both thresholds,
             selected_total, scene_counts in canonical scene order
```

Preparation creates no output root and never calls solve, KDE, support loading,
or evaluation. Real execution requires explicit `--execute` plus both
`--path-ground-effect-support-root` and `--path-output-root`; the typed
`run_virtualcrowd_density_balanced_rcr_cartesian(...)` has all four nonoptional
Path keyword arguments. It writes `summary.json` plus one directory per
canonical config and independently rebuilds the complete chain from the same
three explicit roots. It remains fail-new and has no schema/hash/receipt layer.

Portable gates must prove the exact config order/names, six-to-twelve sharing,
strict threshold semantics, dry-run no-solve/no-output behavior, single-arm
parity for `conf_gt_4p0__ankle_lt_0p20`, and compact full writer/readback
reconstruction. Preparation stops after the real-data dry-run counts and does
not invoke the real Cartesian runner.

## Modification History

- 2026-08-18: User rejected fixed kNN neighbor counts as the formal density
  axis and requested review of a general automatic density algorithm.
- 2026-08-18: Review accepted the Gaussian KDE family but found direct
  training-density self subtraction numerically invalid, finite-support
  boundary bias material, likelihood CV unnecessary/expensive, and full-rank
  covariance a required assumption. This layer freezes stable exact log-space
  LOO, Scott bandwidth, log-space inverse weighting, explicit boundary limits,
  and one selection arm before any Cartesian grid.
- 2026-08-18: Mathematical Architecture review found no Critical and four
  Concerns. Accepted all: centered lower-Cholesky whitening and an elementwise
  negative-distance tolerance are exact; clip bounds are generic inputs but
  fixed for this arm; ESS is defined; and dense/sparse plus boundary acceptance
  fixtures are deterministic.
- 2026-08-18: Code Architecture review found no Critical and three Concerns.
  Accepted all: the validator now receives and rebuilds from all three source
  roots; the complete typed keyword-only API and one canonical variant token
  are frozen; centered Cholesky/tolerance details are shared with the math
  closure. Prototype timing is explicitly not final implementation evidence.
- 2026-08-18: Mathematical re-review left two formula-level Concerns. Closed
  them by parameterizing the log clip with the public bounds and deriving the
  finite log kernel normalizer from Cholesky diagonals without `det(H)`.
- 2026-08-18: The implementation oracle exposed cancellation in the reviewed
  norm/BLAS distance identity for an extreme isolated observation. The design
  now requires chunked direct-difference `cdist`; it retains the same exact KDE
  definition and `O(256N)` temporary bound while removing that failure mode.
- 2026-08-18: Re-review found the chunk boundary and stored Scott covariance
  closure untested. The record now independently recomputes its Scott
  covariance, while an `N=257` wrapped-`cdist` oracle proves the global
  diagonal and exact `[256,N]` / remainder block shapes.
- 2026-08-18: The real producer performance gate measured `0.643 s` for the
  largest `N=4,950` scene and `1.509 s` summed over eight scenes. The maximum
  explicit distance block is `9.67 MiB`; process RSS is not attributed to KDE
  because imports and source loading dominate the process baseline.
- 2026-08-18: User authorized preparation of a 3 confidence x 2 ankle-ratio x
  2 density-mode Cartesian extension, totaling six observation populations and
  12 configs. Exact order/names, shared-population semantics, dry-run boundary,
  and future plain-output runner are frozen; real execution remains unauthorized.
- 2026-08-18: Cartesian preparation was implemented and run on the reviewed
  dataset/tracked inputs. Population totals are 9,427, 17,992, 6,770, 13,524,
  3,839, and 8,326. Portable gates prove strict threshold sentinels, exact
  single-arm parity, default prepare-only CLI behavior, and one collection per
  scene/population; focused Mathematical and Code Architecture re-reviews found
  no remaining Critical or Concern. No Cartesian solve/evaluation was run.
- 2026-08-18: The user subsequently authorized real Cartesian execution. All
  12 configs completed on the six reviewed populations and exact 167,243-row
  support, then passed independent source reconstruction and raw/summary
  readback. KDE lowers combined mean in all six pairs (8.44%--17.37%), with raw
  best `confidence>5.0, ankle<0.20, KDE = 15.727720 m`. That arm has only 256
  observations in its smallest scene, and every pair's support-weighted
  normal-oracle and distance-only means worsen, so low support and systematic
  error cancellation remain explicit limitations.
