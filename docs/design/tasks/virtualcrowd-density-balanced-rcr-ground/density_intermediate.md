# Density Intermediate Representation

## Status

This layer is implemented. Mathematical and Code Architecture design and
implementation reviews are closed with no remaining Critical or Concern.

## Scope

The density intermediate is a generic `hjlib-ground-solver` numerical object
between an unweighted provisional normal and a weighted normal fit. It uses only
bottom image observations, one camera intrinsic matrix, and the provisional
normal. It must not use GT ground, scene names, track identities, evaluation
errors, top-joint locations, or result-artifact metadata.

The intermediate answers one question: how spatially repeated is each retained
ground-contact observation on the provisional ground plane? It does not solve a
new normal or distance and does not decide which observations are retained.

## Mathematical Architecture

### Inputs and validity

Inputs are:

- `bottom_xy_px`: finite float64 `[N,2]` image coordinates;
- `camera_K`: finite, nonsingular float64 `[3,3]` intrinsics;
- `provisional_normal_camera`: finite, nonzero float64 `[3]` camera-frame
  normal; and
- integer `neighbor_count=k`, with `N >= k + 1` and `k >= 1`.

`N` is the exact observation population to be weighted. The function neither
drops nor reorders rows. The normal is normalized internally. Its sign and
input scale must not affect any distance, density, or weight.

For homogeneous pixel `p_i=[u_i,v_i,1]^T`, construct the camera ray

```text
r_i = solve(K, p_i).
```

Let `s_i=n^T r_i`. Every ray must have finite positive Euclidean norm. Every
`s_i` must be finite, have absolute normalized ray-normal cosine
`|s_i|/||r_i||_2 > 1e-10`, and have the same strict sign. Mixed signs mean the
observations cannot all intersect one forward plane with this normal and fail
the whole call.

Choose the sign-equivalent unit-distance plane by

```text
d_unit = -sign(median_i(s_i)),
lambda_i = -d_unit / s_i,
X_i = lambda_i r_i.
```

All `lambda_i` must be finite and positive. `|d_unit|=1` fixes a common
arbitrary scale. Replacing it conceptually with any other positive absolute
distance multiplies all `X_i` and kNN radii by one scalar; the normalized final
weights below are unchanged before floating-point tolerance.

The density module constructs the rays and performs the same-sign gate needed
to choose `d_unit`, then delegates the actual forward intersections and
normalized-cosine validation to the existing
`hjlib_geometry.intersect_rays_with_planes` public API using zero origins and
one repeated unit-distance plane. It does not implement a second ray-plane
intersection routine.

### Deterministic 2D ground coordinates

Choose the Cartesian camera axis least aligned in absolute dot product with
unit normal `n`. With ties resolved in x, y, z order, construct

```text
e1 = normalize(cross(n, chosen_axis)),
e2 = cross(n, e1),
q_i = [e1^T X_i, e2^T X_i].
```

`q` has shape `[N,2]` and units of unit-plane-distance. The frozen basis makes
the intermediate inspectable and reproducible. Any normal sign flip changes at
most the 2D orthonormal basis orientation and therefore cannot change pairwise
distances.

### kNN density and inverse-density weight

Build one Euclidean `cKDTree` over all `q_i`. Query `k+1` neighbors for each
row and take column `k` of the sorted distances, so the row itself occupies one
zero-distance entry and `r_i^(k)` is the distance to its kth other observation.
Ties use distance only; neighbor identities do not affect the radius.

Exact or numerically repeated coordinates may give zero kNN radii. Define:

```text
r_reference = median({r_i^(k) | r_i^(k) > 0})
r_floor = 1e-6 * r_reference
r_eff_i = max(r_i^(k), r_floor)
rho_i = k / (pi * r_eff_i^2)
w_raw_i = 1 / rho_i
```

If every raw kNN radius is zero but multiple unique locations exist,
`r_reference` is instead the median nearest-neighbor distance among unique
locations. Only a population with one unique location is entirely collapsed
and fails. `rho` has inverse-square unit-plane-distance units and
`w_raw` has square unit-plane-distance units. The relative inverse-density
weight is

```text
w_relative_i = w_raw_i / median(w_raw).
```

The task uses finite stabilizing bounds `0.25` and `4.0`. Configurable bounds
must be Python floats satisfying
`0 < minimum_pre_normalization_weight <= 1 <=
maximum_pre_normalization_weight < inf`:

```text
w_clipped_i = clip(
    w_relative_i,
    minimum_pre_normalization_weight,
    maximum_pre_normalization_weight,
)
weight_normalization_factor = mean(w_clipped)
w_i = w_clipped_i / weight_normalization_factor.
```

Thus `mean(w)=1`, `sum(w)=N`, and absolute provisional-plane scale cancels.
The clipping bounds apply before final mean normalization, so final weights are
not required to lie inside the input bounds; their dynamic range is at most
`maximum_pre_normalization_weight / minimum_pre_normalization_weight`. Clipping deliberately
prevents an isolated observation or a repeated-coordinate cluster from
receiving an unbounded or zero contribution.

This is an empirical density-balancing heuristic without finite-support
boundary correction. For a locally stationary 2D point process away from the
support boundary, `r_k^2` approximates inverse local sampling density, so the
sum of weights over equal-area regions with different sampling counts should be
closer than their unweighted counts. Near an unknown finite boundary, the kNN
disk extends outside the observed support and can over-weight boundary rows.
The method does not define or estimate that support and makes no claim of an
unbiased physical crowd-density field or exact spatial-uniform measure. ESS
only reports weight concentration; it does not validate spatial uniformity.
The normalized coefficients balance empirical spatial density, but final SVD
leverage also includes the inherited unnormalized homogeneous line scale;
therefore this baseline does not claim exact final-fit contribution equality.

Density is computed once from the unweighted provisional normal. The method
does not iterate density and normal to a fixed point. A wrong provisional
normal may therefore distort relative ground coordinates, not merely their
common scale; this limitation is part of the named one-pass baseline.

The effective sample size diagnostic is

```text
ESS = (sum_i w_i)^2 / sum_i(w_i^2),
```

with `1 <= ESS <= N` up to float64 tolerance. Uniform spatial observations
produce near-uniform weights and ESS near `N`; oversampled regions receive
smaller per-row weights.

### Output invariant and invalid policy

The output preserves exact input row order and contains these deliberately
unit-explicit fields:

- `provisional_unit_plane_xy`: float64 `[N,2]`;
- `knn_radius_unit_plane`: float64 `[N]`, the unfloored kth-neighbor radius;
- `effective_knn_radius_unit_plane`: float64 `[N]`;
- `empirical_knn_density_per_unit_area`: float64 `[N]`;
- `relative_inverse_empirical_density`: float64 `[N]` before clipping;
- `clipped_relative_inverse_empirical_density`: float64 `[N]` before mean
  normalization;
- `normalized_observation_weights`: float64 `[N]`, clipped and mean-normalized;
- `neighbor_count`, `radius_floor_unit_plane`,
  `minimum_pre_normalization_weight`, `maximum_pre_normalization_weight`,
  `weight_normalization_factor`, and `effective_sample_size` scalars.

All numerical outputs must be finite. Array outputs are owned copies and
read-only. Any invalid member, singular camera, parallel/mixed-sign ray,
collapsed population, invalid bound, or invariant failure rejects the complete
call; density construction never changes the evaluation denominator.

## Code Architecture

Add one method-neutral module:

```text
hjlib-ground-solver/src/hjlib_ground_solver/estimate_ground/
    observation_density.py
```

It owns:

- frozen `Ground_Observation_Density` with the exact fields above and
  validate-once owned read-only arrays;
- public pure `compute_ground_observation_density(bottom_xy_px, camera_K,
  provisional_normal_camera, neighbor_count,
  minimum_pre_normalization_weight=0.25,
  maximum_pre_normalization_weight=4.0)`.

`Ground_Observation_Density.__post_init__` validates every invariant that does
not require rebuilding the tree: exact Python/scalar and array dtypes, shared
row counts, shapes, finiteness, nonnegative/raw and positive/effective radii,
valid bounds and k, `effective_radius=max(raw_radius,floor)`, density formula,
inverse-density median normalization, clipping, stored normalization factor,
mean-one final weights, and ESS. It owns and write-protects all arrays. The
semantic claim that stored raw radii are the exact kNN radii of stored
coordinates is guaranteed by the sole public producer and its tests; the
constructor intentionally does not pay for a second `cKDTree` build.

The implementation uses NumPy float64, reuses the existing direct
`hjlib-geometry` dependency for ray-plane intersections, and uses the existing
direct SciPy dependency for one `scipy.spatial.cKDTree`. The exact query is
Euclidean with `eps=0.0`, `p=2.0`, and `workers=1`; only distances are consumed.
It has no torch/device state, filesystem I/O,
cache, random generator, dataset adapter, or solver callback. Internal
numerical operations remain ordinary noun-named functions only if they are
shared by more than one public operation; otherwise they stay inline to avoid a
helper layer.

The public class/function are re-exported from
`hjlib_ground_solver.estimate_ground` and the package root. No corresponding
API is added to `hjlib-evaluation`; that package only stores the returned arrays
alongside a concrete experiment result.

Cost is one camera solve plus one tree construction/query:
`O(N log N + Nk)` time and `O(Nk)` temporary memory. The largest reviewed scene
has `N=4,950` and `k<=64`, so a full `(N,k+1)` float64 distance matrix is an
explicitly accepted small intermediate. The call constructs the tree exactly
once.

## Smoke-Test Standard

Portable tests live in a separate
`test_smoke/test_observation_density.py`, are wired into the master smoke
runner, and must cover:

- a hand-solvable plane/basis coordinate case;
- normal positive scale invariance, plus sign invariance of pairwise distances,
  density, and weights while allowing `provisional_unit_plane_xy` to differ by
  an orthogonal basis transform;
- an algebraic oracle showing that rescaling the conceptual unit plane scales
  coordinates/radii but leaves normalized weights unchanged;
- symmetric-equivalent lattice points and ESS behavior rather than requiring
  every finite-lattice boundary point to have identical weight;
- two equal-area regions with deliberately different sample counts, requiring
  weighting to make their total contributions closer than the unweighted
  count ratio, plus a separate synthetic check documenting boundary over-weight
  without treating ESS as a spatial-uniformity proof;
- deterministic row preservation and repeated-coordinate radius flooring for
  duplicate multiplicity both no greater than `k` and greater than `k`;
- `k=16,32,64` on a synthetic population large enough for all three;
- singular K, nonfinite inputs, insufficient N, invalid k/bounds,
  parallel/mixed-sign rays, and wholly collapsed coordinates;
- exact `mean(normalized_observation_weights)=1` within float64 tolerance,
  finite outputs, ESS bounds,
  exact reconstruction of clipped/final weights from stored bounds and
  normalization factor, and read-only owned arrays; and
- a counting seam or monkeypatch proving one tree construction/query rather
  than per-observation reconstruction.

## Modification History

- 2026-08-18: Residence created as a separately reviewed intermediate layer.
- 2026-08-18: Froze unit-distance forward projection, deterministic tangent
  basis, kNN inverse-density weights, duplicate-radius floor, robust clipping,
  mean-one normalization, ESS, exact output fields, and fail-closed policy for
  dedicated review.
- 2026-08-18: Mathematical Architecture review reported no Critical and five
  Concerns. Accepted all: the method is now explicitly a one-pass empirical
  heuristic without boundary correction; clipping validity and reconstructible
  diagnostics are exact; normalized cosine is defined; and smoke criteria use
  testable invariants rather than exact finite-lattice uniformity.
- 2026-08-18: Mathematical re-review found one remaining formula ambiguity.
  Accepted and closed it by expressing clipping through the configurable bounds
  and defining the stored normalization factor as `mean(w_clipped)`.
- 2026-08-18: Code Architecture review reported no Critical and three
  Concerns. Accepted all: intersection delegates to `hjlib-geometry`; the
  constructor validates all cheap algebraic invariants while the sole producer
  owns kNN semantics; persistent fields now state unit-plane, empirical, and
  pre-normalization meanings explicitly. Exact deterministic tree parameters
  and a separate smoke module are also frozen.
- 2026-08-18: Implemented the reviewed intermediate in `hjlib-ground-solver`.
  Mathematical implementation review closed at 0 Critical / 0 Concern. Code
  implementation review found one malformed-container validation Concern; it
  was fixed with a shared exact-array validator and regressions, and re-review
  closed at 0 Critical / 0 Concern. Focused density smoke is 6 passed and
  targeted strict Pyright is 0 errors.
