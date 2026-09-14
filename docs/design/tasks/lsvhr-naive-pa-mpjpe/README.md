# LSV-HR NAIVE PA-MPJPE

Task-scoped Layered Design for adding PA-MPJPE to the LSV-HR `NAIVE`
evaluation profile without reinterpreting persisted four-metric results.

## Requirements

- Add `PA-MPJPE` to the ordered NAIVE metric roster after `T-MPJPE`.
- Evaluate every selected, direct-target-matched person-frame independently on
  all SMPL-24 joints.
- Fit prediction to GT with one positive-scale similarity transform whose
  rotation is proper (`det(R) = +1`); reflections are prohibited.
- Reduce over all `24 * selected_gt_count` joint occurrences. No person-frame,
  track, or scene mean receives equal-weight treatment.
- Reuse `hjlib-geometry` similarity registration. Do not add another
  Procrustes implementation.
- Keep the result-loader and association contracts unchanged. Crowd3D-style
  unassociated prediction handling remains outside this result/evaluation
  path.
- Preserve existing `naive-v1` artifacts as four-metric historical evidence.
  New computation and persistence use `VC_NAIVE_COMPARISON_METRICS_V2` and
  `naive-v2`; no artifact migration, overwrite, hash, or attestation is added.
- Re-evaluation starts from existing method results and GT loaders. It does not
  rerun model inference.

## Mathematical Architecture

For each selected matched person-frame `n`, let predicted and GT SMPL-24 world
joints be `P_n, G_n in R^(24x3)`, in metres. Independently fit:

```text
(s_n, R_n, t_n) = argmin sum[j] ||s_n R_n P_n[j] + t_n - G_n[j]||^2
subject to s_n > 0 and R_n in SO(3)
```

The repository uses row vectors, so the implementation applies the equivalent
`s_n * (P_n @ R_n.T) + t_n`. `hjlib-geometry` owns this positive-scale,
reflection-disabled fit.

```text
PA-MPJPE = 1000 / (24N)
            * sum[n=1..N, j=0..23]
                ||aligned(P_n)[j] - G_n[j]||_2
```

The result is millimetres and lower is better. The sufficient statistics are
`pa_mpjpe_sum_m` and `pa_mpjpe_count`; the count must equal the existing world
and pelvis-relative joint support, namely `24 * selected_gt_count`.

The fit fails under the exact existing geometry conditions: centered
prediction spread is non-positive, or correlation after the proper-rotation
fit is non-positive and therefore cannot produce a positive scale. Rank-
deficient support is otherwise accepted even when the fitted rotation is not
unique. A failed occurrence fails the evaluation; it is not dropped, because
dropping it would violate the requested all-occurrence denominator. Any
non-finite prediction or GT joint also fails the method-neutral leaf before
registration, so geometry cannot silently remove a joint from the fixed
`24N` support.

For `N = 0`, the unreduced leaf returns an empty `(0, J)` error array and the
NAIVE summary carries zero PA sum/count. Nullable finalization returns `None`,
matching the existing empty-selection contract.

All other NAIVE metrics and their reductions are unchanged. Scene and track
summaries add PA sums/counts, and scene/global merging uses additive sums and
counts before the final division.

## Code Architecture

`hjlib-evaluation` remains the numerical owner:

- `joint_error.py` adds one method-neutral leaf that accepts equal
  `(N, J, 3)` arrays, fits each occurrence independently through
  `fit_similarity_registration`, applies it, and returns unreduced `(N, J)`
  Euclidean errors. It requires a positive joint dimension and fully finite
  inputs; empty occurrence support is valid.
- `virtualcrowd_naive_comparison.py` exposes explicit V1 and V2 profile IDs,
  with the unqualified current ID pointing to V2. Its existing summary/result
  types become profile-aware: V1 permits no PA fields, while V2 requires PA
  sums/counts/result. This retains one evaluation-owned V1 finalization path
  for historical store validation without copying metric formulas into the
  persistence repository. Current evaluation always constructs V2.
- `naive_track_statistics.py` uses the same primitive before track partition
  reduction. Track, scene, and matrix routes therefore share one PA truth.
- `lsvhr_evaluation.py` requires the V2 result through the existing `NAIVE`
  enum; the enum name does not become a second metric-version identity.

`hjlib-experiments-results` remains the persistence/report/CLI owner:

- `evaluation_store.py` treats metric profile as the schema discriminator.
  It reads both historical `naive-v1` and current `naive-v2`, writes current
  V2 leaves, and validates version-specific exact fields. CLI comparison rows
  expose the V2 union; V1 rows carry no PA value and remain in their separate
  `profile=naive-v1` comparison group.
- `evaluation_execution.py` emits `evaluation_store.v2` / `naive-v2` records.
- registered-output and fast-comparison writers advance their report schema to
  V2 and add the `PA-MPJPE` column. Existing report files remain untouched.
- the aggregate report reader accepts both fast-comparison schema versions and
  maps each to its exact metric roster.
- comparison selection continues to use the four-field source identity
  `(protocol, method, entry, run)`, while deduplication and representation use
  the five-field evaluation-result identity that additionally contains
  `metric_profile`. Therefore one source/run may retain V1 and V2 side by side
  without collision, and the two rows form distinct profile groups.
- the existing public GVHMR candidate-evidence loader remains an explicit V1
  reader and returns a row with no PA value. Current V2 registered-output and
  fast-comparison report objects reject such a row; V2 PA must come from a new
  evaluation, not be synthesized from V1 evidence.

No loader/interpreter API changes, prediction materialization, compatibility
adapter, or new dependency are required.

## Smoke-Test Standard

Data-free tests must prove:

- exact similarity-transformed prediction has non-zero world/T error but PA
  errors no larger than `1e-10 * max(1, max(abs(G)))` metres;
- a reflected non-coplanar, full-rank pose cannot be reduced to zero
  PA-MPJPE;
- PA support is exactly `24 * occurrence_count` at track, scene, and global
  levels; different track/scene partitions agree within `1e-12` relative or
  absolute tolerance because floating-point summation grouping can differ;
- empty `(0, J, 3)` input returns `(0, J)` and produces nullable PA output,
  while any non-finite joint fails before registration;
- an undefined degenerate similarity fit fails instead of shrinking support;
- V2 metric/profile identities and report column order are exact;
- V2 stores round-trip with PA statistics and metrics;
- an existing V1 store still reads with no synthesized PA value, while new
  discovery finds V1 and V2 leaves as separate profiles;
- the same source/run may have V1 and V2 leaves simultaneously; comparison
  emits two rows in distinct profile groups rather than treating them as a
  duplicate identity;
- historical V1 GVHMR evidence still loads, but V2 report construction rejects
  it because PA is absent;
- terminal/CSV/Excel comparison exports include PA-MPJPE for V2 rows;
- registered-output and fast-comparison JSON/Markdown contain the V2 schema and
  PA column.

Run focused smoke tests, repository smoke tests, and every repository pyright
configuration after Python edits.

## Migration Plan

1. Review this mathematical and code architecture.
2. Implement and validate the numerical V2 contract in `hjlib-evaluation`.
3. Implement version-aware storage/report/CLI handling in
   `hjlib-experiments-results`.
4. Land stable design/usage documentation in both repositories.
5. Run focused tests, full smoke, pyright, and bounded implementation review.
6. Separately decide which registered method results to re-evaluate. Do not
   start a potentially long full evaluation as part of the code change.

## Modification History

- 2026-09-13: Requirements and both architecture layers recorded from the
  confirmed definition: independent person-frame SMPL-24 positive-scale proper
  similarity alignment, no reflection, micro-average over all joint
  occurrences, with V1 artifacts retained and new results versioned as V2.
- 2026-09-13: Mathematical review found no critical issue. Accepted concerns
  clarified exact geometry degeneracy conditions, finite/all-joint support,
  empty selection behavior, full-rank reflection evidence, and numeric
  tolerances.
- 2026-09-13: Code-architecture review found one critical identity collision
  and two compatibility concerns. Accepted changes split source identity from
  metric-versioned result identity, keep V1 finalization in the numerical
  owner through profile-aware types, and retain the GVHMR evidence loader as a
  V1-only reader that cannot enter a V2 report.
- 2026-09-13: Bounded code-architecture re-review confirmed the critical and
  both concerns are closed, with no new finding. Implementation may proceed.
- 2026-09-13: Implemented the reviewed V2 contract across evaluation and
  result-storage/report layers. Focused and full smoke suites, both master
  runners, and strict pyright passed; no real evaluation artifact was run or
  rewritten.
- 2026-09-13: Implementation review closed V1 merge/read-only-writer behavior,
  explicit historical-version constants, source-vs-result selection, nullable
  evidence deduplication, current GVHMR evidence schemas, fast-comparison V2
  composition, micro-reduction sentinels, package exports, and master-runner
  coverage. Bounded re-review found no remaining code finding. The downstream
  dependency pin intentionally remains unchanged until this uncommitted
  evaluation delta receives a real commit identity.
