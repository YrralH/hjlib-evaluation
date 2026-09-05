# VirtualCrowd Naive Comparison Metrics

Task-scoped Layered Design for a narrow provisional metric profile used to put
Ours, Crowd4D and DyCrowd under one `vc.visible_common × vc.test6` comparison.

## Requirements

- Add one method-neutral profile identified by
  `VC_NAIVE_COMPARISON_PROFILE_ID = 'VC_NAIVE_COMPARISON_METRICS_V1'`, with
  its exact ordered display roster separately exposed as
  `VC_NAIVE_COMPARISON_METRICS`:
  `MPJPE-WORLD`, `T-MPJPE`, `OKS-VIS`, and `ACC-ROOT-RATIO`.
- Reuse the already reviewed `Corrected_Crowd_Sequence`, its direct-identity
  target mapping, a caller-supplied selected-population mask, the COCO-17
  visibility-aware OKS leaf and exact micro-reduction semantics. Do not
  introduce a second prediction schema or another association rule.
- Accept one selected scene at a time and return additive sufficient
  statistics bound to separate canonical `filtering_id` and `split_id` fields.
  A separate reducer combines scenes so partial/restartable method
  adapters do not average already-reduced scene means.
- Keep dataset/prediction adaptation outside this library. Ours wiring belongs
  to `hjlib-experiments`; Crowd4D and DyCrowd private prediction adapters belong
  to `hj-tpa-crowd4d`.
- This profile is deliberately narrow and provisional. It is not an alias for
  `VC_HJ_DEFAULT_METRICS_V1` or `VC_CROWD4D_NATIVE_METRICS_V1`, and it does not
  modify either stable profile.
- No model inference, file discovery, result registration, hash, artifact
  attestation or output publication is part of the evaluating function.

## Mathematical Architecture

All methods are evaluated after direct identity alignment on the exact selected
population `vc.visible_common × vc.test6`. Let accepted paired occurrences be
`n = 1..N`, SMPL-24 joints be `j = 0..23`, and joint `0` be the pelvis/root.
World coordinates are metres on input.

### MPJPE-WORLD

```text
MPJPE-WORLD = 1000 / (24N)
              * sum[n,j] ||P[n,j] - G[n,j]||_2
```

The result is in millimetres and lower is better.

### T-MPJPE

```text
P_local[n,j] = P[n,j] - P[n,0]
G_local[n,j] = G[n,j] - G[n,0]
T-MPJPE = 1000 / (24N)
          * sum[n,j] ||P_local[n,j] - G_local[n,j]||_2
```

The result is in millimetres and lower is better. This is pelvis-relative pose
error, not an optimized translation fit.

### OKS-VIS

Use the existing corrected-crowd identity-paired OKS semantics. Let `V` be the
subset of selected matched rows with at least one GT native-visible COCO joint:

```text
e[n,k] = ||p2d[n,k] - g2d[n,k]||²
         / ((2 sigma[k])² * bbox_area[n] * 2)
OKS[n] = mean(exp(-e[n,k]) for GT joints with native visibility > 0)
OKS-VIS = mean[n in V] OKS[n]
```

`k` is the conventional COCO-17 ordering and sigma vector already owned by the
corrected-crowd evaluator. `bbox_area` comes from GT `bbox_xyxy`. Predicted 2D
joints must already be produced by the method adapter using that method's
declared camera; the metric does not choose or substitute a camera. The value
is a pure conditional micro mean in `[0, 1]`; higher is better. It has no recall
multiplier and does not perform OKS matching. A selected row with no visible
COCO joint remains part of the MPJPE and temporal populations but contributes
no OKS sample. The summary therefore carries an independent `oks_vis_count`.
An individual scene may have zero OKS support; the global reducer raises only
when total OKS support is zero. For every native-visible joint in each retained
OKS row, its paired prediction's source
`prediction_coco17_camera_depth_m` must be finite and strictly positive;
failure raises rather than silently dropping the joint or row.

### ACC-ROOT-RATIO

This intentionally reuses the repository's earlier `ACC RATIO` meaning rather
than renaming `ACC-ROOT` residual error. For every maximal exact-consecutive
matched identity run, take root trajectories `r[0:T]`. Apply the existing
replicate-padded central first-difference operator twice:

```text
D(x)[t] = 0.5 * (pad(x)[t+2] - pad(x)[t])
A(x) = D(D(x))
```

Trim three values from each end. Runs with `T <= 6` have no ratio support and
are omitted. Across all retained samples:

```text
ACC-ROOT-RATIO = sum ||A(root_pred)||_2 / sum ||A(root_gt)||_2
```

No FPS multiplier is applied, so the ratio is unitless. Its reference value is
`1`; neither monotonically lower nor higher is intrinsically better. Empty
support or a non-positive global GT denominator raises instead of emitting a
sentinel.

### Reduction and invariants

- MPJPE and T-MPJPE reduce over joint-occurrence samples.
- OKS-VIS reduces over identity-paired person-frame samples with at least one
  native-visible GT COCO joint. It does not redefine or shrink the selected
  population used by the other metrics.
- ACC-ROOT-RATIO reduces numerator and denominator globally, never by averaging
  per-run or per-scene ratios.
- Scene summaries carry sums and counts. The reducer sorts by `scene_id`,
  rejects duplicate scenes and mismatched profile/filtering/split identity, and uses
  `math.fsum` across scene contributions.
- Every selected occurrence must have exactly one prediction whose existing
  `prediction_identity_target_gt_rows` value points to that GT row. Before any
  metric is computed, the evaluator constructs this exact join and enforces
  `matched_selected_count == selected_gt_count`; zero or multiple predictions
  for a selected GT row are completeness/identity failures. This path is used
  instead of `prepare_selected_rows`, whose base-visible gate would wrongly
  exclude internal zero-visible occurrences that the population contract keeps.

## Code Architecture

Code home:

```text
src/hjlib_evaluation/virtualcrowd_naive_comparison.py
```

Public surface:

```python
VC_NAIVE_COMPARISON_PROFILE_ID
VC_NAIVE_COMPARISON_METRICS
VirtualCrowd_Naive_Comparison_Sequence_Summary
VirtualCrowd_Naive_Comparison_Result
evaluate_virtualcrowd_naive_comparison
reduce_virtualcrowd_naive_comparison_summaries
```

`evaluate_virtualcrowd_naive_comparison(sequence, filtering_id, split_id,
selected_gt_mask)` reuses:

- `validate_corrected_crowd_sequence` for the normalized schema;
- `compute_joint_position_errors` for MPJPE populations;
- `compute_keypoint_oks_matrix` and `COCO17_SIGMAS` for visibility-aware OKS.

The module owns the narrow exact-target join, gap-aware grouping by native GT
track/frame identity, per-metric sufficient-statistic collection and additive
root-ratio statistics. OKS is computed per native frame with the existing
matrix leaf and its paired diagonal; this avoids a sequence-global quadratic
matrix while preserving the reviewed visibility-aware formula. It does not
call the full corrected-crowd reducer merely to discard most of its output, and
it does not copy OKS math. No `utils`/`helpers` module or mutable accumulator
class is added.

The exact frozen, slotted summary fields are:

```text
profile_id: str
filtering_id: str
split_id: str
scene_id: str
selected_gt_count: int
matched_selected_count: int
mpjpe_world_sum_m: float
mpjpe_world_count: int
t_mpjpe_sum_m: float
t_mpjpe_count: int
oks_vis_sum: float
oks_vis_count: int
acc_root_predicted_sum_m_per_frame2: float
acc_root_reference_sum_m_per_frame2: float
acc_root_sample_count: int
```

Its invariants include `profile_id == VC_NAIVE_COMPARISON_PROFILE_ID`, non-empty
filtering/split/scene identities, exact non-negative integer counts, finite
non-negative sums, selected/matched equality, both MPJPE counts equal to
`24 * selected_gt_count`, `0 <= oks_vis_count <= selected_gt_count`,
`0 <= oks_vis_sum <= oks_vis_count`, and zero sums whenever their support count
is zero. A per-scene summary may have zero OKS or ACC support.

The exact frozen, slotted result fields are:

```text
profile_id: str
filtering_id: str
split_id: str
scene_count: int
selected_gt_count: int
matched_selected_count: int
joint_sample_count: int
oks_vis_count: int
acc_root_sample_count: int
mpjpe_world_mm: float
t_mpjpe_mm: float
oks_vis: float
acc_root_ratio: float
```

The reducer requires at least one scene, one selected occurrence, non-zero
global OKS and ACC supports, and a strictly positive global ACC reference sum.
It rejects profile/filtering/split mismatches and duplicate scene IDs. Neither
object owns JSON or filesystem serialization in this task. Result construction
repeats the profile equality, positive scene/population/support,
selected/matched equality and
`joint_sample_count == 24 * selected_gt_count` invariants; all metrics are
finite and non-negative, with `0 <= oks_vis <= 1`.

`src/hjlib_evaluation/__init__.py` re-exports the six public symbols. No new
dependency is introduced.

## Smoke-Test Standard

Data-free tests must prove:

- exact zero-error geometry on a non-linearly moving root gives zero
  MPJPE/T-MPJPE, OKS one and acceleration ratio one;
- a constant world translation changes MPJPE-WORLD but leaves T-MPJPE,
  OKS (when 2D inputs are unchanged), and acceleration ratio unchanged;
- predicted root acceleration scaling produces the expected global ratio and
  is reduced from sums rather than mean-of-ratios;
- GT visibility excludes hidden COCO-17 joints, GT bbox area controls OKS, a
  zero-visible selected row remains in the full population while adding no OKS
  support, and a native-visible joint with non-positive prediction camera depth
  fails explicitly;
- non-consecutive frames are split before temporal evaluation and short runs
  contribute no acceleration samples;
- missing selected predictions, duplicate predictions targeting one selected
  GT row, duplicate scene summaries, invalid shapes, empty support and a zero
  GT acceleration denominator fail explicitly;
- reducer output is invariant to input scene-summary order;
- the public import surface is available from `hjlib_evaluation`.

Run repository smoke and strict pyright after implementation. Coverage self-
check must include the new module and test.

## Migration Plan

1. Land and review this task design.
2. Implement the new leaf/composition module and public exports.
3. Add synthetic smoke for math, support and reduction invariants.
4. Land stable design/usage navigation without changing existing profile
   identities.
5. Run checks and implementation review.

No existing result artifact or metric profile is migrated or rewritten.

## Modification History

- 2026-09-01: Initial requirements, mathematical architecture, code
  architecture and smoke standard recorded. `ACC-ROOT-RATIO` reuses the earlier
  global predicted/GT pelvis acceleration-magnitude ratio; `OKS-VIS` is selected
  instead of the author evaluator's matching- and recall-coupled `OKS`.
- 2026-09-01: Mathematical and code-architecture reviews found that the stable
  corrected-crowd selected-row helper excludes zero-visible internal population
  rows and does not enforce complete prediction coverage. The design now joins
  through the existing direct-identity target mapping, requires exactly one
  prediction per selected GT row before metric work, gives OKS its own visible
  support, freezes all summary/result fields, and specifies a bounded per-frame
  reuse path for the existing OKS leaf.
- 2026-09-01: Re-review closed the original findings and requested the remaining
  constructor, unit and source-depth details. The contract now pins profile
  identity, result invariants, OKS bounds, unit-bearing ACC sum fields,
  duplicate-target smoke and strict positive source depth for every evaluated
  OKS joint.
- 2026-09-01: Delta review restored the registry-owned population identity
  boundary: summaries, results and evaluator arguments now carry canonical
  `filtering_id` and `split_id` separately instead of inventing a combined
  `population_id` string.
