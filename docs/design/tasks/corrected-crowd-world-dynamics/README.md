# Corrected-Crowd World Dynamics

## Requirements

Add an explicitly additive world-dynamics result to corrected crowd evaluation.
The frozen corrected schema-v1 metrics and their serialized results remain
byte-semantically unchanged. The new result evaluates four GT-relative temporal
metrics on an explicitly supplied selected GT population:

```text
ACC-JOINT
ACC-ROOT
JERK-JOINT
JERK-ROOT
```

All four metrics use `Corrected_Crowd_Sequence` world-space SMPL-24 joints,
the existing explicit GT association, exact source frame IDs, and one named
selected GT mask. They do not rerun a method, decode images, infer association,
smooth trajectories, align poses, or bridge missing frames.

The selected-mask contract is exactly the existing selected-view contract:
`view_name` is non-empty and not one of the two reserved legacy view names;
`selected_gt_mask` has exact boolean dtype and shape `[G]`; and every selected
row is in the base GT-visible domain `any(gt_visibility_native > 0)`. Numeric
arrays are never coerced to boolean masks.

The first executable target is GroupRec on
`C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9`, together with the already accepted 15
selected-view metrics. This operation reinterprets accepted GroupRec parameters
and reruns evaluation only; it does not rerun GroupRec inference.

### Reference status

- `ACC-JOINT` has a direct evaluation precedent in VIBE, MEVA, and RoHM:
  predicted-versus-GT 3D-joint acceleration residual.
- `ACC-ROOT` has a direct root-acceleration-error precedent in Bae et al.,
  *Versatile Physics-based Character Control with Hybrid Latent
  Representation* (Computer Graphics Forum 2025).
- `JERK-JOINT` is a GT-relative extension. PIP and GVHMR directly support
  global joint jerk magnitude as a jitter/smoothness metric, but not this exact
  GT-residual form.
- `JERK-ROOT` is the corresponding root-only GT-relative extension. No exact
  named HMR precedent is claimed.

Primary references:

- Kocabas et al., VIBE, CVPR 2020,
  <https://arxiv.org/abs/1912.05656>.
- Luo et al., MEVA, ACCV 2020,
  <https://openaccess.thecvf.com/content/ACCV2020/html/Luo_3D_Human_Motion_Estimation_via_Motion_Compression_and_Refinement_ACCV_2020_paper.html>.
- Zhang et al., RoHM, CVPR 2024,
  <https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RoHM_Robust_Human_Motion_Reconstruction_via_Diffusion_CVPR_2024_paper.html>.
- Bae et al., Versatile Physics-based Character Control, CGF 2025,
  <https://arxiv.org/abs/2503.12814>.
- Yi et al., PIP, CVPR 2022,
  <https://openaccess.thecvf.com/content/CVPR2022/html/Yi_Physical_Inertial_Poser_PIP_Physics-Aware_Real-Time_Human_Motion_Tracking_From_CVPR_2022_paper.html>.
- Shen et al., GVHMR, SIGGRAPH Asia 2024,
  <https://arxiv.org/abs/2409.06662>.

## Mathematical Architecture

Let `X[p,t,j]` and `Y[p,t,j]` be predicted and GT world-space SMPL joint
positions in metres, with `j=0` the SMPL pelvis/root joint. No per-frame or
sequence alignment is applied.

For one exact-consecutive triple `(t-1,t,t+1)`:

```text
d2_X(t,j) = X(t+1,j) - 2 X(t,j) + X(t-1,j)
d2_Y(t,j) = Y(t+1,j) - 2 Y(t,j) + Y(t-1,j)
e_acc(t,j) = ||d2_X(t,j) - d2_Y(t,j)||_2
```

For one exact-consecutive quadruple `(t-1,t,t+1,t+2)`:

```text
d3_X(t,j) = X(t+2,j) - 3 X(t+1,j) + 3 X(t,j) - X(t-1,j)
d3_Y(t,j) = Y(t+2,j) - 3 Y(t+1,j) + 3 Y(t,j) - Y(t-1,j)
e_jerk(t,j) = ||d3_X(t,j) - d3_Y(t,j)||_2
```

The metric populations are:

```text
ACC-JOINT  = micro mean of e_acc(t,j)  over every eligible window and j=0..23
ACC-ROOT   = micro mean of e_acc(t,0)  over every eligible window
JERK-JOINT = micro mean of e_jerk(t,j) over every eligible window and j=0..23
JERK-ROOT  = micro mean of e_jerk(t,0) over every eligible window
```

An eligible window contains one GT track's exact consecutive frame IDs, and
every occurrence in the window must be selected and explicitly matched. Missing
or unselected frames split support; rows are never compacted across a gap. A
window may overlap its neighbours. Results use float64 arithmetic and are
reported in `mm/frame^2` and `mm/frame^3`; no FPS factor is applied.

`ACC-JOINT` is mathematically identical to legacy selected-view
`ACCEL-WORLD`. On every non-empty result their values and exact triple-window
counts must agree. The combined evaluator collects acceleration once and passes
the same array into both consumers, so its parity is exact rather than
tolerance-based: per-scene raw metre sum, sample count, and triple count must be
bit-equal before display conversion. Standalone evaluators share the same exact
segment splitter and acceleration primitive, but collect independently. This is
a compatibility invariant and an implementation smoke gate.

Scene summary sufficient-statistic sums remain in `metre/frame^2` and
`metre/frame^3`; reduction converts the final means by `1000` to display units.
Every accepted summary enforces:

```text
count(ACC-JOINT)  = 24 * triple_count
count(ACC-ROOT)   = triple_count
count(JERK-JOINT) = 24 * quadruple_count
count(JERK-ROOT)  = quadruple_count
```

Zero count requires zero sum. Cross-scene reduction sorts summaries by lexical
`scene_id`, accumulates each metric with `math.fsum`, and divides the global
sum by the global count. Input summary order therefore cannot change a result.

Prediction-only derivative magnitude is outside this contract. A future
smoothness result must use an explicit `JITTER-*` name rather than changing the
four GT-relative metrics.

## Code Architecture

Stable owner: `hjlib-evaluation`.

```text
src/hjlib_evaluation/joint_jerk.py
    compute_joint_jerk_errors(...)

src/hjlib_evaluation/corrected_crowd_world_dynamics.py
    Corrected_Crowd_World_Dynamics_Sequence_Summary
    Corrected_Crowd_World_Dynamics_Result
    evaluate_corrected_crowd_world_dynamics(...)
    evaluate_corrected_crowd_selected_view_and_world_dynamics(...)
    reduce_corrected_crowd_world_dynamics_summaries(...)
    versioned JSON serialization

src/hjlib_evaluation/corrected_crowd_protocol.py
    legacy acceleration consumer and shared exact segment traversal
```

The generic acceleration primitive remains the existing
`compute_joint_acceleration_errors`; jerk receives the parallel
`compute_joint_jerk_errors`. Exact windows are grouped into maximal contiguous
matched+selected track segments, then each primitive is called once per segment
and emits temporal-major, joint-minor rows. Segment and row order preserve the
legacy stable track/frame traversal and therefore the exact acceleration sample
order. The legacy acceleration consumer remains owned by
`corrected_crowd_protocol.py`; the additive module imports the shared exact
segment traversal and generic acceleration primitive in one direction.
Removing the additive profile therefore cannot remove a legacy dependency.

The evaluator validates through the existing `Corrected_Crowd_Sequence` gate,
then applies the same name/mask validation as the selected-view evaluator and
performs an explicit matched-row join. In the combined entry, the additive
module collects acceleration once and supplies it to the legacy metric core,
preserving exact row and summation order. A sibling jerk traversal computes the
two jerk populations and exact quadruple count. Generic finite-difference math
remains method-neutral.

No metric is appended to `CORRECTED_CROWD_METRICS`; legacy summary/result
shapes and JSON remain unchanged. The additive dynamics summary independently
binds `scene_id`, `view_name`, selected/matched counts, metric order/units,
sufficient-statistic sums/counts, and exact window counts.

The combined public entry
`evaluate_corrected_crowd_selected_view_and_world_dynamics` validates and makes
one immutable sequence snapshot, prepares the selected rows once, then invokes
the existing selected-view core and the additive dynamics core. The standalone
dynamics entry remains available for callers that do not need the legacy
metrics. GroupRec uses the combined entry; it must not deep-copy/preflight the
same full scene twice.

Producer adapter owner: `hj-tpa-grouprec`.

The existing corrected scene worker computes the legacy selected-view summary
and the additive dynamics summary from the same in-memory
`GroupRec_Corrected_Scene_Build`. The worker envelope and operation receipt are
versioned forward and bind both serialized results. The operation publishes:

```text
result.json
world_dynamics_result.json
receipt.json
```

The GroupRec operation manifest advances to version 2 and includes an exact
`output_contracts` identity: the legacy selected-view schema/version and metric
order plus the dynamics schema/version, metric order, and unit order. A v2
worker rejects a manifest lacking or changing this identity. Existing v1
output roots remain readable but are never resumed or rewritten as v2.

The producer, interpreter, association, population mask, and model execution
are unchanged. Other methods can later call the stable evaluator without
depending on GroupRec-private schemas.

## Smoke-Test Standard

The committed data-free stable-owner gates cover:

1. analytic constant, linear, quadratic, and cubic trajectories;
2. vector L2 residual rather than coordinate absolute-value reduction;
3. joint/root population sizes and root=`SMPL-24[0]`;
4. exact triple/quadruple windows, overlap, gaps, unselected rows, and unmatched
   rows;
5. empty dynamics populations produce `None`, never zero;
6. `ACC-JOINT == ACCEL-WORLD`, including raw metre sums, sample counts, and
   exact triple counts;
7. per-scene sufficient-statistic reduction equals direct concatenation;
8. JSON round trip, numeric-mask rejection, and count/window invariant drift;
9. selected-view equivalence with old-common, COCO-17 GE9 half-visibility, and
   reserved-name rejection;
10. a TPA data-free selected-view golden constructs a fixed 15-metric result,
   serializes it through `corrected_crowd_selected_view_result_to_json` and
   `collector.canonical_json_bytes`, and compares exact frozen bytes. This locks
   fields, metric/unit order, `None`, numbers, and the actual publication format
   after introducing the combined acceleration path.

Broader malformed JSON field/order/unit and non-finite/negative matrices remain
normal schema hardening; they are not claimed as present commit gates here.

GroupRec tests must cover one scene worker envelope containing both summaries,
operation loading/reduction of both, no-clobber behavior, and receipt hashes for
both final results. Existing corrected producer and interpreter parity gates
remain required.

The real run must reconcile eight scenes, selected/matched count `159405`, zero
failed workers, `ACC-JOINT` parity with the recomputed legacy `ACCEL-WORLD`, and
published source/operation/result hashes. Its recomputed legacy `result.json`
must be byte-identical to the accepted pre-extension file with SHA-256
`f7c36b7b7d038002a5b1fd5accbb566b3a9208afa7803139768312c69e8c2c36`.

## Migration Plan

1. Review this Mathematical Architecture and Code Architecture before source
   changes.
2. Implement and smoke the stable evaluator without changing legacy schema-v1.
3. Update the GroupRec worker/operation envelope additively and smoke it.
4. Run the eight-scene GroupRec evaluation into a fresh no-clobber output root.
5. Reconcile support, legacy metrics, new dynamics, receipts, and documentation.

Rollback removes only the additive module, GroupRec v2 envelope support, and
new artifacts. Existing corrected results are never rewritten.

## Modification History

- 2026-08-19: Created after selecting four world-space GT-relative dynamics
  metrics. Froze additive compatibility, formulas, support, units, exact versus
  related reference status, and the GroupRec re-evaluation target.
- 2026-08-19: Mathematical Architecture review found no Critical and four
  Concerns. Accepted all four: froze the exact selected-mask preconditions,
  combined-path bit-exact legacy parity, raw summary units/count invariants,
  and lexical `math.fsum` cross-scene reduction.
- 2026-08-19: Mathematical Architecture focused re-review reported zero
  remaining findings. During Code Architecture review, tightened direct reuse
  of the existing acceleration primitive, maximal contiguous segment traversal,
  a validate-once combined evaluator for GroupRec, and the operation-manifest
  v2 output-contract identity.
- 2026-08-19: Code Architecture review found no Critical and three Concerns.
  The count/window gate, run-wise existing-leaf reuse, validate-once combined
  entry, and manifest v2 identity were accepted. Focused review then moved the
  legacy acceleration ownership under the existing protocol module and froze
  a TPA canonical-byte legacy-result golden plus the accepted real-result hash.
- 2026-08-19: Final focused Code Architecture re-review reported zero remaining
  findings. Implemented the stable dynamics schema/evaluator/JSON surface,
  generic jerk leaf, combined acceleration reuse, GroupRec v2
  dual-result operation, and exact regression gates.
- 2026-08-19: Completed the real eight-scene GroupRec re-evaluation with zero
  failed workers. Support was 159,405 selected/matched occurrences, 156,263
  triples, and 154,883 quadruples. `ACC-JOINT`, `ACC-ROOT`, `JERK-JOINT`, and
  `JERK-ROOT` were respectively 1248.409754, 1247.640137, 2185.012503, and
  2183.732903 in `mm/frame^2` / `mm/frame^3`. Legacy result SHA-256 remained
  `f7c36b7b7d038002a5b1fd5accbb566b3a9208afa7803139768312c69e8c2c36`.
- 2026-08-20: Wrap-up review aligned the architecture text with the implemented
  combined-path reuse and added exact count/window support validation on stable
  summary and result readback. Two focused Behavior re-reviews closed the
  remaining shared pair-population invariant and reported zero Criticals.
