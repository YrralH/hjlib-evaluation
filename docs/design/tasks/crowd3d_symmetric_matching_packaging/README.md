# Crowd3D Symmetric Matching Packaging

## Requirements

The evaluation owner exposes one method-neutral public dispatcher,
`match_coco17_people(...)`, two named profiles and one immutable
`Person_OKS_Association` result. Each profile is implemented as a sibling pure
operation with the same input and output types. The dispatcher, profile names
and result schema do not change.

`standard-matching-v1` retains its accepted visibility-aware, thresholded,
cardinality-first Hungarian semantics. `crowd4d-author-greedy-v1` retains the
accepted all-joint OKS, strict threshold, first-best selection and collision
drop, and additionally preserves the supplied evaluator's returned pair order.
Neither profile includes temporal tracking repair or metric reduction.

## Mathematical Architecture

Let `S[g,p]` be the profile-specific OKS matrix and let `V[p]` mark a prediction
whose 17 camera-space joints project at positive depth.

For `standard-matching-v1`, admissible edges satisfy
`S[g,p] >= 0.5 and V[p]`. One assignment maximizes accepted cardinality, then
the sum of OKS quantized at `1e12`. The resulting pair arrays are emitted in
ascending GT-row order. This is the existing canonical representation and is
unchanged. GT rows with no visible COCO17 joint have no definable OKS edge:
they are excluded from the assignment matrix, restored as FN in the original
GT row domain, and remain in the GT population denominator. Removing them
before the solve preserves the fixed supported-row tie behavior.

For `crowd4d-author-greedy-v1`, all 17 GT joints contribute and bbox area is
`max((x2-x1)*(y2-y1), spacing(1))`. For each GT row, compute its first minimum
distance `d[g] = min_p(1-S[g,p])`; process GT rows in the exact
`np.argsort(d)` order used by the supplied evaluator. If
`d[g] < 1-1e-6`, select the first prediction attaining that minimum. Accept it
only if unused; a collision consumes the attempt and does not try a second
prediction. Projection-invalid predictions are unavailable and remain FP.
The accepted pair arrays retain process order. FN and FP are the ascending
complements, so the association remains a complete disjoint partition.

The supplied evaluator receives already projected 2D predictions and has no
depth-validity guard. The HJ API receives camera-space joints and retains its
existing requirement that all 17 joints have positive depth. Therefore author
fidelity is claimed only on the projection-valid domain; an invalid prediction
is deliberately unavailable and remains FP rather than reproducing undefined
2D projection behavior.

Equal `d[g]` values inherit the supplied runtime's default `np.argsort`
behavior without selecting a stable `kind`. Their relative order is not
promised across NumPy versions or runtimes. The exact-compatibility claim is
same inputs plus the recorded supplied runtime, not a newly invented
cross-runtime tie rule.

Pair order does not change association cardinality or metric pairing because
the GT and prediction arrays move together. It does define serialized pair
order and pair-color order in visualization, so it is explicit contract rather
than incidental implementation detail.

## Code Architecture

Residence remains
`src/hjlib_evaluation/person_oks_association.py`. No new module, class,
registry, cache or dependency is introduced.

Both profile functions keep the same full input signature and return
`Person_OKS_Association`. A single internal association constructor accepts
already ordered matched GT/prediction arrays, derives canonical FN/FP
complements and validates them through the dataclass. The standard path turns
its fixed assignment mapping into ascending-GT pairs before calling that
constructor. The author path appends accepted pairs in release process order
and calls the same constructor directly. Empty inputs use the same constructor.

The public `match_coco17_people(...)` dispatcher remains the only public
matching dispatcher; profile-specific functions and construction primitives
remain module-local implementation surfaces. Existing public projection API is
unchanged. `hjlib-experiments-results` continues to select exact guide GT rows
and delegate once to the dispatcher. Its result serializer and renderer
preserve the association arrays they receive.

Public result containers reject rather than coerce non-integer identity/index
arrays or non-boolean validity masks. `Crowd3D_Matching_Frame` also closes the
normal factory invariants at its constructor boundary: GT track IDs equal the
prediction's ordered guide IDs, author matching requires a
`vc.crowd4d-style` source, standard pairs are in ascending GT order, and stored
projected COCO17 equals projection from the frame's native prediction joints
and camera. The association's projection-valid mask must equal the validity
returned by that same projection. Serializer and renderer may then treat any
constructed frame as truth without depending on a private factory history.

## Smoke-Test Standard

- Keep every existing standard threshold, objective, tie and partition test.
- Add a synthetic case whose author process order differs from GT-row order;
  assert the exact returned release order and aligned matched OKS.
- Add a partial standard case with noncontiguous matched GT rows and assert
  ascending GT order plus aligned prediction and OKS arrays.
- Keep collision/no-second-choice and strict-threshold cases.
- Add non-mocked author OKS regressions for all-17-joint support and degenerate
  bbox area clamping, plus explicit first-column and projection-invalid cases.
- Assert FN/FP complements remain complete when matched pairs are not GT-sorted.
- In Experiments Results, construct a non-GT-sorted association and assert that
  JSON serialization preserves its pair arrays and rendering assigns the first
  pair color to the first stored pair rather than silently sorting it.
- Reject fractional identity/index inputs before integer normalization, and
  reject manually constructed matching frames whose source/profile, guide IDs
  or projected COCO17 contradict their prediction source.
- Coincident matched roots still emit one zero-length connector draw call, so
  connector invocation remains equivalent to matched-pair membership.
- Run the full evaluation and Experiments Results smoke suites plus all strict
  pyright configurations and coverage checks.

## Migration Plan

1. Review the Mathematical Architecture and Code Architecture independently.
2. Replace mapping-only association construction with ordered-pair
   construction and route both profiles through it.
3. Add the discriminating regression before accepting the implementation.
4. Update stable design/usage documentation and the Crowd3D consumer usage.
5. Run focused/full gates and a bounded cross-repository closure review.

No artifact migration or result rewrite is required. Existing serialized
author-profile outputs, if any, must be regenerated rather than silently
reordered; Campaign 05 has not yet materialized those cells.

## Modification History

- 2026-09-12: task opened after the user accepted release-ordered Crowd4D Match
  visualizations. Requirements, mathematics and code architecture frozen for
  independent review before implementation.
- 2026-09-12: Mathematical Architecture review found no Critical and three
  Concerns; accepted by documenting the projection-valid fidelity domain and
  runtime-local equal-minimum ordering, plus a standard partial-order test.
  Code Architecture review found no Critical and one test-coverage Concern;
  accepted by adding a downstream serializer/renderer order regression. Its
  wording Note also corrected “public callable” to “public matching
  dispatcher”.
- 2026-09-12: bounded Mathematical Architecture and Code Architecture
  re-reviews found no residual Critical or Concern; implementation authorized.
- 2026-09-12: implemented one ordered-pair association constructor used by both
  profiles. The new author-order regression failed against the old GT-sorted
  representation and passes after the change; standard partial-order and
  Experiments Results serializer/renderer preservation regressions also pass.
- 2026-09-12: closure reviews found no Critical. Accepted Concerns add strict
  public-container integer and frame-consistency validation, non-mocked author
  OKS boundary tests, complete serializer/renderer pair-alignment assertions,
  and a zero-length connector call for coincident matched roots. A stale
  evaluation dependency pin remains the already recorded publish/pin
  prerequisite outside this no-commit task.
- 2026-09-12: bounded Code Architecture re-review found no Critical and one
  closure Concern: the frame must validate both projected COCO17 and the
  projection-valid mask derived by the same projection. Accepted without a new
  abstraction; the additional constructor work is `O(P * 17)`.
- 2026-09-12: implemented the accepted public integer and frame-closure
  validation and its negative regressions. Added direct, non-mocked author OKS
  tests for all-joint support and degenerate-area clamping.
- 2026-09-12: closure re-review found that the zero-displacement degenerate
  bbox test could not distinguish the exact epsilon from another positive
  fallback. Replaced it with a sigma-scaled displacement whose expected OKS is
  `exp(-1)` only under the release `spacing(1)` clamp.
- 2026-09-12: whole-behavior re-review found the same pre-cast validation gap
  for projection-valid masks and a same-source sigma oracle. Added strict bool
  rejection and froze an independent release sigma vector in smoke tests.
- 2026-09-12: whole-consistency re-review found scalar frame-ID and integer
  narrowing loopholes. Added strict Python-int frame identity and target-dtype
  range validation before immutable normalization.
- 2026-09-12: bounded re-reviews of every accepted closure finding are clean.
  Final smoke reports Evaluation `147 passed` and Experiments Results
  `408 passed, 1 skipped`; both strict pyright configs report zero diagnostics.
