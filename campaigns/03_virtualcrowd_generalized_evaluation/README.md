# Campaign 03 - VirtualCrowd Generalized Evaluation

## Status

- State: active
- Owner: `hjlib-evaluation`
- Crowd4D native-integration owner: `hj-tpa-crowd4d`
- Current focus: implement the TPA-owned corrected adapter/profile against the
  accepted stable metric schema.
- Next authorized action: add portable adapter/common-manifest/transaction
  gates. Do not run corrected evaluation until those gates pass.

## Goal And Boundary

Create an HJ-owned independent evaluation capability whose stable contracts,
loading boundary, metric semantics, and reduction are reusable by Crowd4D and
the user's own methods. Validate it against frozen VirtualCrowd data identity
and author-evaluator evidence without making Crowd4D's native artifact layout a
public library contract.

This campaign does not reproduce Crowd4D inference, copy or import the author
evaluator, create a universal loader for arbitrary third-party files, or move
method-native parsing into `hjlib-evaluation`.

## Completion Criteria

1. A method-neutral GT/prediction contract has explicit identity, shape, unit,
   coordinate-frame, visibility, missing-data, and aggregation semantics.
2. Independent metric mathematics and evaluation behavior are documented,
   reviewed, implemented, and covered by synthetic and real-data tests.
3. VirtualCrowd GT and normalized predictions can enter the stable evaluation
   path without importing a third-party runtime.
4. Crowd4D native artifacts are mapped by a Crowd4D-owned adapter, and at least
   one non-Crowd4D/own-method-facing adapter smoke demonstrates that the stable
   contract is not Crowd4D-private.
5. Differences from the frozen author baseline are explained, and the public
   usage path is documented.

## Task Structure

| Task | State | Purpose |
| --- | --- | --- |
| [Author evaluator logic analysis](task_author_evaluator_logic_analysis/) | complete | Reconstructed inputs, normalization, matching, metrics, reduction, and Crowd4D/DyCrowd path differences as an accepted mathematical baseline |
| [HJ-composed author-parity reproduction](task_hj_composed_author_parity_reproduction/) | complete | Reproduced both author workflows with HJ-owned composition and classified the sole frozen-token drift |
| [Reviewed protocol and corrected results](task_reviewed_protocol_and_corrected_results/) | active: corrected adapter implementation | Mathematical and Code Architecture are accepted; generic geometry/evaluation facilities pass portable gates |
| [Native output and scene semantics audit](task_native_output_and_scene_semantics_audit/) | complete | Accepted both methods' scene/world support and classified inactive dense human buffers as invalid evaluation padding |

Campaign 01 and 02 are evidence sources, not implementation dependencies. The
completed T2 retains its accepted Layered Design residence. T3 was explicitly
activated on 2026-08-13 with one bounded first work package; later metric and
reduction corrections remain deferred and do not block that package.

## Inputs And Relationships

- Dataset evidence owner:
  [`hj-tpa-crowd4d` Campaign 01](../../../hj-tpa-crowd4d/campaigns/01_virtualcrowd_dataset_crosscheck/)
- Author evaluation evidence owner:
  [`hj-tpa-crowd4d` Campaign 02](../../../hj-tpa-crowd4d/campaigns/02_crowd4d_author_evaluation_audit/)
- Family TPA knowledge:
  [`hjlib-agent/tpa-knowledge`](../../../hjlib-agent/backend/memory/workspaces/Code_as_Libs/tpa-knowledge.md)

The two predecessor campaigns remain independently deliverable; this campaign
consumes frozen evidence rather than mirroring their live status.

## Headline Status Consumer

- [`../README.md`](../README.md)

## Modification History

- 2026-08-09: Recorded the user-selected separate campaign boundary and
  evidence-driven task policy.
- 2026-08-09: User clarified that adapter architecture means the concrete
  `hj-tpa-*` code/file residence pattern. Removed the mistaken data-flow
  outline; concrete Code Architecture remains deferred for user direction.
- 2026-08-11: User activated Campaign 03 and selected author-evaluator logic
  analysis as its first task. Created a task-specific Mathematical Architecture
  residence; no generalized API or implementation is authorized yet.
- 2026-08-11: Completed the first task after dedicated Mathematical
  Architecture review, correction of two Critical fidelity findings, and a
  focused re-review. Campaign 03 remains active without an authorized second
  task.
- 2026-08-12: User activated HJ-composed author-parity reproduction as T2 and
  required Layered Design. Recorded `hj-tpa-crowd4d -> hjlib-evaluation` as the
  preferred, revisable dependency direction: Crowd4D-specific orchestration
  stays in the TPA, while only method-neutral primitives may enter the stable
  library. Registered corrected-protocol work as draft T3 pending user review.
- 2026-08-12: Completed T2 after three explicitly authorized focused
  implementation reviews. All 216 cells have raw comparison evidence;
  Crowd4D is 108/108 exact and DyCrowd is 107/108 exact with one classified
  rounding-boundary frozen-token drift. Campaign 03 now awaits user review and
  does not implicitly activate Draft T3.
- 2026-08-13: Handed the campaign to the next session at the T3 planning
  boundary. The next session must begin with an attended situation check and
  must not infer activation from this handoff.
- 2026-08-13: User explicitly activated T3 and selected its first work package:
  recover DyCrowd's compact GT-MOT identities, use Crowd4D's recorded GT IDs,
  and compute a separately named two-method result set with the existing author
  metrics/reductions unchanged except for direct identity pairing. Other
  candidate protocol corrections remain deferred.
- 2026-08-13: Implemented and smoke-tested the accepted recovery/review/direct-
  identity architecture, then generated all eight DyCrowd mapping drafts
  without running evaluation. Evidence review did not accept the drafts: six
  scenes contain seven row-local identity collisions, eight columns are
  displaced by global injectivity, and the conflicting columns overlap for
  57--200 frames. The author's verbal GT-MOT-input confirmation does not supply
  the missing mapping: the package README documents `--use-gt-mot` only as an
  evaluation-time temporal matching repair that does not modify prediction
  `.pt` files. T3 remains active pending producer mapping evidence.
- 2026-08-13: User authorized this task to determine the missing DyCrowd
  `prediction column -> GT track_id` mapping from reviewed whole-track 2D
  trajectories inside the TPA, without waiting for producer evidence, and
  requested Layered Design plus Sequential Task Level 2 with unattended
  completion in the existing T3 boundary.
- 2026-08-15: Accepted corrected Mathematical and Code Architecture after
  dedicated reviews and focused closures. Implemented reusable registration in
  `hjlib-geometry` and the versioned two-view corrected schema, leaves,
  per-scene evaluator, reduction, serialization, and smoke gates in
  `hjlib-evaluation`. TPA adapter implementation is next; no corrected real-data
  evaluation has run yet.
- 2026-08-13: Completed the bounded T3 first work package. Reviewed and accepted
  all 909 DyCrowd mapping edges, including focused visual checks of eight
  globally displaced edges and their seven row-local owners; reran T2 with
  byte-identical summaries; and completed both eight-scene GT-MOT profiles.
  The supplied residence manifest remained unchanged. The campaign remains
  active only for user selection of any later, separately scoped protocol task.
- 2026-08-14: User activated a read-only T3 metric-semantics audit covering all
  twelve retained author metrics. The audit may recommend later adjustments,
  but it does not authorize formula changes or a corrected-evaluation run.
- 2026-08-14: Completed the twelve-column audit and accepted all five findings
  from dedicated Mathematical Architecture review; focused re-review found no
  remaining Critical or Concern. Candidate corrections remain provisional and
  require user selection before design, implementation, or rerun.
- 2026-08-14: User inserted a read-only native-output and scene-semantics task
  before corrected-protocol design. Whole-sequence association is treated as a
  designable adapter problem rather than the root blocker. The task must first
  classify inactive dense pose/translation slots and determine whether each
  Crowd4D/DyCrowd artifact contains ground/scene state, including whether that
  state is predicted, optimized, copied input, or GT. No corrected evaluation
  or implementation is authorized by this insertion.
- 2026-08-14: Completed the inserted audit. Crowd4D carries SIPC scene geometry
  plus ground/camera/transform state; DyCrowd carries an estimated ground plane
  and camera/alignment transforms. The user accepted both for corrected
  protocol use. Artifact and evaluator evidence classifies `track_flag=False`
  dense pose/translation as invalid padding; DyCrowd occlusion recovery instead
  appears as 11,290 valid outputs without 2D detection. The campaign returned
  to corrected-protocol requirements selection.
- 2026-08-15: Completed attended corrected-protocol requirements selection.
  Completeness is derived only from GT, prediction, and one-to-one association
  sets; generic validity checking is fail-fast and any observed invalid class
  must receive a specific reviewed label rather than a speculative universal
  taxonomy. Activated the task-specific Mathematical Architecture layer; no
  implementation or corrected evaluation run is yet authorized.
