# Campaign 03 - VirtualCrowd Generalized Evaluation

## Status

- State: active
- Owner: `hjlib-evaluation`
- Crowd4D native-integration owner: `hj-tpa-crowd4d`
- Current focus: awaiting user review of completed author-parity evidence.
- Next authorized action: none. The reviewed-protocol task remains Draft and
  requires explicit user activation.

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
| [Reviewed protocol and corrected results](task_reviewed_protocol_and_corrected_results/) | draft | After user review, replace selected legacy semantics and freeze a separate corrected result set without overwriting author parity |

Campaign 01 and 02 are evidence sources, not implementation dependencies. The
completed T2 retains its accepted Layered Design residence. The draft task is a
durable placeholder only: it has no implementation authority and will not
receive a design residence until the user activates it.

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
