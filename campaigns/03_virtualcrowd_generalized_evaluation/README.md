# Campaign 03 - VirtualCrowd Generalized Evaluation

## Status

- State: active
- Owner: `hjlib-evaluation`
- Crowd4D native-integration owner: `hj-tpa-crowd4d`
- Current focus: reconstruct and verify the supplied Crowd4D/DyCrowd evaluator's
  logic before defining any HJ-owned evaluation contract.
- Next authorized action: complete the active Author Evaluator Logic Analysis
  task's Mathematical Architecture from frozen evidence and read-only
  inspection of the supplied evaluator.

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
| [Author evaluator logic analysis](task_author_evaluator_logic_analysis/) | active | Reconstruct inputs, normalization, matching, metrics, reduction, and Crowd4D/DyCrowd path differences as a verified mathematical baseline |

Later task decomposition remains deliberately deferred. Campaign 01 and 02 are
evidence sources, not a predeclared implementation plan. Each later substantial
mathematical, code-architecture, or evaluation-workflow layer will receive its
own Layered Design residence and review only when it becomes current.

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
