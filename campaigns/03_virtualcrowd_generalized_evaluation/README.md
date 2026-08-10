# Campaign 03 - VirtualCrowd Generalized Evaluation

## Status

- State: draft
- Owner: `hjlib-evaluation`
- Crowd4D native-integration owner: `hj-tpa-crowd4d`
- Current focus: campaign boundary recorded; concrete code/file architecture
  awaits user direction and execution is not active.
- Next authorized action: after the relevant dataset identity and author
  baseline are frozen and the user activates this campaign, establish a
  task-specific Layered Design residence for the first evidence-based layer.

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

Task decomposition is deliberately deferred until activation. Campaign 01 and
02 are evidence sources, not a predeclared task plan for this campaign. Each
substantial mathematical, code-architecture, or evaluation-workflow layer will
receive its own Layered Design residence and review only when that layer becomes
current.

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
