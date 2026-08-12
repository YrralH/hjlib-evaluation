# Task - HJ-Composed Author-Parity Reproduction

## Purpose And Boundary

Independently reproduce the supplied Crowd4D and DyCrowd eight-scene evaluation
workflows by composing HJ-owned code. Preserve the supplied evaluator's actual
behavior as an explicit author-parity profile without copying or importing its
implementation and without turning Crowd4D-native schemas into stable
`hjlib-evaluation` contracts.

This task reproduces evaluation only. It does not reproduce inference, declare
the supplied package official upstream source, change the supplied residence,
or decide which legacy protocol choices are scientifically preferable.

## Status

- State: active
- Current layer: Code Architecture review
- Next action type: Layered Design
- Next authorized action: complete Code Architecture review; implement only
  after that layer is accepted. Mathematical Architecture is accepted.
- Blocker: none. The preferred cross-repository dependency direction remains
  revisable if the concrete reuse or ownership analysis disproves it.

## Detailed Residence

- [HJ-Composed Author-Parity Reproduction](../../../docs/design/tasks/hj-composed-author-parity-reproduction/README.md)

## Completion Criteria

1. The frozen Crowd4D and DyCrowd fresh tables and their input identities are
   revalidated and bound as immutable comparison oracles.
2. Mathematical Architecture specifies every author-parity value, association,
   repair, reduction, formatting, and edge behavior needed by the two workflows.
3. Code Architecture assigns Crowd4D-private parsing and orchestration to
   `hj-tpa-crowd4d`, assigns only genuinely method-neutral primitives to
   `hjlib-evaluation` or an existing lower owner, and keeps the stable library
   independent of the TPA.
4. HJ-owned code executes both eight-scene workflows without importing or
   copying the supplied evaluator and without writing to the supplied residence.
5. All 216 displayed numeric cells are compared against the frozen fresh tables;
   any non-exact cell is classified with raw numeric evidence rather than hidden
   by a success label.
6. Tracked receipts bind code, inputs, runtime, outputs, table identities, and
   comparison results. Tests cover independent metric behavior and real-data
   parity in proportion to their portability.
7. The result remains explicitly named author parity. No reviewed/corrected
   protocol is activated or substituted without separate user approval.

## Meaningful Actions And Results

- 2026-08-12: User activated T2 and required Layered Design. Revalidated the
  Campaign 02 table identities: Crowd4D is exact in 108/108 fresh-versus-bundled
  cells; DyCrowd is exact in 107/108, with only a `0.0001` rounding difference
  at `scene4_view2 / WA-MPJPE`.
- 2026-08-12: User selected `hj-tpa-crowd4d -> hjlib-evaluation` as the initial
  dependency direction. It is a provisional design hypothesis: preserve it when
  the concrete owner/reuse analysis supports it, and record any necessary pivot
  rather than forcing the original sketch.
- 2026-08-12: Initial Mathematical review found three Critical and four Concern
  fidelity gaps; all were corrected and focused re-review accepted every
  disposition with no new Critical. Boundary review's stale TPA entrance
  Critical and H36M second-truth Concern were likewise fixed and accepted.

## Handoff

Finish the task-specific Layered Design and its reviews. Do not activate the
draft reviewed-protocol task as a side effect of author-parity implementation.
