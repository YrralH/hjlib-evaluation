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

- State: complete
- Current layer: none
- Next action type: none
- Next authorized action: none; the downstream corrected-protocol task consumed
  this handoff and is complete.
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
- 2026-08-12: Implemented the stable OKS/joint-error leaves, raw H36M owner
  forward, and Crowd4D-native author profile package. The formal isolated
  16-scene operation compared all 216 cells with an unchanged supplied manifest:
  Crowd4D is exact in 108/108 and DyCrowd is exact in 107/108.
- 2026-08-12: Classified the sole DyCrowd difference at `scene2 / matched ratio`:
  the frozen token is `0.9857`, while HJ and a current read-only supplied-
  evaluator run have identical 200-frame populations and identical raw mean
  `0.9856499999999998`, which renders `0.9856` in the current runtime. This is
  recorded as frozen-token drift, not parity success and not an HJ association
  mismatch.
- 2026-08-12: The authorized third focused implementation review accepted the
  final raw-evidence disposition with no new Critical. The tracked comparison
  contains raw HJ values, rendered and oracle tokens, absolute deltas, and exact
  verdicts for all 216 cells. T2 is complete.

## Handoff

Evidence is owned at
[`hj-tpa-crowd4d` Campaign 03 mirror](../../../../hj-tpa-crowd4d/campaigns/03_hj_composed_author_parity/).
Return to the campaign for user review. Do not activate the draft reviewed-
protocol task without explicit user approval.
