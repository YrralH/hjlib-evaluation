# Task - Author Evaluator Logic Analysis

## Purpose And Boundary

Reconstruct the supplied Crowd4D and DyCrowd VirtualCrowd evaluator as an
evidence-backed mathematical and execution-logic baseline. Trace native inputs
through normalization, identity matching, per-frame metrics, temporal metrics,
and scene/global reduction, and identify every intentional divergence between
the Crowd4D and DyCrowd paths.

This task is analysis only. It does not modify or copy third-party material,
implement an HJ evaluator, define a public library contract, reproduce Crowd4D
inference, or treat the supplied package as verified official source.

## Status

- State: active
- Next action type: analysis
- Next authorized action: populate and review the Mathematical Architecture in
  the linked design residence using frozen Campaign 01/02 evidence and
  read-only inspection of the supplied evaluator.
- Blocker: none for analysis. Unverified official source identity remains a
  boundary on provenance claims, not a blocker to documenting the identified
  evaluation package's behavior.

## Detailed Residence

- [Author Evaluator Logic Analysis](../../../docs/design/tasks/author-evaluator-logic-analysis/README.md)

## Evidence Boundary

- Dataset byte identity: `hj-tpa-crowd4d` Campaign 01.
- Package/runtime identity and fresh author baselines: `hj-tpa-crowd4d`
  Campaign 02.
- Supplied evaluator: machine-local third-party material, inspected read-only;
  it is evidence, not tracked implementation or a public contract.
- Independent conclusions must cite concrete source symbols, artifacts, or
  frozen run evidence and distinguish observed behavior from inference.

## Completion Criteria

1. Native GT and prediction schemas, units, coordinate spaces, validity rules,
   and identity lifecycles are mapped.
2. Matching/association, per-frame metrics, temporal metrics, and every
   reduction/aggregation step are expressed precisely enough to reimplement.
3. Crowd4D and DyCrowd branches are compared explicitly, including the effect
   of the DyCrowd `--use-gt-mot` path.
4. Metric names, direction, scale, denominator, missing-data behavior, and
   scene/global weighting are cross-checked against frozen fresh tables.
5. Confirmed behavior, inferred intent, unresolved ambiguity, and behavior that
   should not become an HJ contract are clearly separated.
6. The Mathematical Architecture passes its dedicated review; no implementation
   task is activated implicitly.

## Meaningful Actions And Results

- 2026-08-11: User activated Campaign 03 and selected this analysis as its
  first task. The analysis boundary includes both mathematical semantics and
  execution flow but excludes generalized API and implementation design.

## Artifacts

- No new runtime artifact yet. Durable conclusions belong in the linked design
  residence; large or third-party artifacts remain at their existing owners.

## Handoff

After the Mathematical Architecture is accepted, return to the campaign for a
user-directed decision on the next task. Do not infer authorization to design
or implement the generalized evaluator.
