# Author Evaluator Logic Analysis

## Requirements

Establish a verified description of the supplied Crowd4D/DyCrowd evaluator
before designing an HJ-owned generalized evaluation path. The description must
be precise enough that a later task can decide what behavior to reproduce,
reject, parameterize, or isolate without importing Crowd4D-private schemas into
the public `hjlib-evaluation` contract.

The analysis covers:

1. GT and prediction loading, schema normalization, units, coordinate spaces,
   frame selection, and identity lifecycle;
2. Crowd4D/DyCrowd branch selection and tracking/association behavior;
3. 2D projection, joint selection/mapping, alignment, and each per-frame or
   temporal metric;
4. invalid, missing, unmatched, and empty-case behavior;
5. per-person, per-frame, per-scene, and global reduction, including weights
   and denominators;
6. formatting/scaling between internal values, result tables, and paper rows;
7. cross-checks against the frozen Campaign 02 fresh results.

Third-party material remains read-only and outside Git. Cite source paths and
symbols without copying implementation. Label every conclusion as directly
observed, empirically verified, inferred, or unresolved. Official upstream
provenance remains unverified and must not be implied by behavioral analysis.

## Mathematical Architecture

This layer is active and not yet accepted. Its analysis will be organized as:

1. native data model and normalization;
2. frame/person eligibility and correspondence;
3. spatial and temporal metric definitions;
4. reduction graph from atomic observations to published cells;
5. Crowd4D versus DyCrowd protocol delta;
6. edge/failure semantics;
7. frozen-result cross-check and unresolved questions.

The completed layer must state equations, shapes, units, index domains,
denominators, and empty-set behavior. A call graph alone is insufficient.

## Code Architecture

Deferred. This task does not define code residence, public APIs, adapters, or
implementation decomposition. Those decisions require the accepted
Mathematical Architecture and a separately activated task.

## Smoke-Test Standard

No reusable test facility is authorized in this analysis task. Internal
cross-checks may use disposable read-only probes, but durable claims must be
checked against the frozen Campaign 02 eight-scene results and record exact
evidence identity. Any later implementation test standard belongs to its own
task.

## Modification History

- 2026-08-11: Created when the user activated Campaign 03 and selected analysis
  of the author evaluator's logic as the first task. Requirements and analysis
  boundary were fixed; Mathematical Architecture remains active.
