# GT-MOT Identity Baseline

## Requirements

Campaign 03 T3 first isolates one semantic delta from the accepted T2
author-parity baseline: replace the supplied evaluator's per-frame greedy OKS
association and later identity repair with a direct GT-MOT identity contract.
Compute a separately named result set for both Crowd4D and DyCrowd while
retaining the existing twelve author metrics and every downstream penalty,
temporal formula, reduction, and display rule.

The user confirmed that this task itself determines DyCrowd's missing identity
serialization from the available 2D trajectories; obtaining a producer mapping
is not a prerequisite. The supplied README documents an evaluation-time
`--use-gt-mot` temporal matching repair and explicitly says that it does not
modify the prediction `.pt`. The frozen artifacts therefore retain stable local
prediction columns but serialize `idxs` as compact zero-based columns rather
than VirtualCrowd's positive, sparse-capable `track_id`. A one-time reviewed
whole-trajectory recovery restores that serialization before evaluation.
Crowd4D's stored `idxs` already carry stable GT IDs for every active prediction
assignment.

Required outputs are:

1. one injective `prediction_column -> GT track_id` mapping per DyCrowd scene,
   with full trajectory evidence and ambiguity diagnostics;
2. a receipt-bound mapping sidecar whose prediction and GT input identities are
   frozen;
3. separate Crowd4D and DyCrowd eight-scene GT-MOT identity tables;
4. machine-readable raw and displayed deltas against the frozen T2
   author-parity results;
5. a complete operation receipt covering inputs, source, runtime, mapping,
   outputs, and the unchanged supplied-residence manifest.

T2 outputs and receipts are immutable comparison evidence. This task does not
rerun inference, import or copy the supplied evaluator, overwrite historical
tables, or activate any other proposed metric correction.

### Owner and dependency boundary

- `hjlib-evaluation` continues to own method-neutral metric leaves. No
  Crowd4D-native field or recovery policy enters its public contract.
- `hj-tpa-crowd4d` owns `idxs`, `track_flag`, `det_j2ds`, VirtualCrowd-native
  normalization, DyCrowd identity recovery, direct-ID profile orchestration,
  result comparison, and receipts.
- Dependency direction remains `hj-tpa-crowd4d -> hjlib-evaluation`.
- SciPy's linear-sum assignment may be consumed directly by the TPA if Code
  Architecture confirms a persistent generator needs it; no generic HJ
  assignment abstraction is created from this single method case.

## Mathematical Architecture

### Symbols and domains

For one scene, let:

- `T <= 200` be the same ordered evaluation frames as T2;
- `P` be native prediction columns active at least once in the selected `T`
  frames and `G` the sorted set of positive GT `track_id` values present
  anywhere in those frames; all-window-inactive native columns are recorded but
  do not receive an identity and cannot affect evaluation;
- `a[t,p]` be boolean `track_flag` activity;
- `d[t,p,k]` be DyCrowd's COCO-17 `det_j2ds` pixel coordinate and confidence;
- `q[t,p]` be boolean `det_j2ds_flag` evidence availability;
- `x[t,g,k]` be the GT COCO-17 pixel coordinate and visibility;
- `b[t,g]` be the GT pixel bbox and `z[t,g]` the GT-presence flag;
- `m[p] in G` be the recovered static identity mapping.

All identity recovery uses `float64` pixel coordinates. A recovery joint is
valid only when both coordinate pairs are finite, GT visibility is positive,
and DyCrowd detection confidence is positive. A prediction evidence frame
belongs to `E[p]` only when `a[t,p]` and `q[t,p]` are true and at least four
finite positive-confidence detection joints exist. Four is a conservative
task-local admissibility threshold chosen to prevent a single point or one
small limb fragment from identifying a full-person track; it is not a new
evaluation metric. A recovery frame for `(p,g)` additionally requires
`z[t,g]` and at least four jointly valid recovery joints.

### DyCrowd trajectory cost

For every eligible `(t,p,g)`, define the bbox diagonal

```text
s[t,g] = ||b[t,g,2:4] - b[t,g,0:2]||_2
```

and reject that frame for the pair when `s` is non-finite or not strictly
positive. For valid joint set `K[t,p,g]`, define the per-frame normalized cost

```text
c[t,p,g] = median over k in K[t,p,g] (
    ||d[t,p,k,0:2] - x[t,g,k,0:2]||_2 / s[t,g]
)
```

Let `F[p,g]` be those pair-eligible frames. An edge is admissible only when

```text
|E[p]| >= 3
|F[p,g]| >= max(3, floor(|E[p]| / 2) + 1)
```

Thus every compared candidate must explain a majority of the prediction
track's detection-evidence frames; a briefly overlapping unrelated identity
cannot win through a small favorable subset. A prediction column with fewer
than three evidence frames cannot be recovered by this operation.

The trajectory cost is the median of `c[t,p,g]` over `F[p,g]`. Record, but do
not mix into that geometric cost, `|E[p]|`, `|F[p,g]|`, active frames without
GT presence, invalid-evidence frames, per-frame nearest-GT vote counts, and the
best/second-best per-row costs. A non-admissible edge has infinite cost.

The majority rule makes candidate supports admissible on a common lower bound
without putting absence directly into the objective. The remaining separation
is deliberate: absence may mean that a GT-MOT track remains active while the
view-specific label omits that person. Treating every absence as geometric
error would silently redefine GT-MOT presence rather than recover identity.

### Global one-to-one recovery

Solve the rectangular linear-sum assignment over the `(P, |G|)` trajectory
cost matrix, assigning every prediction column to one distinct GT ID. Fail
before assignment when `P > |G|`, and fail after assignment when any selected
edge is non-finite. Exact equal-cost alternatives are not resolved by an
unstated tie rule: record the runtime/SciPy identity and calculate an alternate
global solution for each selected edge by forbidding it. If the alternate total
cost equals the selected total cost within

```text
1e-12 * max(1, abs(selected_total), abs(alternate_total))
```

the mapping is ambiguous and must not be frozen automatically.

For every selected edge, the draft sidecar records selected cost, prediction
evidence-frame count, pair-support count and fraction,
active-without-presence count, row rank, row second-best margin, per-frame
nearest vote fraction, and edge-forbidden global margin. These are diagnostics,
not extra optimization terms.

There is deliberately no automatic mapping acceptance. The generator produces
only a draft sidecar. A separate finding-focused identity-evidence review must
inspect structured diagnostics for every selected edge and inspect the
assigned-versus-runner-up trajectory overlays for every row-local collision,
globally displaced column, and lowest row/global margin or nearest-vote case.
The task reviewer may accept or reject the mapping; only an accepted review can
promote the complete injective mapping into the frozen evaluation sidecar.

The review judges the scene-level global one-to-one solution, not independent
row-local nearest neighbors. A row-local collision, negative row margin, or
globally displaced column is therefore an inspection trigger rather than an
automatic rejection: under the documented GT-MOT input assumption, two static
prediction identities cannot both own the same GT track. Acceptance requires
that every selected edge passes the hard admissibility gates, the complete
assignment is uniquely optimal under the stated tolerance, and inspected
conflicts have a coherent whole-track one-to-one explanation. One-frame,
fewer-than-four-joint, non-majority-support, non-finite, and exact-global-
ambiguity cases fail before review rather than being offered for acceptance. A
rejected or unresolved scene blocks publication. No per-frame match is written
into either sidecar.

Main-view `idxs + 1` and any future producer-supplied mapping are optional
validation oracles, not prerequisites and not separate scene-specific recovery
algorithms. If a future producer mapping with identified provenance is supplied,
it must be compared against the frozen recovered mapping and any disagreement
recorded as a new evidence finding; it does not retroactively change this
profile's declared 2D-recovered identity semantics.

### Direct identity frame pairs

Crowd4D uses its native stable GT ID. DyCrowd uses recovered `m[p]`. For either
method, frame `t` has exactly the pair set

```text
Q[t] = {(g, p) | a[t,p] and z[t,g] and identity(p) == g}
```

The mapping is injective, so `Q[t]` is one-to-one without geometric matching.
An active prediction whose mapped GT is absent contributes no metric pair in
that frame. It remains an active prediction for the existing redundancy term.
An unpaired present GT remains unmatched for the existing matched-ratio and
150-mm missing-pose penalty.

Before Crowd4D IDs are admitted, every active stored value must be a finite
positive integer, each active prediction column must carry exactly one constant
ID across the selected frames, every value must belong to the same-frame GT
domain, and distinct active columns must have distinct IDs. Any violation is a
hard failure. All-window-inactive columns are ignored as for DyCrowd.

### Frozen metric and reduction mathematics

The following are intentionally unchanged from T2:

- all twelve reported metric formulas: matched ratio, PPDS, PA-PPDS, PCOD,
  MPJPE, PA-MPJPE, WA-MPJPE, W-MPJPE, ACCEL, OKS, GMPJPE, and Score;
- all-joint/no-visibility reported OKS semantics;
- per-frame unmatched-GT pose penalties and redundancy calculation;
- pelvis-relative inputs, alignments, compact-time ACCEL, and track weighting;
- nominal-person temporal missing-slot denominator and 150-mm penalty;
- frame, scene, and global population reductions;
- four-decimal Python display rounding.

Only the source of matched pairs changes. Temporal samples use the same static
identity and include only `a[t,p] and z[t,m[p]]`; no author identity-repair
operation is called. The existing temporal formulas then consume those ordered
samples. Temporal coverage flags mark only the direct pair set `Q[t]`; all
other nominal-slot behavior remains T2-compatible.

Reported OKS still measures the paired predicted projection against GT using
the legacy all-17-joint rule. The visibility/confidence mask used for identity
recovery does not change that reported metric.

### Result comparison

The GT-MOT identity profile produces the same `9 x 12 = 108` displayed cells
per method as T2. Every cell records the raw GT-MOT value, rendered token, raw
T2 value, T2 token, signed delta, and absolute delta. A metric delta is expected
evidence of the identity semantic change and is never labeled an author-parity
failure.

### Failure semantics

The operation fails without publishing a success receipt when:

- a required artifact/schema/axis is absent or inconsistent;
- frame identity/order differs from the frozen T2 coverage;
- a Crowd4D active `idxs` value is not a same-frame GT ID;
- a Crowd4D active `idxs` value is non-integral/non-positive, changes within a
  prediction column, or duplicates another active column's identity;
- DyCrowd has more prediction columns than scene GT identities;
- a DyCrowd prediction column has fewer than three evidence frames, or no GT
  edge explains a majority of its evidence frames with at least four jointly
  valid joints per supporting frame;
- an assigned DyCrowd edge lacks finite trajectory evidence;
- the recovered mapping is non-injective or has an unresolved exact global
  ambiguity;
- the identity-evidence review rejects or cannot resolve any selected edge;
- a supplied validation oracle disagrees with recovered evidence without an
  explicit follow-up finding;
- any input/supplied identity changes during the operation;
- either method lacks eight scene rows, a mean row, or all 108 numeric cells.

## Code Architecture

### Repository and dependency boundary

No production code changes in `hjlib-evaluation`. Its campaign and this design
residence own the cross-method protocol decision; `hj-tpa-crowd4d` implements
the two Crowd4D-native profiles and continues to consume the existing stable
metric leaves. The existing dependency direction is already the highest layer
in the relevant component:

```text
hj-tpa-crowd4d (L7) -> hjlib-evaluation (L6) and lower owners
```

The TPA adds SciPy as a direct external dependency solely for rectangular
linear-sum assignment. No generic assignment API, matching registry, common
TPA base class, or Crowd4D schema enters a stable HJ library.

### Identity and recovery modules

`src/hj_tpa_crowd4d/gt_mot_identity.py` owns the typed, NumPy-only direct-ID
contract:

- define frozen `GT_MOT_Identity_Map(prediction_columns, gt_track_ids)` arrays
  with equal one-dimensional lengths, strictly increasing unique non-negative
  native prediction columns, and positive unique integral GT IDs; inactive
  native columns are omitted rather than encoded with a sentinel or
  renumbered;
- derive and validate Crowd4D's mapping from `idxs`, `track_flag`, and the
  same-frame GT-ID domains;
- select matched GT-row and active-prediction-row indices from one normalized
  frame by native-column lookup without geometric association, failing if any
  active native column lacks a mapping;
- keep absent mapped GTs out of the pair set without suppressing the active
  prediction count.

`src/hj_tpa_crowd4d/dycrowd_identity_recovery.py` owns the Mathematical
Architecture's one-time recovery. Stateless functions validate native
`det_j2ds`/`det_j2ds_flag` shapes, construct `E`, `F`, normalized frame costs,
the trajectory cost matrix, the SciPy assignment, edge-forbidden global
margins, row margins, and nearest-GT votes. The returned frozen dataclasses
contain JSON-safe scalar/list evidence; they do not read or write files and do
not decide acceptance.

`src/hj_tpa_crowd4d/identity_evidence.py` owns the three-residence sidecar
boundary:

1. a draft writer binds prediction/label hashes, ordered frame identities,
   algorithm constants, SciPy/runtime identity, all candidate diagnostics, the
   selected mapping, and deterministic per-edge SVG trajectory evidence;
2. a separately supplied review-decision document names the reviewer and role,
   review time, overall rationale/verdict, and one rationale/verdict plus
   inspected SVG identities for every selected edge;
3. `finalize-review` validates that document against an already promoted draft
   root, then atomically creates a separate immutable review root. It records
   the decision-document identity and draft receipt/sidecar/SVG identities;
   rejected reviews are retained but cannot authorize evaluation;
4. a review loader requires the draft SHA-256, an overall `accepted` verdict,
   and one explicit accepted verdict for every selected edge, rejecting
   missing/extra native columns, GT IDs, SVGs, or any digest mismatch. Only
   this validated object exposes `GT_MOT_Identity_Map` to evaluation.

Draft, review, and evaluation roots are three distinct new paths. A review root
references but never changes or copies its draft root; the evaluation root
references but never changes either evidence root. All three are disjoint from
the supplied residence, dataset, predictions, models, T2 receipt, and each
other. A reviewer may be the user or an explicitly identified task reviewer,
but anonymous acceptance and an empty overall/edge rationale are invalid.

The SVGs use task-local direct SVG serialization from recorded pixel points;
they do not introduce an image/plotting dependency or a reusable renderer.
For every selected edge they show the assigned GT and row runner-up at the
support-frame cost ranks `0`, `floor((n-1)/4)`, `floor((n-1)/2)`,
`floor(3*(n-1)/4)`, and `n-1`, deduplicated in that order. Frames are ordered by
`(assigned_edge_cost, frame_index)`; at least the lowest, median, and highest
support costs therefore appear whenever three support frames exist. A second
panel lists all prediction-evidence frames excluded by GT absence, invalid
joints/bbox, or lack of pair support. GT/pred joints, bboxes, frame index,
valid-joint count, both normalized costs, support flags, and row/global margins
are labeled. Missing runner-up evidence is rendered explicitly, never omitted.
The SVGs are evidence only—not inputs to recovery or evaluation.

### Evaluator reuse boundary

The twelve metrics are not copied. `Author_Parity_Evaluator` remains the single
scene loop and gains two safe overridable methods:

```python
def select_frame_pairs(
    self,
    gt: Author_Frame_GT,
    pred: Author_Frame_Prediction,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

def resolve_temporal_assignments(
    self,
    assignments: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

Their default implementations remain the existing greedy OKS association and
profile-selected author repair. `evaluate_scene_population()` calls those
methods, then calls the existing `compute_frame_metrics()` and temporal metric
reduction exactly as before.

`src/hj_tpa_crowd4d/gt_mot_evaluator.py` adds
`GT_MOT_Identity_Evaluator(Author_Parity_Evaluator)`. It accepts one validated
scene mapping, marks both overrides with `@override`, uses direct identity
pairs, and returns temporal assignments unchanged. Therefore the author repair
function is unreachable from this profile while all metric formulas and
reductions remain one implementation.

Tracking storage continues to index actual positive GT IDs. Its first axis is
allocated from the maximum selected-window GT ID rather than assuming dense
IDs. `compute_tracking_metrics()` receives the frozen legacy nominal-person
count explicitly, so sparse IDs change storage only; the existing missing-slot
denominator remains `legacy_count * frame_count` for both profiles. The T2
author path passes the same legacy count it previously encoded through array
shape, which is covered by the full parity regression.

### Operation and process boundary

The installed `hj-tpa-crowd4d` author-parity command remains byte-compatible at
its CLI surface. A second Typer entry point, `hj-tpa-crowd4d-gt-mot`, provides
one justified workflow namespace with three commands separated by the mandatory
review gate:

- `recover-dycrowd`: validate explicit dataset/prediction/supplied paths, run
  all eight recovery scenes, and atomically promote only the draft sidecars,
  SVGs, and recovery receipt;
- `finalize-review`: consume one explicit decision document plus the immutable
  draft root and atomically promote a separate accepted-or-rejected review root;
- `evaluate`: require the accepted review sidecars, T2 comparison receipt,
  existing model/prediction/dataset/supplied paths, and a new output root; run
  both eight-scene direct-ID profiles and atomically promote their results.

All three handlers remain thin and reuse `validate_disjoint_output()`, input
containment, prediction identity, supplied-residence manifest,
`output_transaction()`, result rendering, source/runtime identity, and atomic
writers from the T2 package. `src/hj_tpa_crowd4d/gt_mot_scene_worker.py` is a
separate short-lived Typer worker so the frozen author worker has no new mode
switch. Each worker writes one scene population plus mapping/validation digest;
the parent retains the one-scene-at-a-time memory bound.

The GT-MOT operation never treats T2 as an equality oracle; it uses the
strictly validated frozen baseline described below only to calculate deltas.

The final receipt has a distinct `gt_mot_identity_v1` schema and binds source,
SciPy/runtime, dataset labels, prediction/model inputs, accepted mapping review
and draft identities, per-scene mapping digests, T2 receipt identity, both
tables/delta files, and before/after supplied-residence manifests. Recovery and
evaluation output roots must be new and disjoint from every protected input;
partial work remains staging-only and is removed on failure.

Before spawning a DyCrowd scene worker, the parent validates each review
against its draft and the current scene inputs. The worker receives explicit
draft-sidecar and review-sidecar paths—not a free-standing mapping—and repeats
the full chain: review digest to draft digest; draft mapping and frame coverage
to the review edge set; draft prediction/label identities to the current files;
and current decoded ordered frame identities to the draft. Only the returned
validated mapping reaches `GT_MOT_Identity_Evaluator`. Thus parent and worker
cannot select different or unreviewed mappings through an alternate CLI field.

### Frozen T2 comparison contract

`src/hj_tpa_crowd4d/gt_mot_comparison.py` validates and loads the tracked T2
receipt before any scene worker starts. It requires the exact
`hj_tpa_crowd4d_author_parity_v1` schema and frozen receipt SHA-256
`6c7a8f44d7efe3398a870ceb33e25846dda09d67e2f40e4708ac678df2774565`, then
verifies against current inputs:

- all eight dataset-label file identities in canonical scene order;
- both profiles' eight prediction identities in canonical scene order;
- all three body-model asset identities;
- both serialized profile records, including names, prediction filenames, and
  temporal modes;
- current constants must equal the frozen literals: the eight canonical scene
  names/order, `EVALUATION_MAX_FRAMES == 200`, and the twelve named metrics;
  for each current scene, coverage is re-derived as
  `min(200, prediction_track_flag_T, decoded_label_T)` from the same
  identity-checked inputs, then required to match the GT-MOT operation's
  decoded ordered frame identities;
- exactly two named profiles, each with exactly nine scene rows and nine raw
  rows in canonical scene order, exactly the twelve finite numeric metric keys
  per row, and complete 108-cell evidence in canonical scene/metric sequence.

Any mismatch is a hard failure; a same-schema receipt from another input set,
profile, scene order, or frame window is not a baseline. The frozen SHA-256 is
recorded in the new receipt after those semantic checks.

`result_table.py` adds:

```python
def build_result_deltas(
    gt_mot_rows: list[dict[str, float | str]],
    gt_mot_raw_rows: list[dict[str, float | str]],
    t2_rows: list[dict[str, float | str]],
    t2_raw_rows: list[dict[str, float | str]],
) -> list[dict[str, str]]: ...
```

All four grids must have canonical scene list order, nine rows, the exact
twelve-key metric set with finite numeric values per row, and raw/display row
agreement. JSON object insertion order is not a metric contract; delta output
iterates explicit `AUTHOR_METRICS` order. The T2 receipt's `cells` list is
separately validated as the canonical 108-item scene/metric sequence. JSON is
loaded with duplicate object keys rejected before conversion to dictionaries.
For every cell the builder records both raw values and four-decimal tokens, then
`signed_delta = gt_mot_raw - t2_raw` and its absolute value. Missing, extra,
or reordered scene rows, and missing/extra/non-canonical cell entries, fail
rather than being zipped away.

## Smoke-Test Standard

Portable smoke tests are split by the nouns above and wired into the existing
flat master runner:

1. direct pairing covers sparse positive GT IDs, view-specific absence,
   active-prediction redundancy preservation, inactive columns, and every
   Crowd4D constancy/integrality/injectivity/same-frame failure; an inactive
   native column between two active columns proves mappings retain native
   indices without a sentinel or compaction;
2. recovery covers four-joint and three-frame boundaries, strict-majority
   support for odd/even evidence counts, competing/disjoint supports,
   rectangular global assignment, non-finite evidence, exact global ambiguity,
   and producer-map disagreement;
3. evidence tests cover the fixed SVG rank sample, assigned/runner-up and
   excluded-frame panels, deterministic draft digest, distinct immutable
   draft/review roots, and rejection of anonymous, rationale-empty,
   unaccepted, incomplete, extra-edge, wrong-SVG, or wrong-digest reviews;
4. evaluator tests prove direct pairs bypass greedy association and repair,
   while the unchanged author hooks retain the existing collision and repair
   behavior; sparse-ID storage must leave the legacy nominal denominator
   unchanged;
5. T2 binding rejects wrong label/prediction/model identity, wrong profile,
   non-frozen constants, wrong re-derived coverage, missing/extra/reordered
   scenes/cells, duplicate JSON keys, and non-finite values; row dict key order
   is deliberately ignored, while delta tests fix the `GT-MOT - T2` sign and
   keep raw values separate from rendered tokens;
6. CLI top-level and all three subcommand `--help` calls succeed, and failed
   recovery/review/evaluation transactions leave no final root.

Strict Pyright runs over `src`, `test_smoke`, and `test`. Because production T2
modules are touched to expose the two hooks and explicit nominal count, the
existing full eight-scene author-parity operation must be rerun and reproduce
its accepted `108/108` Crowd4D plus classified `107/108` DyCrowd outcome before
the GT-MOT result run is admissible. The GT-MOT run itself occurs only after
all eight DyCrowd draft mappings receive accepted review sidecars.

## Migration Plan

No migration or overwrite is allowed. The new mapping sidecars and result
profile receive new names and residences; author-parity code paths and evidence
remain callable and immutable.

Implementation also lands the public surface in `hj-tpa-crowd4d`:

- add `docs/usage/gt-mot-identity.md` with the three-stage commands, review
  decision schema, picking table, and evidence interpretation;
- update `docs/usage/README.md` decision tree/Common API and remove its Draft
  statement;
- update `README.md` and `docs/design/README.md` status, repo layout, must-read,
  current verification/open items, and link this cross-owner residence;
- update `docs/design/test.md` with the new smoke and explicit-path real-run
  gates;
- create a TPA-side Campaign 03 evidence mirror for accepted mapping reviews,
  tables, deltas, and receipts while `hjlib-evaluation` retains live status.

## Modification History

- 2026-08-13: User activated T3 and selected the GT-MOT identity baseline as
  its first bounded work package. Created Requirements and Mathematical
  Architecture before implementation; later metric corrections remain
  deferred.
- 2026-08-13: Mathematical Architecture review found one Critical weak-evidence
  acceptance gap, candidate-specific support comparability, missing Crowd4D
  constancy/injectivity invariants, and inactive-column ambiguity. Added
  four-joint/three-frame/majority-support admissibility, removed all automatic
  mapping acceptance in favor of a separate evidence review, formalized
  Crowd4D identity invariants, and limited `P` to active columns. Focused
  re-review found only that `ceil(|E|/2)` was not a strict majority for even
  evidence counts; replaced it with `floor(|E|/2)+1`. Final focused re-review
  accepted the Mathematical Architecture with no remaining findings.
- 2026-08-13: Added Code Architecture and Smoke-Test Standard after the accepted
  Mathematical Architecture. The design keeps all native recovery and
  orchestration in `hj-tpa-crowd4d`, reuses a single metric/reduction
  implementation through two evaluator hooks, and makes DyCrowd review a hard
  process boundary before evaluation. Initial Code Architecture review found
  three Critical gaps in sparse-native-column mapping, review-to-worker
  artifact flow, and T2 input binding, plus SVG, delta-grid, CLI-help, and docs
  concerns. Replaced the dense mapping with an explicit native-column record;
  froze distinct draft/review/evaluation roots and double-validated worker
  handoff; fully bound T2 inputs/profile/coverage; and specified the remaining
  evidence, test, CLI, and docs contracts. Focused re-review found that the
  frozen T2 receipt has no stored frame-population lengths and that JSON row
  dict order is alphabetical, not metric order. Replaced the unavailable field
  check with frozen-literal/current-input coverage derivation, moved canonical
  metric-order validation to the T2 cell list, and added duplicate-key-safe
  JSON loading. Final focused re-review accepted the Code Architecture and
  Smoke-Test Standard with no remaining findings.
- 2026-08-13: Implemented the accepted architecture in `hj-tpa-crowd4d` and
  passed the portable smoke/Pyright gates. Generated eight external DyCrowd
  draft sidecars without running evaluation. Evidence review found seven
  row-local GT collisions across six scenes and eight columns displaced by the
  global injective assignment; conflicting pairs remain simultaneously active
  for 57--200 frames. Representative overlays show displaced predictions much
  closer to the runner-up GT than their selected edge. Main-view `idxs+1`
  validates scenes where IDs remain continuous, but sparse-ID artifacts contain
  no producer mapping to resolve the other cases. The drafts are not accepted;
  evaluation is blocked pending a producer mapping or exact GT-MOT input
  artifact.
- 2026-08-13: Rechecked the supplied package README after the author's verbal
  clarification. Its documented `--use-gt-mot` flag is evaluation-time
  temporal matching repair and does not modify prediction `.pt` files; it is
  not evidence that prediction columns already encode GT `track_id`.
- 2026-08-13: User selected reviewed whole-track 2D recovery itself as the
  authority for `prediction column -> GT track_id` and authorized unattended
  completion. Producer mapping evidence is no longer a prerequisite. Clarified
  that row-local collisions and global displacement trigger focused inspection
  but are judged under the scene-level one-to-one assignment rather than
  rejected independently.
- 2026-08-13: Focused Mathematical Architecture re-review of that authority
  change accepted the layer with no Critical or Concern findings. The reviewer
  confirmed that hard admissibility, unique global assignment, focused conflict
  inspection, and explicit 2D-recovered semantics form a closed contract.
- 2026-08-13: Implementation audit found that the accepted-review loader did
  not independently revalidate per-edge verdicts or current SVG bytes. Brought
  implementation back to the already accepted Code Architecture by validating
  contained SVG identities and the complete edge/mapping/verdict chain both at
  finalization and before evaluation; strict Pyright and the focused tamper
  smoke pass.
- 2026-08-13: Completed the real-data gates. The accepted review covers all 909
  selected edges and focused visual checks of eight globally displaced edges
  plus their seven row-local owners. T2 regression summaries are byte-identical
  to accepted evidence. Both eight-scene GT-MOT profiles produced complete
  `9 x 12` tables and deltas with an unchanged supplied-residence manifest.
  A receipt path audit found and fixed staging-only output locations; the final
  rerun is byte-identical in all six result/summary/delta files and records
  stable root-relative output paths. Strict Pyright and 19 smoke tests pass.
