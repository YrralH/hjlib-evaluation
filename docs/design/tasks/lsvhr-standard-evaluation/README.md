# LSV-HR Standard Evaluation

## Requirements

The standard evaluation consumes one normalized `Corrected_Crowd_Sequence`
produced by an existing Standard Loader and parameter interpreter, one selected
GT mask, and protocol-owned base-VISRUN labels aligned to the sequence GT rows.
It owns no method-native decoding and does not change Loader behavior.

The public metric profile contains exactly these 14 values, in this order:

```text
MPJPE-WORLD
T-MPJPE
RT-MPJPE
SEQ-T-MPJPE-VISRUN
SEQ-RT-MPJPE-VISRUN
SEQ-T-MPJPE-TRACK
SEQ-RT-MPJPE-TRACK
RTE-WORLD
ACC-ROOT
ACC-ROOT-RATIO
ACC-JOINT-RATIO
OKS-VIS
PPDS
PA-PPDS
```

`ACC-JOINT`/legacy `ACCEL-WORLD`, `PA-MPJPE`, sequence PA-MPJPE, and PCOD are
not part of this profile. Existing profiles retain their existing contracts.

One exact direct-target join and one shared scope index are built per scene.
Metric implementations consume scopes and cannot reconstruct TRACK, VISRUN, or
FRAME partitions. Scope-index time is reported separately in the one-scene
profile; the acceptance target is at most 1% of index-plus-metric wall time.

The result-side CLI consumes the existing source catalog only for selection and
dataset/population inputs. For formal protocols it resolves the existing
machine-local Standard Loader binding and invokes the method-owned runtime. It
does not use the source's legacy adapter, native result path, or model path to
load predictions. Fast-test compatibility remains on the legacy path.

No SHA/hash, authentication, artifact attestation, or additional artifact
validation is added. The Loader is changed only if profiling identifies it as
the bottleneck.

## Mathematical Architecture

### Population and shared partitions

For one scene, the caller supplies `base_visrun_label[g]` for every normalized
GT row `g`: equal nonnegative labels mean the rows belong to the same
protocol-defined base run and `-1` means the row is outside that domain. For VC,
the run is the accepted population span, including internal unqualified frames
that `vc.visibility_continuity_v1` deliberately retains; for WP, it is the
protocol filtering range. Labels are constructed before applying evaluation
selection, so a selected-out occurrence does not erase run identity. The
evaluator rejects any selected row with label `-1`.

Let `M` be the exact one-to-one direct-target join between the selected GT
occurrences and predictions. Every selected GT row must have one prediction.
Each joined occurrence carries native frame `f`, GT identity `q`, its supplied
base-VISRUN label, SMPL-24 world joints `X` (prediction) and `Y` (GT), COCO-17
projections, native visibility, GT bbox, and paired prediction camera depth.

The shared index contains three partitions of `M`:

- `TRACK(q)`: every selected joined occurrence of identity `q`, ordered by
  native frame and retained across gaps;
- `VISRUN(q,k)`: one protocol-provided base run of identity `q`;
  selected/matched holes do not split this alignment scope;
- `FRAME(f)`: every selected joined occurrence at frame `f`, ordered by GT
  identity.

TRACK and VISRUN share one `(q,f)` order over joined rows while VISRUN boundaries
come from the independently supplied base labels rather than frame gaps.
Each VISRUN additionally carries precomputed exact-consecutive sub-boundaries
over its selected joined rows for acceleration only. Thus selection holes remain
inside one sequence-alignment VISRUN but split acceleration support. FRAME uses
one `(f,q)` order. The index stores occurrence orders, boundary offsets, and the
joined VISRUN labels; it does not copy joint tensors. Its public constructor
rederives canonical partitions from the join and labels, so a shape-valid FRAME
partition cannot be substituted for TRACK or vice versa.

The standard join is deliberately lightweight: it validates and stores only
GT/prediction row vectors against the already immutable normalized sequence. It
does not reuse the legacy defensive join constructor, whose fresh snapshot
would clone every metric tensor and make that copy appear as indexing cost.

### TRACK metrics

For every TRACK scope, flatten all SMPL-24 points. `SEQ-T-MPJPE-TRACK` fits one
least-squares translation to the flattened paired point sets.
`SEQ-RT-MPJPE-TRACK` fits one reflection-disabled rigid transform with no
scale. Both contribute every aligned point error to a global micro mean.

### VISRUN metrics

`SEQ-T-MPJPE-VISRUN` and `SEQ-RT-MPJPE-VISRUN` use the same objectives as the
TRACK variants but fit one transform per VISRUN.

Within each base VISRUN, acceleration consumes its precomputed maximal
exact-consecutive selected sub-runs. `ACC-ROOT` uses the second-difference
residual on pelvis joint zero:

```text
a_X(t) = X(t+1,0) - 2 X(t,0) + X(t-1,0)
a_Y(t) = Y(t+1,0) - 2 Y(t,0) + Y(t-1,0)
e(t) = ||a_X(t) - a_Y(t)||_2
```

It is the micro mean of all supported values and is reported in
`mm/frame^2`.

The two ratio metrics retain the established twice-central-difference operator:

```text
D(Z)[t] = 0.5 * (pad(Z)[t+2] - pad(Z)[t])
A(Z) = D(D(Z))
```

Three samples are trimmed from each end; VISRUNs of length at most six have no
ratio support. `ACC-ROOT-RATIO` globally sums the pelvis acceleration magnitudes
before division:

```text
sum ||A(X[:,0])||_2 / sum ||A(Y[:,0])||_2
```

For `ACC-JOINT-RATIO`, each SMPL joint has its own globally reduced ratio, then
the 24 ratios receive equal weight:

```text
r_j = sum ||A(X[:,j])||_2 / sum ||A(Y[:,j])||_2
ACC-JOINT-RATIO = mean_j r_j, j=0..23
```

Each scene summary carries the root predicted/reference scalar sums, root
support count, joint predicted/reference arrays of shape `(24,)`, and one joint
support count shared by all 24 joints. Denominator checks happen only after
cross-scene sums are reduced. Joint-zero predicted/reference sums and support
must be exactly equal to the root-ratio statistics. The per-joint sums are
accumulated across all scenes before division. Every globally reduced GT joint
denominator must be finite and strictly positive; a scene-local zero is allowed,
but no epsilon and no silent joint omission are permitted. Both ratios are
unitless with target value one. They measure acceleration-magnitude calibration
rather than vector residual accuracy.

### FRAME metrics

For every matched person-frame:

- `MPJPE-WORLD` is the mean absolute SMPL-24 world-position error;
- `T-MPJPE` subtracts each skeleton's own SMPL pelvis before error;
- `RT-MPJPE` fits one reflection-disabled, no-scale rigid transform;
- `RTE-WORLD` is `||X[0]-Y[0]||_2`;
- `OKS-VIS` uses COCO-17, GT bbox area, frozen COCO sigmas, native visibility
  greater than zero, and strictly positive paired prediction camera depth.

For every FRAME scope, `PPDS` evaluates every unordered pair of matched pelvis
anchors once. `PA-PPDS` fits one positive-scale reflection-disabled similarity
transform over all matched pelvis anchors in the frame and applies its one scale
to every pair distance before PPDS. Frames with fewer than two people contribute
no pair samples.

All position errors are reduced in metres and converted to millimetres after
global reduction. OKS and PPDS are unitless fractions. Mean metrics use one
global micro mean over their primitive samples; scopes are never macro-averaged.

## Code Architecture

`hjlib-evaluation` owns three new modules:

```text
standard_evaluation_scope.py
    Standard_Evaluation_Scope_Partition
    Standard_Evaluation_Scope_Index
    build_standard_evaluation_scope_index(
        sequence, selected_gt_mask, base_visrun_labels)
    iter_standard_evaluation_scope_rows(...)

standard_evaluation_metrics.py
    metric-specific pure contribution functions accepting only already sliced
    metric-native tensors, never a sequence, index, frame ID, or track ID

standard_evaluation.py
    immutable scene summary and final result contracts
    private heterogeneous accumulator
    add_standard_track_metrics(...)
    add_standard_visrun_metrics(...)
    add_standard_frame_metrics(...)
    evaluate_standard_evaluation_scene(...)
    evaluate_standard_evaluation_scene_indexed(index, filtering_id, split_id)
    reduce_standard_evaluation_summaries(...)
    JSON projection for result-side persistence
```

The evaluator has exactly three top-level metric traversal loops:

```text
for TRACK scope:  add_standard_track_metrics(...)
for VISRUN scope: add_standard_visrun_metrics(...)
for FRAME scope:  add_standard_frame_metrics(...)
```

The three add functions live beside the private accumulator and orchestrate
independent pure metric leaves. Each leaf returns an immutable scalar/vector
contribution and receives only already sliced native tensors; it cannot observe
scope identity or rebuild traversal. No metric leaf calls another metric leaf.
Public inputs, summaries, results, scope orders, and offsets are immutable.

`hjlib-experiments-results` changes only the heavy explicit evaluation path and
its versioned store/report projection:

- formal source identity and GT/population options still come from the existing
  catalog;
- the result-side composition maps VirtualCrowd accepted spans or WorldPose
  filtering ranges to immutable base-VISRUN labels before evaluation; it does
  not infer base runs from selected Loader output;
- `(protocol_id, entry)` resolves one existing local Standard Loader binding;
- one lazy adapter-factory dispatcher constructs the already implemented Genmo,
  GVHMR, or TRAM binding and runtime;
- existing standard VC/WP normalization bridges load scenes;
- the evaluator returns scene sufficient statistics and one global result;
- the store writes a new `standard-v1` profile without rewriting v1/v2 records;
- fast-test protocols retain the old adapter path.

`standard-v1` persists overall and scene sufficient statistics only. Cross-person
FRAME metrics cannot be assigned to one track without inventing a second
weighting contract, so this profile deliberately has no per-track persisted
records and rejects `compare --level sequence`; existing v1/v2 sequence records
remain readable. Its codec/validator dispatch is profile-specific rather than
forcing the old NAIVE track schema onto the new metrics.

The lazy dispatcher is an application/plugin boundary, not a declared reverse
dependency from `hjlib-experiments-results` to the TPA packages. Missing method
plugins fail only after that standard entry is explicitly selected.

## Smoke-Test Standard

Synthetic smoke must prove:

1. TRACK retains gaps, equal supplied base-VISRUN labels survive a selection
   hole for SEQ alignment, that same hole splits acceleration support, and FRAME
   groups identities;
2. all 14 metrics have a non-empty support on a constructed scene;
3. translation, rotation, root error, acceleration, OKS, PPDS, and PA-PPDS each
   change under an input perturbation that targets only that behavior;
4. `ACC-JOINT-RATIO` divides per joint before its equal-weight mean;
5. global reduction combines raw sufficient statistics before division;
6. metric leaf signatures expose no sequence/index/identity inputs from which
   scope traversal could be rebuilt;
7. CLI dispatch for a formal source calls the Standard Loader path and does not
   call the legacy adapter evaluator;
8. fast-test compatibility still calls the legacy adapter path;
9. store validation accepts `standard-v1`, rejects malformed counts/sums, and
   continues to read v1/v2.

The real-data check selects one formal scene, runs through the existing local
binding, Loader, interpreter, normalization, evaluator, and result projection.
It separately reports Loader/interpreter time, normalization/join time, index
time, metric time, and reduction time. Index acceptance is
`index/(index+metrics) <= 0.01`.

## Migration Plan

1. Add and review the scope/index and metric mathematics.
2. Implement the evaluation core and synthetic smoke in `hjlib-evaluation`.
3. Add `standard-v1` persistence and formal Standard Loader dispatch in
   `hjlib-experiments-results` while preserving fast-test compatibility.
4. Run targeted smoke and strict typing in both repositories.
5. Run one formal scene and profile the named phases.
6. Optimize only a measured bottleneck; do not change the Loader without Loader
   evidence.
7. Run the separate metric-audit task.

## Modification History

- 2026-09-15: created from the user-confirmed 14-metric profile, three shared
  scope traversals, two-task execution split, and negligible-index requirement.
- 2026-09-15: accepted Mathematical Architecture review findings: protocol base
  VISRUNs no longer split on selection holes, acceleration receives separate
  precomputed exact-run sub-boundaries, ratio sufficient-statistic shapes and
  global denominator timing are explicit, and index timing is isolated.
- 2026-09-15: accepted Code Architecture review findings: pure metric leaves
  cannot access traversal identity, accumulator ownership is single-module, and
  `standard-v1` explicitly omits/rejects the non-reconstructible sequence-level
  persistence view while preserving legacy profiles.
- 2026-09-15: accepted the bounded Mathematical Architecture re-review finding
  by making protocol-owned base-VISRUN labels an explicit evaluator input;
  selected rows can no longer be mistaken for the complete run domain.
- 2026-09-15: implemented the three-loop evaluator, lightweight tensor-free
  join/index, 14 independent metric leaves, standard-v1 CLI/store projection,
  and formal Standard Loader dispatch. Real Genmo VC scene2 profiling after
  semantic index validation measured 0.0560 s indexing versus 7.203 s metrics
  (0.772%), satisfying the 1% gate; runtime open and
  Loader/interpreter/normalization measured 72.96 s and 65.50 s.
