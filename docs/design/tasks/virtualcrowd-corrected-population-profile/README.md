# VirtualCrowd Corrected Selected-Population Profile

## Requirements

The completed corrected crowd protocol owns an immutable schema-v1 two-view
result: `GT_VISIBLE` and the frozen 167,243-key `C4D_DYCROWD_COMMON`. Campaign 04
needs one additional comparison population without relabeling either legacy
view:

`C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9`

It is exactly the old common key set filtered by
`count(gt_visibility_native > 0, axis=COCO17) >= 9`. The released source channel
values are `0`, `0.5`, and `1`; both `0.5` and `1` count because the accepted
author-relayed semantics classify only zero as fully/scene-occluded. The
observed release count is 159,405, but that count is run evidence rather than a
generic library allowlist.

The addition must:

- reuse the existing 15 metrics, alignment fits, temporal scopes, and exact
  micro-reduction;
- leave every existing schema-v1 type, constant, JSON shape, parser, and result
  unchanged;
- carry the selected population's honest name in every new summary/result;
- expose support counts but not invent completeness/precision/recall for a
  method-intersection view;
- add no population registry, enum, plugin system, or automatic approval gate;
- remain method-neutral and accept normalized `Corrected_Crowd_Sequence` input.

## Mathematical Architecture

Let one normalized scene contain `G` GT occurrences, `P` prediction
occurrences, accepted one-to-one matched row arrays `(m_g, m_p)`, native
COCO-17 visibility `V in {0,0.5,1}^{Gx17}`, an old-common mask
`C in {false,true}^G`, and an externally supplied boolean selection
`S in {false,true}^G`.

The selected-view preconditions are:

1. `S.shape == (G,)` with exact boolean dtype;
2. `C.shape == (G,)` with exact boolean dtype and `C` is a subset of the
   corrected base visible domain `B_g = any(V_g > 0)`;
3. `S` is a subset of the corrected base visible domain
   `B_g = any(V_g > 0)`;
4. the view name is nonempty, differs from both reserved legacy names
   `GT_VISIBLE` and `C4D_DYCROWD_COMMON`, and is carried as an opaque semantic
   label rather than resolved through a registry.

For Campaign 04 the selection is:

```text
S_g = C_g AND count_j(V[g,j] > 0) >= 9
```

The selected matched pairs are:

```text
Q = {q | S[m_g[q]]}
selected_gt_rows = m_g[Q]
selected_prediction_rows = m_p[Q]
```

All metric populations use the existing corrected functions with those two row
arrays. In particular:

- SMPL-24 joint position/alignment metrics keep metre input and millimetre
  presentation scaling;
- OKS includes only source joints with `V > 0`, so `0.5` and `1` contribute;
- VISRUN labels are built once from the legacy base-visible domain before
  applying `S`, preserving the accepted run definition;
- TRACK scopes retain native GT track IDs;
- acceleration still requires exact consecutive frame triples after selected
  row restriction;
- per-scene sufficient-statistic sums use float64 and counts use int64; scene
  reduction keeps lexical scene order and `math.fsum` across scenes.

The selected-view summary records:

```text
scene_id
view_name
selected_gt_count
matched_selected_count
metric_sample_sums[15]
metric_sample_counts[15]
accel_exact_consecutive_triple_count
```

The result records summed support, the 15 reduced values, metric units, and
triple count. It deliberately has no `tp/fn/fp/precision/recall/f1`: the frozen
common population is a comparison support, not a detector-completeness domain.
Campaign receipts may separately require `matched_selected_count ==
selected_gt_count` for the three accepted native result sets.

Empty selected populations are structurally valid at a single-scene helper
boundary and produce zero counts/`None` metric values after reduction. The
campaign operation requires a nonempty full population and the observed exact
159,405 count. Non-finite inputs, foreign rows, a non-visible selected row, or
mixed view names across scene summaries fail before publication.

## Code Architecture

Residence remains in the existing corrected modules:

```text
src/hjlib_evaluation/corrected_crowd_data.py
    CORRECTED_CROWD_SELECTED_VIEW_SCHEMA_VERSION
    Corrected_Crowd_Selected_View_Sequence_Summary
    Corrected_Crowd_Selected_View_Result
    validate_corrected_crowd_selected_view_name(...)
    selected-view JSON readers/writers

src/hjlib_evaluation/corrected_crowd_protocol.py
    evaluate_corrected_crowd_selected_view(...)
    reduce_corrected_crowd_selected_view_summaries(...)

src/hjlib_evaluation/corrected_crowd_population.py
    C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9
    make_coco17_visible_ge9_common_mask(...)
```

The new population module owns only the named mask operation. It accepts the
already-normalized native visibility and old-common mask; it does not load a
dataset, inspect method output, or enforce the observed release count.

The evaluator adds one module-local, stateless, non-underscored function:

```text
evaluate_corrected_crowd_matched_rows(
    validated_sequence,
    matched_gt_rows,
    matched_prediction_rows,
    base_visible_visrun_labels,
) -> (metric_sums[15], metric_counts[15], exact_triple_count)
```

It owns a temporary `(1,15)` float64/int64 accumulator, calls the existing four
metric helper families at `view_index=0`, and returns owned one-dimensional
statistics. It never constructs VISRUN labels or selects rows. The legacy entry
builds base-visible VISRUN labels once, invokes this function for each of its
two masks, and installs the returned vectors into the existing two rows. The
selected entry builds the same base-visible labels once and invokes it for its
single explicit selection. Existing public two-view dataclasses and serializers
are not changed. New selected-view serialization has its own schema/version and
exact keys. Both selected-view dataclass constructors call the non-underscored
`validate_corrected_crowd_selected_view_name`; it rejects both reserved legacy
names before any instance can be serialized. Parsers construct those types and
therefore inherit the same gate. Other nonempty names remain opaque and require
no allowlist.

The package root re-exports the new constant, mask function, selected-view
types, evaluator, reducer, and JSON functions. No TPA package is imported.

The operation owner remains the TPA/application that has native artifacts. It
constructs the normalized sequence, supplies `S`, evaluates a scene, writes a
small summary, and releases the scene arrays. This keeps the existing bounded
one-method/one-scene worker lifecycle.

## Smoke-Test Standard

Data-free tests must prove:

- `0` does not count and both `0.5` and `1` count toward the threshold;
- exactly 8 positive channels is excluded and exactly 9 is included;
- the selected mask is always a subset of the supplied old common mask;
- invalid dtype/shape/visibility and selected-invisible rows fail;
- selected-view metrics equal the legacy common-view row when supplied the
  exact old common mask;
- a three-frame visible track with the middle frame removed by selection keeps
  its original VISRUN/TRACK scope labels, while ACCEL contributes no triple
  across that selection hole;
- legacy schema-v1 JSON bytes/round-trip expectations remain unchanged;
- direct summary/result construction and JSON parsing both reject each of the
  two reserved legacy names;
- selected summaries/results round-trip their exact view name and reject mixed
  names during reduction;
- scene-order-independent reduction preserves exact counts and expected values.

Real pilot/full gates additionally reconcile 159,405 selected GT rows across
the eight released scenes and bind the old common manifest identity.

## Migration Plan

1. Add the independent selected-view data/serialization surface.
2. Factor metric-row evaluation without changing legacy outputs.
3. Add the GE9 named mask primitive and public re-exports.
4. Run strict pyright, corrected-protocol smoke, and legacy tests.
5. Let method-owned producers/operations adopt the selected-view entry.

No legacy artifact migration occurs. Existing 167,243-key manifests, worker
summaries, result JSON, and campaign evidence remain byte-identifiable under the
old contract.

## Modification History

- 2026-08-19: Drafted the additive selected-view architecture for Campaign 04.
  It preserves the old two-view schema, treats GE9 as an explicit mask over the
  frozen common set, and adds no registry or count gate.
- 2026-08-19: Mathematical Architecture review found no Critical issue. Accepted
  three concerns by reserving both legacy view names, defining the old-common
  mask contract, and adding a selection-hole temporal smoke gate.
- 2026-08-19: Code Architecture review's reserved-name Critical was already
  closed mathematically but remained open at direct construction. The accepted
  revision puts one validator in both selected-view dataclass constructors;
  parsers inherit it. Also froze the shared matched-row evaluator signature,
  accumulator ownership, and base-visible VISRUN input.
