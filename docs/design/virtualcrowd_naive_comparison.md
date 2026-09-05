# VirtualCrowd provisional four-metric comparison

`VC_NAIVE_COMPARISON_METRICS_V1` is a narrow, method-neutral comparison profile
for Ours, Crowd4D and DyCrowd on one caller-selected VirtualCrowd population. It
does not replace the stable default or Crowd4D-native profiles.

## Ownership and boundary

`hjlib-evaluation` owns the metric mathematics and additive cross-scene
reduction. Dataset assembly, model inference, camera choice, prediction
adaptation, result registration and visualization remain in their respective
experiment/adapter layers.

The evaluator consumes `Corrected_Crowd_Sequence`, the registry-owned
`filtering_id` and `split_id` as separate identities, plus an exact GT-row
boolean mask. It joins selected rows through the sequence's existing
`prediction_identity_target_gt_rows`; every selected GT row must have exactly
one direct-target prediction before any metric is computed. This deliberately
does not use the stable corrected-crowd matched-row subset because that subset
excludes zero-visible rows retained inside `vc.visible_common` runs.

## Ordered metrics

| Metric | Unit | Display direction | Reduction |
|---|---:|---:|---|
| `MPJPE-WORLD` | mm | ↓ | micro mean over selected SMPL-24 joint occurrences |
| `T-MPJPE` | mm | ↓ | pelvis-relative micro mean over the same joints |
| `OKS-VIS` | fraction | ↑ | direct-paired mean over rows with native-visible GT COCO joints |
| `ACC-ROOT-RATIO` | unitless | →1 | global predicted/reference root-acceleration magnitude sums |

`T-MPJPE` subtracts joint 0 independently from prediction and GT; it is not an
optimized translation fit. `OKS-VIS` uses COCO-17 sigmas, GT bbox area and only
GT native visibility `> 0`, with no matching or recall multiplier. A zero-visible
row stays in the 3D/temporal population but adds no OKS support.

`direct_target_join()` constructs one immutable
`VirtualCrowd_Direct_Target_Join`: it takes a fresh validated sequence
snapshot and verifies aligned unique GT/prediction rows, bounds, exact direct
targets and frame identity. The public metric leaves accept only that typed
join:

- `compute_virtualcrowd_mpjpe_world_statistics`;
- `compute_virtualcrowd_t_mpjpe_statistics`;
- `compute_virtualcrowd_oks_vis_statistics`;
- `compute_virtualcrowd_acc_root_ratio_statistics`.

Each returns additive statistics only. Population projection and cross-scene
reduction remain outside the leaves;
`evaluate_virtualcrowd_naive_comparison()` is the fixed wiring over one shared
join. Normalized `Corrected_Crowd_Sequence` construction no longer executes
PPDS or similarity-fit preflight for profiles that do not request those
metrics. Full corrected-profile evaluators still execute their layout metrics
when evaluated.

`ACC-ROOT-RATIO` applies replicate-padded central difference twice to each
maximal exact-consecutive selected GT track segment, trims three samples from
each end, then divides global predicted magnitude sum by global GT magnitude
sum. There is no FPS multiplier. Empty global support or a non-positive GT
denominator is invalid.

Scene summaries preserve metric sums and support counts. The reducer sorts
scene IDs, rejects duplicates and profile/population mismatches, and uses global
sufficient statistics rather than averaging scene ratios.

## Registered LSV-HR matrix composition

`lsvhr_evaluation.py` owns the reusable composition immediately above the
per-scene metric wiring. `LSVHR_Evaluation_Population` contains one
dataset-std `VirtualCrowd_Eval_Population_Selection` plus exact filtering,
split, rule and split-scene identities. Its spans are derived by filtering the
selection; callers cannot inject a second independent span list.

`LSVHR_Evaluation_Entry` binds an official publication identity to a
structural `LSVHR_Method_Loader.load_scene(scene_id)`. The loader owns method
decoding; evaluation owns selected-GT key projection, calls the fixed NAIVE
evaluator once per scene, and reduces once per entry.
`evaluate_lsvhr_virtualcrowd_matrix()` preserves caller order and rejects empty
or duplicate entry sets. It does not discover registry records, inspect
artifacts, render reports or cache normalized scenes.

The registry/report owner must validate that every row has the same scene,
selected, matched, joint, OKS and acceleration support. This remains outside
the method-neutral matrix because official row rosters and report identity are
registry facts.

The reviewed task design and edge-case rationale are in
[the task residence](tasks/virtualcrowd-naive-comparison-metrics/README.md).
