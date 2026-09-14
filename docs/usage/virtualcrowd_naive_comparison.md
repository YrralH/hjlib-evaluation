# VirtualCrowd provisional NAIVE evaluator

Use this API after a method adapter has normalized one scene into
`Corrected_Crowd_Sequence` and the population owner has produced the exact GT
row mask. The evaluator does not load dumps or run a model.

For WP, the method adapter may supply schema 2 `Corrected_Crowd_Sequence` with
`coordinate_frame='WORLD_METRES'`, native non-negative GT row IDs and WP
filtering/split IDs. Use the same scene evaluator/reducer, not the VC-specific
registered matrix. The adapter must preserve source cadence and compute GT
and prediction projections/depth from their respective cameras. The five
formulas are unchanged; retain the explicit WP protocol label in reports.

```python
from hjlib_evaluation import (
    compute_virtualcrowd_mpjpe_world_statistics,
    compute_virtualcrowd_pa_mpjpe_statistics,
    direct_target_join,
    evaluate_virtualcrowd_naive_comparison,
    reduce_virtualcrowd_naive_comparison_summaries,
)

join = direct_target_join(sequence, selected_gt_mask)
mpjpe_sum_m, joint_count = (
    compute_virtualcrowd_mpjpe_world_statistics(join)
)

scene_summary = evaluate_virtualcrowd_naive_comparison(
    sequence=normalized_scene,
    filtering_id='vc.visible_common',
    split_id='vc.test6',
    selected_gt_mask=selected_gt_mask,
)

result = reduce_virtualcrowd_naive_comparison_summaries([scene_summary])
print(result.mpjpe_world_mm)
print(result.t_mpjpe_mm)
print(result.pa_mpjpe_mm)
print(result.oks_vis)
print(result.acc_root_ratio)
```

Use the five public `compute_virtualcrowd_*_statistics` functions when a caller
needs metric-specific additive statistics. Construct the join once and pass it
to every leaf; naked GT/prediction row arrays are deliberately not accepted.

The returned result also exposes population/support counts. Check at least
`selected_gt_count`, `matched_selected_count`, `oks_vis_count` and
`acc_root_sample_count` when presenting a table. `matched_selected_count` is
required to equal `selected_gt_count`; missing or duplicate direct-target
predictions raise.

Interpret the metrics as:

- `MPJPE-WORLD ↓`: absolute SMPL-24 world error in mm.
- `T-MPJPE ↓`: pelvis-relative SMPL-24 error in mm.
- `PA-MPJPE ↓`: per-person-frame positive-scale proper-similarity SMPL-24
  error in mm; all 24 joints are included and reflection is prohibited.
- `OKS-VIS ↑`: direct-paired COCO-17 OKS on its native-visible support.
- `ACC-ROOT-RATIO →1`: predicted/GT root acceleration-magnitude ratio.

Finite signed prediction camera depth is valid normalized input. OKS includes
only joints that are both native-visible and positive-depth in the paired
prediction. A row with no remaining joint contributes no OKS sum/count while
remaining in every 3D population.

`ACC-ROOT-RATIO` is not a lower-is-better residual. Values below and above one
respectively mean lower and higher predicted acceleration magnitude than GT.
It is undefined when the selected data has no supported temporal sample or the
global GT acceleration sum is zero.

This is a provisional comparison profile. Do not label its values as
Crowd4D-author metrics or mix them with `VC_HJ_DEFAULT_V1` / native-profile
columns without an explicit protocol label.
Current calls emit `VC_NAIVE_COMPARISON_METRICS_V2`. Historical V1 records have
no PA-MPJPE and must remain visibly version-separated.

## Evaluating an official-entry matrix

When the caller already has one dataset-std selection, exact split scenes and
method-owned loaders, use the LSV-HR composer instead of reimplementing the
scene loop:

```python
from hjlib_evaluation import (
    LSVHR_Evaluation_Entry,
    LSVHR_Evaluation_Population,
    LSVHR_Evaluation_Profile,
    evaluate_lsvhr_virtualcrowd_matrix,
)

population = LSVHR_Evaluation_Population(
    filtering_id='vc.visible_common',
    split_id='vc.test6',
    rule_id='vc.visibility_continuity_v1',
    selection=visible_common_all8,
    split_scene_ids=(
        'scene2', 'scene2_view2', 'scene3',
        'scene3_view2', 'scene4', 'scene4_view2',
    ),
)
entries = tuple(
    LSVHR_Evaluation_Entry(entry_id, loaders[entry_id])
    for entry_id in ordered_entry_ids
)
rows = evaluate_lsvhr_virtualcrowd_matrix(
    LSVHR_Evaluation_Profile.NAIVE,
    entries,
    population,
)
```

The selection must cover every exact split scene. Each loader may stream one
scene and discard it after reduction; do not build a whole-matrix scene cache.
Use `hjlib-experiments-results` when the entry order and population must be
resolved from a registered complete protocol and serialized as a report.
