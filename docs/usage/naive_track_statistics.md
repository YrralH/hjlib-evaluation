# GT-MOT sequence statistics

`evaluate_naive_tracks(sequence, filtering_id, split_id, selected_gt_mask)` returns
`tuple[Naive_Track_Statistics, ...]`, sorted by native GT track ID. Each record holds
`gt_track_id`, maximal half-open `frame_ranges`, and additive `statistics` using
the existing `VirtualCrowd_Naive_Comparison_Sequence_Summary` schema. Its historical
Sequence name denotes a scene; in this API each instance instead summarizes one
track partition and still carries the actual scene identity.

The input is one complete scene in `Corrected_Crowd_Sequence`. Selection and exact
GT identity remain caller-owned. No local person-frame metric arrays are persisted.
GT row 0 is a normal selected target. Empty selection returns an empty tuple.
Temporal support crosses prediction runner chunks but never missing selected GT
frames. Do not submit each runner chunk independently and merge its acceleration.
Finite signed prediction camera depth is a valid normalized result fact.
`evaluate_naive_tracks()` excludes non-positive-depth joints only from OKS
support; a person-frame with no remaining joint adds no OKS sum/count while
remaining in all non-OKS track populations.

`merge_naive_statistics(records, scene_id)` adds compatible sufficient statistics;
filtering, split and profile must agree. Multiple records from the same scene are
expected. The caller must enforce disjoint GT track identities: summary statistics
alone contain no identity list from which overlap could be detected.

`finalize_naive_statistics(summary)` returns `mpjpe_world_mm`, `t_mpjpe_mm`,
`pa_mpjpe_mm`, `oks_vis`, and `acc_root_ratio` for current V2 summaries.
Historical V1 summaries return their original four keys. Unsupported metrics
or a zero reference
acceleration denominator are `None`, not zero. Overall strict evaluation can still
use `reduce_virtualcrowd_naive_comparison_summaries` on merged unique scenes.

Persist each record with its full scene/GT track identity and frame ranges.
Aggregate sums/counts, not per-person means or per-person acceleration ratios.
Disk layout and presentation formats belong to the consuming experiment store.

## Example

```python
from hjlib_evaluation import (
    evaluate_naive_tracks,
    finalize_naive_statistics,
    merge_naive_statistics,
)

records = evaluate_naive_tracks(
    sequence,
    filtering_id='vc.visible_common',
    split_id='vc.test6',
    selected_gt_mask=selected_gt_mask,
)
scene_summary = merge_naive_statistics(
    [record.statistics for record in records],
    scene_id=sequence.scene_id,
)
metrics = finalize_naive_statistics(scene_summary)
```

## Signatures and selection

| Need | API |
|---|---|
| One validated scene → ordered GT-track records | `evaluate_naive_tracks(sequence, filtering_id, split_id, selected_gt_mask)` |
| Compatible per-track summaries → one scene summary | `merge_naive_statistics(summaries, scene_id)` |
| Additive summary → version-exact nullable metric values | `finalize_naive_statistics(summary)` |
| Persist records or format a report | Implement in the consuming result store |
