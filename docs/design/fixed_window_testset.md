# Fixed-Window TestSet

## Purpose

`TestSet` owns the alignment between:

```text
Filtered_Sub_Seq_Divider entry
Test_Segment metadata
dump filename tag
GT lookup range
```

Any operation that turns standard long segments into fixed windows must live on `TestSet`,
not in an experiments-local helper.

## API

Add a pure-return method:

```python
class TestSet:
    def make_fixed_window_testset(
        self,
        length_window: int,
        length_overlap: int = 0,
        tail_policy: str = 'drop',
    ) -> 'TestSet':
        ...
```

The method returns a new `TestSet` and does not mutate the original.

## Semantics

For each original segment range `[start, end)`, with:

```text
stride = length_window - length_overlap
```

derive windows:

```text
[start + k * stride, start + k * stride + length_window)
```

Keep only windows inside `[start, end)` for `tail_policy='drop'`.

The first implementation supports only `tail_policy='drop'`. Shorter tails, padding, and
overlap-aware de-duplication are deferred.

## Coordinate Formula

The divider stores sequence-local coordinates. `Test_Segment` stores scene-level original
frame coordinates. The derived fixed-window subset must bridge both explicitly:

```python
info = self.divider.get_seq_info(index_segment)
segment = self.test_segments[index_segment]
scene_offset = segment.index_frame_original_start - info.index_within_singleseq_start

subset_ranges.append((info.name_scene, info.name_seq, subset_start, subset_end))
subset_test_segments.append(replace(
    segment,
    index_frame_original_start=scene_offset + subset_start,
    index_frame_original_end=scene_offset + subset_end,
))
```

`subset_start` and `subset_end` are sequence-local. The derived segment preserves
`name_dataset`, `name_scene`, `name_seq`, and `id_person`, and updates only the original
scene-level frame interval.

## Naming

Use `subset` in implementation variables for the derived object:

```text
subset_ranges
subset_test_segments
subset_filter_stats
fixed_window_subset_testset
```

This prevents confusing the derived fixed-window testset with the original standard testset.

## Filter Stats

The derived `Filter_Stats` should preserve seq-level provenance/counts and update only the
segment-level fields:

```text
test_segment_count
shortest_segment
longest_segment
total_frames
```

## Tests

Add:

```text
test_smoke/test_testset_fixed_window.py
```

Required smoke cases:

- A long segment produces fixed windows with expected local and original frame ranges.
- The original `TestSet` remains unchanged.
- Include a case with nonzero scene offset and nonzero `id_person`.
- Segment count and frame counts in `Filter_Stats` are updated.
- `length_overlap >= length_window` raises.
- `tail_policy != 'drop'` raises until another policy is designed.

## Relationship To Experiments

`hjlib-experiments` uses this API to build fixed-window subsets for cached-fusion
standard eval. `hjlib-evaluation` does not import experiments or model code.
