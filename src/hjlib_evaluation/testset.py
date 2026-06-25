'''TestSet: a dataset's test segments under one subset policy, plus its divider.

Port of monolith ``test/protocol_dynamic/testset.py``. Two divergences from the
monolith (both follow from reusing the washed hjlib-dataset-assembly filter store
instead of the monolith pkls + ``Dataset_Single_Seq``):

- the divider is the assembly ``Filtered_Sub_Seq_Divider`` (imported, not re-homed);
- ``Filter_Stats`` provenance fields (bias_config_name / bias_tag / ...) are now
  best-effort ``None``: the one-time wash into the versioned store dropped the
  monolith ``_meta.json`` sidecars (the store version metadata carries only
  created / note / producer / source), so only the count fields are derivable.

The TestSet also carries the recipe ``build_test_assembly`` needs to construct the
inference-input Dataset via the assembly factory (``path_root_label`` + ``fps``);
``name_dataset`` + ``divider`` are already held.
'''
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Set

from hjlib_dataset_assembly.dataset_builder.divider import Filtered_Sub_Seq_Divider

from hjlib_evaluation.test_segment import Test_Segment


@dataclass(frozen=True)
class Filter_Stats:
    '''Provenance + drop counts attached to a TestSet by its builder.

    Provenance fields are ``Optional``: the washed versioned store carries no
    monolith ``_meta.json`` sidecar, so they are ``None`` and ``summary()`` prints
    '(unknown)'. The count fields are always computed from the modifications.
    '''
    # Provenance (None on the washed store -- no _meta.json sidecar).
    bias_config_name:        Optional[str]
    bias_tag:                Optional[str]
    min_bias_segment_length: Optional[int]
    n_frame_min_seq:         Optional[int]
    produced_at:             Optional[str]

    # Drop counts at the seq level (mod -> outcome).
    raw_seq_count:        int
    bias_dropped_count:   int     # mod.dropped_whole == True
    short_dropped_count:  int     # mod.dropped_whole == False AND len(kept_subsegments) == 0
    kept_seq_count:       int     # mod.dropped_whole == False AND len(kept_subsegments) > 0

    # Test-segment level (after kept-subsegment splitting).
    test_segment_count:   int
    shortest_segment:     int     # 0 if no segments
    longest_segment:      int     # 0 if no segments
    total_frames:         int


class TestSet:
    '''Read-only container.

    - len(testset) == len(testset.divider) == len(testset.test_segments)
    - testset.divider feeds the assembly Dataset (seq-local coords)
    - testset.test_segments[i] has the matching scene-level coords for GT lookup
    - path_root_label + fps + name_dataset let build_test_assembly route the
      divider through the assembly factory.
    '''

    def __init__(
            self,
            name_dataset:    str,
            policy:          str,
            split:           str,
            divider:         Filtered_Sub_Seq_Divider,
            test_segments:   List[Test_Segment],
            path_root_label: str,
            fps:             float,
            filter_stats:    Optional[Filter_Stats] = None,
        ):
        assert len(divider) == len(test_segments), (len(divider), len(test_segments))
        self.name_dataset    = name_dataset
        self.policy          = policy
        self.split           = split
        self.divider         = divider
        self.test_segments   = test_segments
        self.path_root_label = path_root_label
        self.fps             = fps
        self.filter_stats    = filter_stats

    def __len__(self) -> int:
        return len(self.test_segments)

    def get_test_segment(self, flat_index: int) -> Test_Segment:
        return self.test_segments[flat_index]

    def restrict_to_scenes(self, kept_scenes: Iterable[str]) -> None:
        '''Mutate in place to keep only segments whose name_scene is in kept_scenes.

        Rebuilds the divider from the filtered (scene, seq, within_start, within_end)
        list and updates the segment-level fields of filter_stats. seq-level counts
        (raw / bias_dropped / short_dropped / kept) are NOT touched -- they describe the
        upstream pipeline, not this scene-level cut.'''
        kept_set: Set[str] = set(kept_scenes)
        keep_mask = [seg.name_scene in kept_set for seg in self.test_segments]
        n_old = len(self.test_segments)
        new_segments = [s for s, k in zip(self.test_segments, keep_mask) if k]
        new_ranges = [
            (info.name_scene, info.name_seq,
             info.index_within_singleseq_start, info.index_within_singleseq_end)
            for index, keep in enumerate(keep_mask) if keep
            for info in (self.divider.get_seq_info(index),)
        ]
        self.test_segments = new_segments
        self.divider       = Filtered_Sub_Seq_Divider(new_ranges)
        if self.filter_stats is not None:
            seg_lens = [s.length for s in new_segments]
            self.filter_stats = replace(
                self.filter_stats,
                test_segment_count = len(new_segments),
                shortest_segment   = min(seg_lens) if seg_lens else 0,
                longest_segment    = max(seg_lens) if seg_lens else 0,
                total_frames       = sum(seg_lens),
            )
        n_scenes_after = len(set(s.name_scene for s in new_segments))
        print('TestSet.restrict_to_scenes: kept %d / %d segments across %d scenes'
              % (len(new_segments), n_old, n_scenes_after))

    def summary(self) -> str:
        '''Filter-provenance banner + per-scene table + grand total.'''
        lines: List[str] = []
        lines.append('TestSet  dataset=%s  policy=%s  split=%s'
                     % (self.name_dataset, self.policy, self.split))

        # ---- Provenance + filter stats banner ----
        fs = self.filter_stats
        if fs is None:
            lines.append('  filter_stats: (unattached)')
        else:
            cfg = fs.bias_config_name if fs.bias_config_name is not None else '(unknown)'
            tag = fs.bias_tag         if fs.bias_tag         is not None else '(unknown)'
            mbl = fs.min_bias_segment_length if fs.min_bias_segment_length is not None else -1
            nfm = fs.n_frame_min_seq  if fs.n_frame_min_seq  is not None else -1
            ts_str = fs.produced_at   if fs.produced_at      is not None else '(unknown)'
            lines.append('  filter:   bias_config=%s  min_bias_seg_len=%s  N_FRAME_MIN_SEQ=%s'
                         % (cfg, mbl if mbl >= 0 else '?', nfm if nfm >= 0 else '?'))
            lines.append('            bias_tag=%s' % tag)
            lines.append('            produced_at=%s' % ts_str)
            pct_bias  = 100.0 * fs.bias_dropped_count  / max(fs.raw_seq_count, 1)
            pct_short = 100.0 * fs.short_dropped_count / max(fs.raw_seq_count, 1)
            pct_kept  = 100.0 * fs.kept_seq_count      / max(fs.raw_seq_count, 1)
            lines.append('  raw seqs:           %5d' % fs.raw_seq_count)
            lines.append('    bias-dropped:     %5d  (%.1f%%)' % (fs.bias_dropped_count,  pct_bias))
            lines.append('    short-dropped:    %5d  (%.1f%%)  -- no kept_subsegments'
                         % (fs.short_dropped_count, pct_short))
            lines.append('    kept seqs:        %5d  (%.1f%%)' % (fs.kept_seq_count,      pct_kept))
            lines.append('  test_segments:      %5d' % fs.test_segment_count)
            lines.append('  segment length:  shortest=%d  longest=%d  total_frames=%d'
                         % (fs.shortest_segment, fs.longest_segment, fs.total_frames))

        # ---- Per-scene table ----
        per_scene_count:       Counter[str] = Counter()
        per_scene_frame_total: Counter[str] = Counter()
        for ts in self.test_segments:
            per_scene_count      [ts.name_scene] += 1
            per_scene_frame_total[ts.name_scene] += ts.length

        for name_scene in sorted(per_scene_count.keys()):
            lines.append('  %-22s  segments=%3d  frames=%7d'
                         % (name_scene, per_scene_count[name_scene], per_scene_frame_total[name_scene]))

        lines.append('=== grand total: segments=%d  frames=%d  scenes=%d ==='
                     % (len(self.test_segments),
                        sum(per_scene_frame_total.values()),
                        len(per_scene_count)))
        return '\n'.join(lines)
