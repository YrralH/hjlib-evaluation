'''Generic TestSet builder over the washed hjlib-dataset-assembly filter store.

Consolidates the monolith ``per_dataset/testset_builder_{wp,jta}.py`` (which were
identical except ``name_dataset`` + the ``wp`` vs ``<dataset>`` filter-stats dir
token + a one-line meta difference) into ONE parametrized builder. Per-dataset
facts (dump folder / fps / filter-store root) are resolved in ``get_by_dataset``
and injected at construction.

Alignment guarantee: the divider's flat-index list and the scene-level
Test_Segment list are produced in ONE loop, iterating scenes + each scene's
modifications in the exact order ``partition_filtered_sub_seq`` uses, so
``len(divider) == len(test_segments)`` and index ``i`` of each refers to the same
kept sub-segment. ``build_test_assembly`` then hands ``testset.divider`` to the
assembly factory (injection), so the Dataset index matches the GT-lookup index.

Scope: ``full`` / ``visualize`` (split-aligned: every split scene has an entry in
the store under these policies). The curated subsets (``visualize_v2`` /
``visualize_v3`` / ``*_smoke``) narrow scenes below the split and are deferred --
see ``docs/design/migration.md`` + EVAL Cross-lib TODO.
'''
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, override

from hjlib_dataset_assembly.dataset_builder.divider import Filtered_Sub_Seq_Divider
from hjlib_dataset_assembly.dataset_builder.filter_store import Filter_Modifications_Store
from hjlib_dataset_assembly.dataset_builder.label_manager import Seq_Label_Manager
from hjlib_dataset_assembly.dataset_builder.seq_windows import read_split_scenes

from hjlib_evaluation.test_segment import Test_Segment
from hjlib_evaluation.testset import Filter_Stats, TestSet
from hjlib_evaluation.testset_builder_base import TestSet_Builder_Base


POLICIES_SUPPORTED: Tuple[str, ...] = ('full', 'visualize')
POLICIES_DEFERRED:  Tuple[str, ...] = (
    'visualize_v2', 'visualize_v3',
    'visualize_smoke', 'visualize_v2_smoke', 'visualize_v3_smoke',
)


class TestSet_Builder(TestSet_Builder_Base):
    '''Configured per dataset: dump-label root (for the seq-local -> scene-level
    offset) + filter-store root + fps.'''

    def __init__(
            self,
            name_dataset:      str,
            path_root_label:   str,
            path_filter_store: str,
            fps:               float,
            filter_version:    Optional[str] = None,
        ):
        '''@param path_root_label: the dumped ``<dataset>_<leaf>`` root (e.g.
            ``<dump_root>/jta_smpl_fitted``); used both for the seq scene-offset and
            as the assembly factory's ``path_root``.
        @param path_filter_store: the ``seq_modifications_jsonbin`` store root.
        @param filter_version: store version; ``None`` follows the store's current.'''
        self.name_dataset      = name_dataset
        self.path_root_label   = path_root_label
        self.path_filter_store = path_filter_store
        self.fps               = fps
        self.filter_version    = filter_version

    @override
    def build(self, policy: str, split: str) -> TestSet:
        if policy in POLICIES_DEFERRED:
            raise NotImplementedError(
                'policy %r (curated subset) is deferred: it narrows scenes below the '
                'split, which the assembly factory split path cannot express yet. See '
                'docs/design/migration.md + EVAL Cross-lib TODO. Supported: %s'
                % (policy, list(POLICIES_SUPPORTED)))
        assert policy in POLICIES_SUPPORTED, (
            'unknown policy %r; supported: %s' % (policy, list(POLICIES_SUPPORTED)))

        store = Filter_Modifications_Store(Path(self.path_filter_store))
        # flag_test_mode is fine for the offset: index_original_multi_person_seq_start is
        # metadata present on both label variants. Used ONLY for the scene-level offset.
        label_manager = Seq_Label_Manager(self.path_root_label, flag_test_mode=True)

        # Same scene order the factory uses (read_split_scenes), so the factory-side
        # divider built from testset.divider stays aligned with these segments.
        list_name_scene = read_split_scenes(self.path_root_label, split)
        if split == 'all' and not list_name_scene:
            list_name_scene = label_manager.list_name_scene()

        list_scene_seq_range: List[Tuple[str, str, int, int]] = []
        list_test_segment:    List[Test_Segment]              = []

        n_raw = n_bias_drop = n_short_drop = n_kept_seq = 0
        seg_lengths: List[int] = []

        for name_scene in list_name_scene:
            modifications = store.read_modifications(policy, name_scene, version=self.filter_version)
            for mod in modifications:
                assert mod.name_scene == name_scene, (mod.name_scene, name_scene)
                n_raw += 1
                if mod.dropped_whole:
                    n_bias_drop += 1
                    continue
                if len(mod.kept_subsegments) == 0:
                    n_short_drop += 1
                    continue
                n_kept_seq += 1

                # IFO_S = scene-level start of this raw name_seq; converts seq-local ->
                # scene-level (monolith get_single_seq_info().index_frame_original_start).
                label = label_manager.get_full_label(name_scene, mod.name_seq)
                index_frame_original_start = label.index_original_multi_person_seq_start
                for (s_local, e_local) in mod.kept_subsegments:
                    list_scene_seq_range.append((name_scene, mod.name_seq, int(s_local), int(e_local)))
                    seg_lengths.append(int(e_local - s_local))
                    list_test_segment.append(Test_Segment(
                        name_dataset               =self.name_dataset,
                        name_scene                 =name_scene,
                        name_seq                   =mod.name_seq,
                        id_person                  =mod.id_person,
                        index_frame_original_start =int(index_frame_original_start + s_local),
                        index_frame_original_end   =int(index_frame_original_start + e_local),
                    ))

        filter_stats = Filter_Stats(
            bias_config_name        =None,
            bias_tag                =None,
            min_bias_segment_length =None,
            n_frame_min_seq         =None,
            produced_at             =None,
            raw_seq_count       =n_raw,
            bias_dropped_count  =n_bias_drop,
            short_dropped_count =n_short_drop,
            kept_seq_count      =n_kept_seq,
            test_segment_count  =len(list_test_segment),
            shortest_segment    =min(seg_lengths) if seg_lengths else 0,
            longest_segment     =max(seg_lengths) if seg_lengths else 0,
            total_frames        =sum(seg_lengths),
        )

        divider = Filtered_Sub_Seq_Divider(list_scene_seq_range)
        return TestSet(
            name_dataset    =self.name_dataset,
            policy          =policy,
            split           =split,
            divider         =divider,
            test_segments   =list_test_segment,
            path_root_label =self.path_root_label,
            fps             =self.fps,
            filter_stats    =filter_stats,
        )
