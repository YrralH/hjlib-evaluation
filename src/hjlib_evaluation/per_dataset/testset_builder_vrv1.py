'''Whole dumped VRv1 runs, with explicit main-camera versus GoPro selection.'''
from typing import override

from hjlib_dataset_assembly import get_dataset_facts, get_vrv1_scene_names
from hjlib_dataset_assembly.dataset_builder.divider import Filtered_Sub_Seq_Divider
from hjlib_dataset_assembly.dataset_builder.label_manager import (
    Seq_Label_Manager, extract_name_seq_info,
)
from hjlib_dataset_assembly.dataset_builder.seq_windows import read_split_scenes

from hjlib_evaluation.test_segment import Test_Segment
from hjlib_evaluation.testset import Filter_Stats, TestSet
from hjlib_evaluation.testset_builder_base import TestSet_Builder_Base


class VRv1_TestSet_Builder(TestSet_Builder_Base):
    '''Preserve all synchronized frames and never consume a filter store.'''

    def __init__(self, name_dataset: str, path_root_label: str, fps: float):
        self.scene_names = get_vrv1_scene_names(name_dataset)
        if fps != get_dataset_facts(name_dataset).fps:
            raise ValueError('VRv1 dataset and FPS differ')
        self.name_dataset = name_dataset
        self.path_root_label = path_root_label
        self.fps = fps

    def require_scene_view(self, name_scene: str) -> None:
        '''Require the assembly-owned exact camera-family roster.'''
        if name_scene not in self.scene_names:
            raise ValueError('VRv1 scene belongs to the other camera family or is unknown: %s' % name_scene)

    @override
    def build(self, policy: str, split: str) -> TestSet:
        if policy != 'full':
            raise ValueError('VRv1 dumped-run evaluation supports only full policy')
        manager = Seq_Label_Manager(self.path_root_label, flag_test_mode=True)
        scenes = read_split_scenes(self.path_root_label, split)
        if split == 'all' and not scenes:
            scenes = manager.list_name_scene()
        # Reject cross-view roots even when the chosen split hides the foreign scene.
        for name_scene in manager.list_name_scene():
            self.require_scene_view(name_scene)
        ranges: list[tuple[str, str, int, int]] = []
        segments: list[Test_Segment] = []
        for name_scene in scenes:
            self.require_scene_view(name_scene)
            names = manager.list_name_seq(name_scene)
            if not names:
                raise ValueError('VRv1 split scene has no dumped runs: %s' % name_scene)
            for name_seq in names:
                label = manager.get_full_label(name_scene, name_seq)
                start = label.index_original_multi_person_seq_start
                end = label.index_original_multi_person_seq_end
                if end - start != label.num_frame_seq or label.num_frame_seq < 1:
                    raise ValueError('VRv1 dumped run frame range differs')
                id_person, _ = extract_name_seq_info(name_seq)
                ranges.append((name_scene, name_seq, 0, label.num_frame_seq))
                segments.append(Test_Segment(
                    self.name_dataset, name_scene, name_seq, id_person, start, end))
        lengths = [segment.length for segment in segments]
        stats = Filter_Stats(
            None, None, None, None, None,
            len(segments), 0, 0, len(segments), len(segments),
            min(lengths) if lengths else 0,
            max(lengths) if lengths else 0, sum(lengths),
        )
        return TestSet(
            self.name_dataset, policy, split, Filtered_Sub_Seq_Divider(ranges),
            segments, self.path_root_label, self.fps, stats)
