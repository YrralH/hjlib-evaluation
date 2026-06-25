'''WorldPose GT provider: world-space SMPL joints + params from the assembly dump.

Port of monolith ``per_dataset/gt_provider_wp.py``, scoped to the metric baseline.
The world-space SMPL fit is already carried by the dumped assembly full label
(``smpl_joints_54_world`` / ``smpl_param_world``), so WP GT needs no raw-dataset
reader -- just the assembly label manager in FULL (non-test) mode. Camera / ground /
video (deferred, base raises) are the only things that would need the raw reader.
'''
from __future__ import annotations

from typing import Dict, Tuple, override

import numpy as np

from hjlib_dataset_assembly.dataset_builder.label_manager import Seq_Label_Manager
from hjlib_dataset_assembly.single_seq.label import Single_Seq_Label

from hjlib_evaluation.eval_meta import Eval_Meta
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.per_dataset.wp_eval_meta import WP_EVAL_META


class WP_GT_Provider(GT_Provider_Base):

    name_dataset = 'worldpose_smpl'

    def __init__(self, path_root_label: str) -> None:
        '''@param path_root_label: the dumped ``<dump_root>/worldpose`` label root.'''
        # flag_test_mode=False: GT SMPL fields (smpl_joints_54_world / smpl_param_world)
        # only exist on the FULL Single_Seq_Label.
        self.label_manager = Seq_Label_Manager(path_root_label, flag_test_mode=False)

    def _full_label(self, name_scene: str, name_seq: str) -> Single_Seq_Label:
        label = self.label_manager.get_full_label(name_scene, name_seq)
        assert isinstance(label, Single_Seq_Label), (type(label), name_scene, name_seq)
        return label

    @staticmethod
    def _seq_local_slice(
            label: Single_Seq_Label,
            frame_range_scene_level: Tuple[int, int],
        ) -> Tuple[int, int]:
        ifo_s = label.index_original_multi_person_seq_start
        ifo_e = label.index_original_multi_person_seq_end
        s, e = frame_range_scene_level
        assert ifo_s <= s < e <= ifo_e, (ifo_s, ifo_e, s, e)
        return s - ifo_s, e - ifo_s

    @override
    def get_smpl_joints_54_world(
            self,
            name_scene: str,
            name_seq:   str,
            frame_range_scene_level: Tuple[int, int],
        ) -> np.ndarray:
        label = self._full_label(name_scene, name_seq)
        s_loc, e_loc = self._seq_local_slice(label, frame_range_scene_level)
        return label.smpl_joints_54_world[s_loc:e_loc]

    @override
    def get_smpl_param_world(
            self,
            name_scene: str,
            name_seq:   str,
            frame_range_scene_level: Tuple[int, int],
        ) -> Dict[str, np.ndarray]:
        label = self._full_label(name_scene, name_seq)
        s_loc, e_loc = self._seq_local_slice(label, frame_range_scene_level)
        pw = label.smpl_param_world
        return {
            'poses':  pw['poses'] [s_loc:e_loc],
            'shapes': pw['shapes'][s_loc:e_loc],
            'trans':  pw['trans'] [s_loc:e_loc],
        }

    @override
    def get_eval_meta(self) -> Eval_Meta:
        return WP_EVAL_META
