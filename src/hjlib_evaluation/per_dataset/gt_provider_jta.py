'''JTA / JTA_Ext GT provider (raw 22-joint label, via hjlib-dataset-std).

Port of monolith ``per_dataset/gt_provider_jta.py``. JTA's GT is 22 raw joints (not
SMPL); we expose them via the GT_Provider interface by remapping into SMPL_54 slots
ONLY for the 12 limb endpoints whose semantics cross-match SMPL_24 (upper:
L/R_Shoulder, L/R_Elbow, L/R_Wrist; lower: L/R_Hip_SMPL, L/R_Knee, L/R_Ankle).
Neck / collars / spine / head are excluded -- their SMPL definition does not align
with the JTA-22 raw joints. All other SMPL_54 slots stay NaN (sparse provider); the
eval-meta declares which indices are read.

Source change vs monolith: the raw 22-joint kp3d comes from the std joints leaf
(``hjlib_dataset_std`` ``Joints_Only_Dynamic_Std.get_joints_3d_by_name_scene`` ->
``(P, F, 22, 3)``, camera space == world since JTA RT is identity) instead of the
monolith ``JTA_Raw_Label``. ``get_smpl_param_world`` raises (JTA has no SMPL fit).
'''
from __future__ import annotations

from typing import Dict, Tuple, override

import numpy as np

from hjlib_dataset_assembly.dataset_builder.label_manager import extract_name_seq_info
from hjlib_dataset_std import Joints_Only_Dynamic_Std

from hjlib_skeleton import SMPL_24, get_jta_22_with_smpl_names

from hjlib_evaluation.eval_meta import Eval_Meta
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.per_dataset.jta_eval_meta import JTA_EVAL_META, JTA_EXT_EVAL_META


N_JOINT_SMPL_54: int = 54

# Limb endpoints have SMPL-aligned semantics; neck / collars / spine / head do not.
_LIMB_ENDPOINT_SMPL_NAMES = frozenset({
    'L_Shoulder', 'R_Shoulder',
    'L_Elbow',    'R_Elbow',
    'L_Wrist',    'R_Wrist',
    'L_Hip_SMPL', 'R_Hip_SMPL',
    'L_Knee',     'R_Knee',
    'L_Ankle',    'R_Ankle',
})


def _build_jta22_to_smpl54_map() -> Dict[int, int]:
    '''{jta_22_idx -> smpl_54_idx} for the 12 limb-endpoint joints whose names
    cross-match SMPL_24. Other JTA names (head / neck / collars / spine) dropped.'''
    jta_fmt = get_jta_22_with_smpl_names(include_hips=True)
    out: Dict[int, int] = {}
    for jta_name, jta_idx in jta_fmt.items():
        if jta_name in _LIMB_ENDPOINT_SMPL_NAMES:
            out[int(jta_idx)] = int(SMPL_24[jta_name])
    assert len(out) == 12, len(out)
    return out


class _JTA_GT_Provider_Common(GT_Provider_Base):
    '''Shared implementation; JTA / JTA_Ext only differ by the std joints leaf +
    the eval meta.'''

    def __init__(self, std_joints: Joints_Only_Dynamic_Std) -> None:
        self._std = std_joints
        self._jta22_to_smpl54 = _build_jta22_to_smpl54_map()
        self._cache_kp3d: Dict[str, np.ndarray] = {}    # scene -> (P, F, 22, 3)

    def _kp3d_one_scene(self, name_scene: str) -> np.ndarray:
        if name_scene not in self._cache_kp3d:
            self._cache_kp3d[name_scene] = self._std.get_joints_3d_by_name_scene(name_scene).array
        return self._cache_kp3d[name_scene]

    @override
    def get_smpl_joints_54_world(
            self,
            name_scene: str,
            name_seq:   str,
            frame_range_scene_level: Tuple[int, int],
        ) -> np.ndarray:
        '''(L, 54, 3) world-space joints; only the 12 limb-endpoint SMPL slots are
        populated, the rest NaN. Consumers read only get_eval_meta() indices.'''
        s, e = frame_range_scene_level
        length = int(e - s)
        id_person, _ = extract_name_seq_info(name_seq)
        kp3d_scene = self._kp3d_one_scene(name_scene)        # (P, F, 22, 3)
        kp3d_seq = kp3d_scene[id_person, s:e, :, :]          # (L, 22, 3)
        out = np.full((length, N_JOINT_SMPL_54, 3), np.nan, dtype=kp3d_seq.dtype)
        for jta_idx, smpl_idx in self._jta22_to_smpl54.items():
            out[:, smpl_idx, :] = kp3d_seq[:, jta_idx, :]
        return out

    @override
    def get_smpl_param_world(
            self,
            name_scene: str,
            name_seq:   str,
            frame_range_scene_level: Tuple[int, int],
        ) -> Dict[str, np.ndarray]:
        raise NotImplementedError(
            'JTA has no SMPL fit. Use get_smpl_joints_54_world + the limb-endpoint '
            'metric (see get_eval_meta).')


class JTA_GT_Provider(_JTA_GT_Provider_Common):
    name_dataset = 'jta_smpl_fitted'

    @override
    def get_eval_meta(self) -> Eval_Meta:
        return JTA_EVAL_META


class JTA_Ext_GT_Provider(_JTA_GT_Provider_Common):
    name_dataset = 'jta_ext_smpl_fitted'

    @override
    def get_eval_meta(self) -> Eval_Meta:
        return JTA_EXT_EVAL_META
