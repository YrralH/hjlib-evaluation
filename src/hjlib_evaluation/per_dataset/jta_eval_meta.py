'''JTA / JTA_Ext eval meta. GT is raw 22-joint annotation. Only limb endpoints
have reliable SMPL semantic correspondence (neck / collars / spine / head are
defined differently between JTA-22 and SMPL_24, so they are excluded).

Two MPJPE variants exposed in parallel:
    limb_endpoints_with_hip:  12 joints (incl. L/R_Hip_SMPL)
    limb_endpoints_no_hip:    10 joints (drop L/R_Hip_SMPL)

T-MPJPE root: (L_Hip_SMPL + R_Hip_SMPL) / 2 in both variants. JTA has no SMPL
Pelvis joint; the hip midpoint stands in. Note: the no-hip variant still uses the
hip midpoint as alignment anchor (alignment recipe is independent of the joint
subset on which the residual is measured).

Port of monolith ``per_dataset/jta_eval_meta.py`` (SMPL_24 now from hjlib-skeleton).
'''
from __future__ import annotations

from typing import Tuple

from hjlib_skeleton import SMPL_24

from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_3D


def _smpl24(*names: str) -> Tuple[int, ...]:
    return tuple(SMPL_24[n] for n in names)


_LIMB_ENDPOINTS_NO_HIP = _smpl24(
    'L_Shoulder', 'R_Shoulder',
    'L_Elbow',    'R_Elbow',
    'L_Wrist',    'R_Wrist',
    'L_Knee',     'R_Knee',
    'L_Ankle',    'R_Ankle',
)
_LIMB_ENDPOINTS_WITH_HIP = _LIMB_ENDPOINTS_NO_HIP + _smpl24('L_Hip_SMPL', 'R_Hip_SMPL')
_ROOT_HIP_PAIR           = _smpl24('L_Hip_SMPL', 'R_Hip_SMPL')


def _build_jta_meta(name_dataset: str) -> Eval_Meta:
    return Eval_Meta(
        name_dataset    = name_dataset,
        meta_version    = '2026-05-02_v1',
        unit_world      = 'm',
        k_rt_relation   = 'shared',
        metrics_3d      = (
            Metric_Spec_3D(
                name                                = 'limb_endpoints_with_hip',
                joint_indices_smpl_54               = _LIMB_ENDPOINTS_WITH_HIP,
                root_indices_smpl_54_for_alignment  = _ROOT_HIP_PAIR,
            ),
            Metric_Spec_3D(
                name                                = 'limb_endpoints_no_hip',
                joint_indices_smpl_54               = _LIMB_ENDPOINTS_NO_HIP,
                root_indices_smpl_54_for_alignment  = _ROOT_HIP_PAIR,
            ),
        ),
        metrics_2d_oks  = (),
    )


JTA_EVAL_META     = _build_jta_meta('jta_smpl_fitted')
JTA_EXT_EVAL_META = _build_jta_meta('jta_ext_smpl_fitted')
