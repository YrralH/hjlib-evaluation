'''WorldPose eval meta. GT is a dense SMPL fit -- all 24 SMPL_24 slots are
trustworthy. Single 3D metric over the full SMPL_24 body subset.

T-MPJPE root: Pelvis_SMPL (SMPL_24 idx 0). WP has the real SMPL Pelvis joint from
the fit, so we anchor on it rather than synthesizing a hip midpoint. JTA, which has
no Pelvis annotation, uses the L/R_Hip midpoint instead -- the choice of root is
intentionally per-dataset, not cross-dataset uniform.

Port of monolith ``per_dataset/wp_eval_meta.py`` (SMPL_24 now from hjlib-skeleton).
'''
from __future__ import annotations

from hjlib_skeleton import SMPL_24

from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_3D


_PELVIS_SMPL = SMPL_24['Pelvis_SMPL']
_SMPL_24_ALL = tuple(range(24))


WP_EVAL_META = Eval_Meta(
    name_dataset    = 'worldpose_smpl',
    meta_version    = '2026-05-02_v1',
    unit_world      = 'm',
    k_rt_relation   = 'shared',
    metrics_3d      = (
        Metric_Spec_3D(
            name                                = 'SMPL_24_full',
            joint_indices_smpl_54               = _SMPL_24_ALL,
            root_indices_smpl_54_for_alignment  = (_PELVIS_SMPL,),
        ),
    ),
    metrics_2d_oks  = (),    # 2D OKS pipeline not plumbed yet
)
