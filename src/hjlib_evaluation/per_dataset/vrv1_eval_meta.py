'''Preserved VRv1 fitted-SMPL metric identity for disjoint camera families.'''
from dataclasses import replace

from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_3D


VRV1_EVAL_META = Eval_Meta(
    name_dataset='vrv1_smpl_fitted',
    meta_version='2026-05-20_v1',
    unit_world='m',
    k_rt_relation='shared',
    metrics_3d=(Metric_Spec_3D(
        name='SMPL_24_full',
        joint_indices_smpl_54=tuple(range(24)),
        root_indices_smpl_54_for_alignment=(0,),
    ),),
    metrics_2d_oks=(),
)
VRV1_GOPRO_EVAL_META = replace(
    VRV1_EVAL_META, name_dataset='vrv1_gopro_smpl_fitted')
