'''Test-protocol eval contract: dataset-level spec consumed by the Tester (and any
external-method tester) so result tables stay apples-to-apples across methods.

Port of monolith ``test/protocol_dynamic/eval_meta.py`` (verbatim -- pure
dataclasses, no external deps).

Design tenets:
    - Dataset profile, not method profile. Methods are required to output dense
      SMPL_54 world-space joints (and, for OKS, a believed K/RT + bbox); the meta
      describes which SMPL_54 slots are trustworthy in this dataset's GT and how
      the metric is reduced.
    - No valid-mask, no per-slot NaN handling. Eval asserts non-NaN at the indices
      declared here. NaN is a provider / inference bug, not a metric branch.
    - All ambiguity (joint subset / root for alignment / 2D space / which K,RT to
      project through) is pinned in the data, not in eval-stage branches.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Metric_Spec_3D:
    '''3D joint metric (MPJPE + T-MPJPE) over a fixed SMPL_54 joint subset.

    joint_indices_smpl_54:
        SMPL_54 indices that are the metric subject. Must be non-NaN in both GT
        and pred at evaluation time (strict, no masking).

    root_indices_smpl_54_for_alignment:
        SMPL_54 indices whose per-frame mean is subtracted from both pred and GT
        before T-MPJPE. Independent of joint_indices_smpl_54 (does not have to be a
        subset). Per-dataset choice: SMPL-fit datasets (e.g. WP) use
        (Pelvis_SMPL,); raw-joint datasets without Pelvis annotation (e.g. JTA)
        fall back to the L/R_Hip midpoint via (L_Hip_SMPL, R_Hip_SMPL). T-MPJPE
        numbers are therefore not strictly cross-dataset comparable on the absolute
        scale -- the root anchor differs.
    '''
    name:                                str
    joint_indices_smpl_54:               Tuple[int, ...]
    root_indices_smpl_54_for_alignment:  Tuple[int, ...]


@dataclass(frozen=True)
class Metric_Spec_2D_OKS:
    '''2D OKS over projected joints. Both pred and GT 3D joints are projected to
    pixel space via the K, RT declared by k_rt_source; OKS is computed in pixel
    space using per-joint COCO-style sigma + bbox scale (bbox_source).

    Note: not implemented in stage_eval yet (requires K, RT believe + bbox to ride
    with the inference dump). Declared here so the meta schema is stable.
    '''
    name:                  str
    joint_indices_smpl_54: Tuple[int, ...]
    sigmas_per_joint:      Tuple[float, ...]    # same length as joint_indices_smpl_54
    k_rt_source:           str                  # 'gt' | 'believe'
    bbox_source:           str                  # 'gt_track' | 'believe'


@dataclass(frozen=True)
class Eval_Meta:
    '''Test-protocol eval contract for one dataset.

    name_dataset, meta_version:
        Identity. External testers should record the (name_dataset, meta_version)
        used so result tables can be safely cross-compared.

    unit_world:
        'm' if GT + pred world-space joints come back in meters (the network output
        convention), 'mm' if already millimeters. Eval scales to mm on output
        regardless.

    k_rt_relation:
        'shared' iff the method's believed K, RT must equal GT K, RT (current
        situation). 'free' if they may differ; in that case 2D metrics must
        explicitly declare which K, RT to project through.
    '''
    name_dataset:    str
    meta_version:    str
    unit_world:      str                            # 'm' | 'mm'
    k_rt_relation:   str                            # 'shared' | 'free'
    metrics_3d:      Tuple[Metric_Spec_3D,     ...]
    metrics_2d_oks:  Tuple[Metric_Spec_2D_OKS, ...]

    def __post_init__(self) -> None:
        assert self.unit_world    in ('m', 'mm'),         self.unit_world
        assert self.k_rt_relation in ('shared', 'free'),  self.k_rt_relation
        for m in self.metrics_3d:
            assert len(m.joint_indices_smpl_54)              >= 1, (self.name_dataset, m.name)
            assert len(m.root_indices_smpl_54_for_alignment) >= 1, (self.name_dataset, m.name)
        for m in self.metrics_2d_oks:
            assert len(m.joint_indices_smpl_54) == len(m.sigmas_per_joint), (
                self.name_dataset, m.name,
                len(m.joint_indices_smpl_54), len(m.sigmas_per_joint),
            )
            assert m.k_rt_source in ('gt', 'believe'),   m.k_rt_source
            assert m.bbox_source in ('gt_track', 'believe'), m.bbox_source
