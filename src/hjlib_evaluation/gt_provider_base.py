'''Abstract per-dataset GT provider for evaluation.

Port of monolith ``test/protocol_dynamic/gt_provider_base.py``, scoped to the
metric baseline (MPJPE / T-MPJPE): the world-space SMPL joints / params are the
abstract surface concrete providers must implement. The camera K/RT + ground +
scene video-streamer methods (for 2D OKS -- not implemented in stage_eval -- and
reprojection vis -- out of scope) are kept on the interface but DEFAULT to raising
NotImplementedError; they will be wired (via hjlib-dataset-std) when 2D OKS / vis
are taken up. See docs/design/migration.md + EVAL Cross-lib TODO.

All outputs live in world space unless otherwise noted. Frame ranges are scene-level
absolute [start, end) as recorded in Test_Segment. Camera K/RT returned by
get_camera_K_RT_gt() are the GT camera (the method's believed K/RT rides with the
predictions, not through this interface).
'''
from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import numpy as np

from hjlib_evaluation.eval_meta import Eval_Meta


class GT_Provider_Base(abc.ABC):
    '''One implementation per dataset, supplying world-space GT for evaluation.'''

    name_dataset: str = ''

    @abc.abstractmethod
    def get_smpl_joints_54_world(
            self,
            name_scene: str,
            name_seq:   str,
            frame_range_scene_level: Tuple[int, int],
        ) -> np.ndarray:
        '''@return (N, 54, 3) world-space SMPL-54 joints for (scene, name_seq) over
        [start, end). Sparse providers (e.g. JTA) populate only the metric-relevant
        slots and leave the rest NaN; consumers must read only the indices declared
        in get_eval_meta().'''
        raise NotImplementedError

    @abc.abstractmethod
    def get_smpl_param_world(
            self,
            name_scene: str,
            name_seq:   str,
            frame_range_scene_level: Tuple[int, int],
        ) -> Dict[str, np.ndarray]:
        '''@return dict with 'poses' (N, 72), 'shapes' (N, 10), 'trans' (N, 3) in
        world space. Providers without a SMPL fit (e.g. JTA) raise
        NotImplementedError.'''
        raise NotImplementedError

    @abc.abstractmethod
    def get_eval_meta(self) -> Eval_Meta:
        '''@return the test-protocol Eval_Meta for this dataset (consumed by the
        Tester + any external-method tester so result tables stay apples-to-apples).
        See eval_meta.py.'''
        raise NotImplementedError

    # --- deferred: camera / ground / video (2D OKS + reprojection vis) ---------
    # Kept on the interface for shape-stability; raise until 2D OKS / vis are in
    # scope (then wired via hjlib-dataset-std). See module docstring.

    def get_camera_K_RT_gt(
            self,
            name_scene: str,
            frame_range_scene_level: Tuple[int, int],
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''@return (K (N, 3, 3), RT_world_to_camera (N, 4, 4)). Deferred.'''
        raise NotImplementedError(
            'get_camera_K_RT_gt is deferred (2D OKS not in stage_eval scope yet); '
            'wire via hjlib-dataset-std when 2D OKS / vis are taken up')

    def get_ground_param_world(self, name_scene: str) -> np.ndarray:
        '''@return (4,) world-space ground plane [a, b, c, d]. Deferred.'''
        raise NotImplementedError(
            'get_ground_param_world is deferred (used by vis; out of scope)')

    def get_scene_video_streamer(self, name_scene: str) -> Any:
        '''@return a random-access undistorted-frame streamer. Deferred (vis only).'''
        raise NotImplementedError(
            'get_scene_video_streamer is deferred (reprojection vis; out of scope)')

    def get_visibility_gt(
            self,
            name_scene: str,
            id_person:  int,
            frame_range_scene_level: Tuple[int, int],
        ) -> Optional[np.ndarray]:
        '''Optional: (N,) bool, True = person visibly annotated that frame. Default
        None -- datasets without visibility labels (or while deferred) return None.'''
        return None
