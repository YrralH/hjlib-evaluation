'''Stateless eval reducer used by Tester.stage_eval and any external-method tester
whose dump conforms to the per-segment schema:

    {'segment': Test_Segment, 'pred': {'joints_54_world': (L, 54, 3), ...}}

The reducer iterates the testset's flat indices, loads the matching pred dump,
asserts non-NaN at the meta-declared indices, computes per-metric MPJPE + T-MPJPE +
absolute jitter, and prints one per-scene + ALL table per metric variant.

Port of monolith ``test/protocol_dynamic/eval_reducer.py``. Changes: the pred dump is
read via ``dump_reader.load_inference_dump`` (qualname-routing, so the monolith dumps
load without importing the monolith); fps comes from the assembly dataset registry;
``compute_jitter`` is inlined (was monolith ``evaluate.compute_metric.util_metric``).

NaN policy: strict raise. NaN at meta indices is a provider / inference bug (the
test-protocol contract requires dense GT + pred at those indices).
'''
from __future__ import annotations

import os.path as osp
from typing import Callable, Dict, Tuple

import numpy as np
from tqdm import tqdm

from hjlib_dataset_assembly.dataset_builder.dataset_registry import get_dataset_facts

from hjlib_evaluation.dump_reader import load_inference_dump
from hjlib_evaluation.eval_meta import Eval_Meta, Metric_Spec_3D
from hjlib_evaluation.gt_provider_base import GT_Provider_Base
from hjlib_evaluation.test_segment import Test_Segment
from hjlib_evaluation.testset import TestSet


Path_Pkl_For_Segment = Callable[[str, Test_Segment], str]


def compute_jitter(joints_3d: np.ndarray, fps: float, divide_by_10: bool = False) -> float:
    '''Average finite-difference jerk magnitude (absolute smoothness; does NOT
    compare against GT). Inputs (T, J, 3) in meters, world-frame, NO NaN, one
    continuously-valid run. Matches GVHMR / GENMO / WHAM convention. NaN if T < 4.

    Port of monolith ``util_metric.compute_JITTER``.'''
    T = joints_3d.shape[0]
    J = joints_3d.shape[1]
    assert joints_3d.shape == (T, J, 3), joints_3d.shape
    assert fps > 0, fps
    if T < 4:
        return float('nan')
    dt = 1.0 / float(fps)
    jerk = (
        joints_3d[3:]
        - 3.0 * joints_3d[2:-1]
        + 3.0 * joints_3d[1:-2]
        - joints_3d[:-3]
    ) / (dt ** 3)
    jerk_norm = np.linalg.norm(jerk, axis=2)  # (T-3, J)
    avg = float(jerk_norm.mean())
    return avg / 10.0 if divide_by_10 else avg


# --- public entry --------------------------------------------------------

def eval_dumps_against_gt(
        testset:       TestSet,
        gt_provider:   GT_Provider_Base,
        path_dump_dir: str,
        path_pkl_for_segment: Path_Pkl_For_Segment,
    ) -> None:
    '''Reduce per-segment pred dumps vs GT into per-scene + ALL frame-weighted
    MPJPE / T-MPJPE tables, one table per metric variant in eval meta.

    path_pkl_for_segment(path_dump_dir, seg) -> filesystem path of the dump pkl.
    Injected to keep the reducer agnostic of the Tester's filename scheme.'''
    meta = gt_provider.get_eval_meta()
    scale_mm = {'m': 1000.0, 'mm': 1.0}[meta.unit_world]
    fps = get_dataset_facts(meta.name_dataset).fps

    # buckets[metric_name][scene] tracks frame-weighted MPJPE/T-MPJPE plus
    # Jitter-Ratio accumulators (accumulate numerator / denominator separately
    # per (T-3)*J jerk windows, divide once).
    buckets: Dict[str, Dict[str, Dict[str, float]]] = {m.name: {} for m in meta.metrics_3d}

    for flat_idx in tqdm(range(len(testset)), desc='eval'):
        seg = testset.get_test_segment(flat_idx)
        pred_joints, gt_j = _load_one_segment(
            gt_provider, path_dump_dir, path_pkl_for_segment, flat_idx, seg)

        for metric in meta.metrics_3d:
            mpjpe_mm, tmpjpe_mm, j_pred, j_gt, n_windows = _compute_one_metric(
                metric, gt_j, pred_joints, scale_mm, fps, seg)
            num_frames = pred_joints.shape[0]
            bucket = buckets[metric.name].setdefault(seg.name_scene, {
                'sum_mpjpe_frames':  0.0, 'sum_tmpjpe_frames': 0.0, 'frames': 0.0,
                'jitter_pred_num':   0.0, 'jitter_gt_den':     0.0, 'jitter_n_windows': 0.0,
            })
            bucket['sum_mpjpe_frames']  += mpjpe_mm  * num_frames
            bucket['sum_tmpjpe_frames'] += tmpjpe_mm * num_frames
            bucket['frames']            += num_frames
            if n_windows > 0:
                bucket['jitter_pred_num']  += j_pred * n_windows
                bucket['jitter_gt_den']    += j_gt   * n_windows
                bucket['jitter_n_windows'] += n_windows

    _print_buckets(meta, buckets, fps)


# --- per-segment IO + per-metric reduction (stateless) -------------------

def _load_one_segment(
        gt_provider:          GT_Provider_Base,
        path_dump_dir:        str,
        path_pkl_for_segment: Path_Pkl_For_Segment,
        flat_idx:             int,
        seg:                  Test_Segment,
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''Load pred dump, assert segment match, fetch GT joints, assert shape.'''
    path_pkl = path_pkl_for_segment(path_dump_dir, seg)
    assert osp.isfile(path_pkl), (
        'eval reducer: missing inference pkl for flat_idx=%d (%s). Expected: %s'
        % (flat_idx, seg.to_str(), path_pkl))
    seg_on_disk, pred = load_inference_dump(path_pkl)
    assert seg_on_disk == seg, (
        'eval reducer: pkl at %s describes a different segment than TestSet[flat_idx=%d]. '
        'Likely a stale inference dump. Disk: %s  vs  TestSet: %s'
        % (path_pkl, flat_idx, seg_on_disk.to_str(), seg.to_str()))

    pred_joints: np.ndarray = pred['joints_54_world']
    gt_joints: np.ndarray = gt_provider.get_smpl_joints_54_world(
        seg.name_scene, seg.name_seq,
        (seg.index_frame_original_start, seg.index_frame_original_end),
    )
    assert pred_joints.shape == gt_joints.shape, (pred_joints.shape, gt_joints.shape, seg.to_str())
    return pred_joints, gt_joints


def _compute_one_metric(
        metric:      Metric_Spec_3D,
        gt_joints:   np.ndarray,                       # (L, 54, 3)
        pred_joints: np.ndarray,                       # (L, 54, 3)
        scale_mm:    float,
        fps:         float,
        seg:         Test_Segment,
    ) -> Tuple[float, float, float, float, int]:
    '''Strict NaN assertion + MPJPE / T-MPJPE / absolute jitter (pred & GT).

    T-MPJPE recipe: subtract per-frame mean over root_indices from both pred and GT,
    then MPJPE on the joint subset. Jitter: absolute m/s^3 jerk over the joint subset
    (single continuously-valid run). Returns (j_pred, j_gt, n_windows=(T-3)*J).'''
    joint_idx = list(metric.joint_indices_smpl_54)
    root_idx = list(metric.root_indices_smpl_54_for_alignment)

    gt_sub = gt_joints[:, joint_idx, :]
    pred_sub = pred_joints[:, joint_idx, :]
    gt_root = gt_joints[:, root_idx, :].mean(axis=1, keepdims=True)
    pred_root = pred_joints[:, root_idx, :].mean(axis=1, keepdims=True)

    _assert_no_nan(gt_sub,    'GT',   metric.name, 'joint_indices', seg)
    _assert_no_nan(pred_sub,  'pred', metric.name, 'joint_indices', seg)
    _assert_no_nan(gt_root,   'GT',   metric.name, 'root_indices',  seg)
    _assert_no_nan(pred_root, 'pred', metric.name, 'root_indices',  seg)

    mpjpe_mm = float(np.linalg.norm(gt_sub - pred_sub, axis=2).mean()) * scale_mm
    tmpjpe_mm = float(np.linalg.norm(
        (gt_sub - gt_root) - (pred_sub - pred_root), axis=2,
    ).mean()) * scale_mm

    pred_sub_m = pred_sub if scale_mm == 1000.0 else pred_sub / 1000.0
    gt_sub_m = gt_sub if scale_mm == 1000.0 else gt_sub / 1000.0
    T = pred_sub_m.shape[0]
    J = pred_sub_m.shape[1]
    if T >= 4:
        j_pred = compute_jitter(pred_sub_m, fps=fps, divide_by_10=False)
        j_gt = compute_jitter(gt_sub_m, fps=fps, divide_by_10=False)
        n_windows = (T - 3) * J
    else:
        j_pred = float('nan')
        j_gt = float('nan')
        n_windows = 0
    return mpjpe_mm, tmpjpe_mm, j_pred, j_gt, n_windows


def _assert_no_nan(arr: np.ndarray, side: str, metric_name: str, field: str, seg: Test_Segment) -> None:
    if not np.isnan(arr).any():
        return
    raise AssertionError(
        'NaN in %s %s for metric %r at scene=%s seq=%s frames=[%d,%d). The test-protocol '
        'contract requires GT + pred dense at the meta-declared indices; fix the provider '
        'or the inference dump.'
        % (side, field, metric_name, seg.name_scene, seg.name_seq,
           seg.index_frame_original_start, seg.index_frame_original_end))


# --- output formatting ---------------------------------------------------

def _print_buckets(meta: Eval_Meta, buckets: Dict[str, Dict[str, Dict[str, float]]], fps: float) -> None:
    for metric in meta.metrics_3d:
        print()
        print('=== stage_eval[%s] (%s, meta=%s, fps=%.1f): per-scene frame-weighted metrics ==='
              % (metric.name, meta.name_dataset, meta.meta_version, fps))
        print('%-22s  %10s  %11s  %12s  %12s  %12s  %9s' %
              ('scene', 'MPJPE(mm)', 'T-MPJPE(mm)', 'JitRatio', 'JPred(m/s^3)', 'JGT(m/s^3)', 'frames'))
        per_scene = buckets[metric.name]
        total_m = total_t = total_f = total_jp = total_jg = total_jw = 0.0
        for scene in sorted(per_scene.keys()):
            st = per_scene[scene]
            m = st['sum_mpjpe_frames'] / st['frames']
            t = st['sum_tmpjpe_frames'] / st['frames']
            jw = st['jitter_n_windows']
            if jw > 0:
                jp_avg = st['jitter_pred_num'] / jw
                jg_avg = st['jitter_gt_den'] / jw
                ratio = (st['jitter_pred_num'] / st['jitter_gt_den']) if st['jitter_gt_den'] > 0 else float('nan')
            else:
                jp_avg = jg_avg = ratio = float('nan')
            print('%-22s  %10.2f  %11.2f  %12.3f  %12.3f  %12.3f  %9d' %
                  (scene, m, t, ratio, jp_avg, jg_avg, int(st['frames'])))
            total_m  += st['sum_mpjpe_frames']
            total_t  += st['sum_tmpjpe_frames']
            total_f  += st['frames']
            total_jp += st['jitter_pred_num']
            total_jg += st['jitter_gt_den']
            total_jw += st['jitter_n_windows']
        denom = max(total_f, 1.0)
        if total_jw > 0:
            all_jp = total_jp / total_jw
            all_jg = total_jg / total_jw
            all_r = (total_jp / total_jg) if total_jg > 0 else float('nan')
        else:
            all_jp = all_jg = all_r = float('nan')
        print('%-22s  %10.2f  %11.2f  %12.3f  %12.3f  %12.3f  %9d' %
              ('ALL', total_m / denom, total_t / denom, all_r, all_jp, all_jg, int(total_f)))
