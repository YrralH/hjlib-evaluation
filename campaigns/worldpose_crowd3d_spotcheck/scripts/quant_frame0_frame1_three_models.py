'''Quantitative frame0/frame1 comparison for Crowd3D, abla_v1, and v18agg.

The script first reports monolith metrics on the people rendered in
``artifacts/overlays/{abla_v1,v18agg}``, then matches Crowd3D to GT with the
2D-Hungarian matcher, and finally restricts all three methods to the common GT
person subset that has predictions from every method.
'''

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
OUT_ROOT = ARTIFACT_ROOT / 'frame01_quant'
CROWD_MATCH_ROOT = OUT_ROOT / 'crowd3d_matching'
OUT_JSON = OUT_ROOT / 'metrics.json'
OUT_MD = OUT_ROOT / 'metrics.md'

SCENE = 'NET_ARG_231908'
FRAME_INDICES = (0, 1)
JOINT_INDICES = tuple(range(24))
ROOT_INDICES = (0,)


def ensure_import_paths() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    for rel in (
        'hjlib-dataset-std/src',
        'hjlib-dataset-raw/src',
        'hjlib-camera/src',
        'hjlib-streamer/src',
        'hjlib-smpl/src',
        'hjlib-vis-2d/src',
        'hjlib-geometry/src',
        'hjlib-cache/src',
        'hjlib-evaluation/src',
    ):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def mpjpe_tmpjpe_mm(pred_joints: np.ndarray, gt_joints: np.ndarray) -> Tuple[float, float]:
    pred_sub = np.asarray(pred_joints[list(JOINT_INDICES)], dtype=np.float64)
    gt_sub = np.asarray(gt_joints[list(JOINT_INDICES)], dtype=np.float64)
    pred_root = np.asarray(pred_joints[list(ROOT_INDICES)], dtype=np.float64).mean(axis=0, keepdims=True)
    gt_root = np.asarray(gt_joints[list(ROOT_INDICES)], dtype=np.float64).mean(axis=0, keepdims=True)
    assert pred_sub.shape == gt_sub.shape == (len(JOINT_INDICES), 3), (pred_sub.shape, gt_sub.shape)
    assert np.isfinite(pred_sub).all(), 'non-finite pred joints'
    assert np.isfinite(gt_sub).all(), 'non-finite GT joints'
    mpjpe = float(np.linalg.norm(pred_sub - gt_sub, axis=1).mean()) * 1000.0
    tmpjpe = float(np.linalg.norm(
        (pred_sub - pred_root) - (gt_sub - gt_root),
        axis=1,
    ).mean()) * 1000.0
    return mpjpe, tmpjpe


def summarize_rows(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    if not rows:
        return {
            'label': label,
            'n_person_frames': 0,
            'mpjpe_mm': float('nan'),
            'tmpjpe_mm': float('nan'),
        }
    return {
        'label': label,
        'n_person_frames': len(rows),
        'mpjpe_mm': float(np.mean([float(row['mpjpe_mm']) for row in rows])),
        'tmpjpe_mm': float(np.mean([float(row['tmpjpe_mm']) for row in rows])),
    }


def collect_gt_world_by_person_frame(ds: object) -> Dict[Tuple[int, int], np.ndarray]:
    _num_person, _num_frame, per_person = ds.forward_per_person_joints_by_name_scene(SCENE)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for id_person, frame_indices, joints_seq in per_person:
        for local_idx, frame_index in enumerate(frame_indices):
            frame = int(frame_index)
            if frame not in FRAME_INDICES:
                continue
            out[(int(id_person), frame)] = np.asarray(joints_seq[local_idx], dtype=np.float64)
    return out


def gt_camera_for_frame(
        ds: object,
        gt_world: Dict[Tuple[int, int], np.ndarray],
    ) -> Dict[Tuple[int, int], np.ndarray]:
    from hjlib_dataset_std.vis.overlay_2d import sub_camera_one_frame

    camera = ds.get_camera_by_name_scene(SCENE)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for frame_index in FRAME_INDICES:
        sub_camera = sub_camera_one_frame(camera, frame_index)
        for key, joints_world in gt_world.items():
            id_person, frame = key
            if frame != frame_index:
                continue
            out[(id_person, frame)] = sub_camera.extrinsics_batch.world_to_camera_points(
                joints_world[None])[0]
    return out


def make_candidate(scene: str, frame_index: int) -> object:
    from select_worldpose_frames import Frame_Candidate, METHOD_DUMP_DIRS, load_segments

    segments = load_segments(METHOD_DUMP_DIRS['abla_v1'])
    covering = [
        seg for seg in segments
        if seg.scene == scene and seg.frame_start <= frame_index < seg.frame_end
    ]
    assert covering, 'no abla_v1 segments cover %s frame %d' % (scene, frame_index)
    return Frame_Candidate(
        scene=scene,
        frame_index=frame_index,
        num_covering_segments=len(covering),
        person_ids=sorted({seg.id_person for seg in covering}),
        segment_files=[Path(seg.path_pkl).name for seg in covering],
    )


def monolith_predictions_by_method() -> Dict[str, Dict[Tuple[int, int], np.ndarray]]:
    from select_worldpose_frames import METHOD_DUMP_DIRS, load_prediction_joints_for_candidate

    out: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {}
    for method in ('abla_v1', 'v18agg'):
        method_out: Dict[Tuple[int, int], np.ndarray] = {}
        for frame_index in FRAME_INDICES:
            candidate = make_candidate(SCENE, frame_index)
            pred_by_person = load_prediction_joints_for_candidate(
                candidate,
                METHOD_DUMP_DIRS[method],
            )
            for id_person, joints in pred_by_person.items():
                method_out[(int(id_person), frame_index)] = np.asarray(joints, dtype=np.float64)
        out[method] = method_out
    return out


def rows_for_method(
        method: str,
        predictions: Dict[Tuple[int, int], np.ndarray],
        gt_world: Dict[Tuple[int, int], np.ndarray],
        subset: Iterable[Tuple[int, int]] | None = None,
    ) -> List[Dict[str, Any]]:
    keys = sorted(predictions.keys()) if subset is None else sorted(subset)
    rows: List[Dict[str, Any]] = []
    for id_person, frame_index in keys:
        key = (int(id_person), int(frame_index))
        if key not in predictions or key not in gt_world:
            continue
        mpjpe, tmpjpe = mpjpe_tmpjpe_mm(predictions[key], gt_world[key])
        rows.append({
            'method': method,
            'frame': '%s_frame%d' % (SCENE, frame_index),
            'frame_index': frame_index,
            'gt_id': id_person,
            'mpjpe_mm': mpjpe,
            'tmpjpe_mm': tmpjpe,
        })
    return rows


def run_crowd3d_matching() -> Dict[str, Any]:
    from match_crowd3d_to_gt_2d import (
        DEFAULT_PACKAGE_ROOT,
        DEFAULT_WORLDPOSE_ROOT,
        DEFAULT_WORLDPOSE_UNDISTORT_FRAMES,
        compute_matches,
    )

    CROWD_MATCH_ROOT.mkdir(parents=True, exist_ok=True)
    summary = compute_matches(
        package_root=DEFAULT_PACKAGE_ROOT,
        output_root=CROWD_MATCH_ROOT,
        frame_start=0,
        frame_end=2,
        max_cost_px=90.0,
        trim_ratio=0.8,
        center_weight=0.25,
        worldpose_root=DEFAULT_WORLDPOSE_ROOT,
        worldpose_undistort_frames=DEFAULT_WORLDPOSE_UNDISTORT_FRAMES,
    )
    (CROWD_MATCH_ROOT / 'matches.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return summary


def crowd3d_predictions_for_matches(
        match_summary: Dict[str, Any],
    ) -> Dict[Tuple[int, int], np.ndarray]:
    from match_crowd3d_to_gt_2d import DEFAULT_PACKAGE_ROOT

    out: Dict[Tuple[int, int], np.ndarray] = {}
    for frame in match_summary['frames']:
        frame_index = int(frame['frame_index'])
        path_npz = (
            DEFAULT_PACKAGE_ROOT
            / 'results'
            / 'joints'
            / ('%s_frame%d_joints.npz' % (SCENE, frame_index))
        )
        data = np.load(path_npz)
        joints_camera = np.asarray(data['smpl_joints_54_camera_m'], dtype=np.float64)
        for match in frame['matches']:
            gt_id = int(match['gt_id'])
            crowd_index = int(match['crowd_index'])
            out[(gt_id, frame_index)] = joints_camera[crowd_index]
    return out


def build_common_subset(
        crowd_pred: Dict[Tuple[int, int], np.ndarray],
        monolith_pred: Dict[str, Dict[Tuple[int, int], np.ndarray]],
    ) -> List[Tuple[int, int]]:
    keys = set(crowd_pred.keys())
    for method in ('abla_v1', 'v18agg'):
        keys &= set(monolith_pred[method].keys())
    return sorted(keys)


def write_markdown(result: Dict[str, Any], path_md: Path) -> None:
    lines = [
        '# Frame0/Frame1 Quantitative Reference',
        '',
        'Frames: `NET_ARG_231908_frame0`, `NET_ARG_231908_frame1`.',
        'Metrics are SMPL-24 MPJPE / T-MPJPE in millimeters.',
        '',
        '## Monolith Overlay Scope',
        '',
        '| method | person-frames | MPJPE mm | T-MPJPE mm |',
        '| --- | ---: | ---: | ---: |',
    ]
    for row in result['monolith_overlay_scope']['summaries']:
        lines.append(
            '| `%s` | %d | %.2f | %.2f |'
            % (
                row['label'],
                int(row['n_person_frames']),
                float(row['mpjpe_mm']),
                float(row['tmpjpe_mm']),
            )
        )

    lines.extend([
        '',
        '## Crowd3D Matched Scope',
        '',
        '| method | matched person-frames | MPJPE mm | T-MPJPE mm |',
        '| --- | ---: | ---: | ---: |',
    ])
    crowd_summary = result['crowd3d_matched_scope']['summary']
    lines.append(
        '| `crowd3d` | %d | %.2f | %.2f |'
        % (
            int(crowd_summary['n_person_frames']),
            float(crowd_summary['mpjpe_mm']),
            float(crowd_summary['tmpjpe_mm']),
        )
    )

    lines.extend([
        '',
        'Matching: accepted `%d` of GT `%d`; Crowd3D detections `%d`.'
        % (
            int(result['crowd3d_matching']['total_accepted']),
            int(result['crowd3d_matching']['total_gt']),
            int(result['crowd3d_matching']['total_crowd']),
        ),
        '',
        '## Common All-Three Subset',
        '',
        '| method | person-frames | MPJPE mm | T-MPJPE mm |',
        '| --- | ---: | ---: | ---: |',
    ])
    for row in result['common_all_three_scope']['summaries']:
        lines.append(
            '| `%s` | %d | %.2f | %.2f |'
            % (
                row['label'],
                int(row['n_person_frames']),
                float(row['mpjpe_mm']),
                float(row['tmpjpe_mm']),
            )
        )
    lines.extend([
        '',
        'Common GT person/frame keys:',
        '',
        '`%s`' % ', '.join(
            '%s:%d' % (key['frame'], int(key['gt_id']))
            for key in result['common_all_three_scope']['keys']
        ),
        '',
    ])
    path_md.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    ensure_import_paths()
    from hjlib_dataset_std.datasets.worldpose.worldpose import WorldPose_Std
    from match_crowd3d_to_gt_2d import DEFAULT_WORLDPOSE_ROOT, DEFAULT_WORLDPOSE_UNDISTORT_FRAMES

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds = WorldPose_Std(
        path_data_root=str(DEFAULT_WORLDPOSE_ROOT),
        path_undistort_frames=str(DEFAULT_WORLDPOSE_UNDISTORT_FRAMES),
    )
    gt_world = collect_gt_world_by_person_frame(ds)
    gt_camera = gt_camera_for_frame(ds, gt_world)

    monolith_pred = monolith_predictions_by_method()
    monolith_rows = {
        method: rows_for_method(method, monolith_pred[method], gt_world)
        for method in ('abla_v1', 'v18agg')
    }

    match_summary = run_crowd3d_matching()
    crowd_pred = crowd3d_predictions_for_matches(match_summary)
    crowd_rows = rows_for_method('crowd3d', crowd_pred, gt_camera)

    common_keys = build_common_subset(crowd_pred, monolith_pred)
    common_rows = {
        'crowd3d': rows_for_method('crowd3d', crowd_pred, gt_camera, common_keys),
        'abla_v1': rows_for_method('abla_v1', monolith_pred['abla_v1'], gt_world, common_keys),
        'v18agg': rows_for_method('v18agg', monolith_pred['v18agg'], gt_world, common_keys),
    }

    result = {
        'frames': ['%s_frame%d' % (SCENE, frame_index) for frame_index in FRAME_INDICES],
        'metric': 'SMPL_24_full',
        'unit': 'mm',
        'monolith_overlay_scope': {
            'summaries': [
                summarize_rows(monolith_rows['abla_v1'], 'abla_v1'),
                summarize_rows(monolith_rows['v18agg'], 'v18agg'),
            ],
            'rows': monolith_rows,
        },
        'crowd3d_matching': {
            'total_gt': match_summary['total_gt'],
            'total_crowd': match_summary['total_crowd'],
            'total_accepted': match_summary['total_accepted'],
            'mean_accepted_cost_px': match_summary['mean_accepted_cost_px'],
            'median_accepted_cost_px': match_summary['median_accepted_cost_px'],
            'matches_json': str(CROWD_MATCH_ROOT / 'matches.json'),
            'contact_sheet': match_summary['contact_sheet'],
            'overlay_root': match_summary['overlay_root'],
        },
        'crowd3d_matched_scope': {
            'summary': summarize_rows(crowd_rows, 'crowd3d'),
            'rows': crowd_rows,
        },
        'common_all_three_scope': {
            'keys': [
                {
                    'frame': '%s_frame%d' % (SCENE, frame_index),
                    'frame_index': frame_index,
                    'gt_id': id_person,
                }
                for id_person, frame_index in common_keys
            ],
            'summaries': [
                summarize_rows(common_rows['crowd3d'], 'crowd3d'),
                summarize_rows(common_rows['abla_v1'], 'abla_v1'),
                summarize_rows(common_rows['v18agg'], 'v18agg'),
            ],
            'rows': common_rows,
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    write_markdown(result, OUT_MD)

    print('monolith overlay scope')
    for row in result['monolith_overlay_scope']['summaries']:
        print('%s n=%d MPJPE=%.2f T-MPJPE=%.2f' % (
            row['label'], int(row['n_person_frames']),
            float(row['mpjpe_mm']), float(row['tmpjpe_mm'])))
    print('crowd3d matched scope')
    row = result['crowd3d_matched_scope']['summary']
    print('%s n=%d MPJPE=%.2f T-MPJPE=%.2f' % (
        row['label'], int(row['n_person_frames']),
        float(row['mpjpe_mm']), float(row['tmpjpe_mm'])))
    print('common all-three scope')
    for row in result['common_all_three_scope']['summaries']:
        print('%s n=%d MPJPE=%.2f T-MPJPE=%.2f' % (
            row['label'], int(row['n_person_frames']),
            float(row['mpjpe_mm']), float(row['tmpjpe_mm'])))
    print('json: %s' % OUT_JSON)
    print('md: %s' % OUT_MD)


if __name__ == '__main__':
    main()
