'''Compute MPJPE / T-MPJPE on the selected WorldPose spot-check frames.

Campaign-local script. It reuses the selected frame list produced by
``select_worldpose_frames.py`` and evaluates the same person segments for the
configured monolith method dumps.
'''

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
DEFAULT_SELECTION_PATH = ARTIFACT_ROOT / 'selected_worldpose_frames.json'
DEFAULT_OUTPUT_JSON = ARTIFACT_ROOT / 'spotcheck_metrics.json'
DEFAULT_OUTPUT_MD = ARTIFACT_ROOT / 'spotcheck_metrics.md'
DEFAULT_LABEL_DUMP_ROOT = Path('/data2/hj/Data_Process/__As_Single_Bbox_hjlib__')
DEFAULT_PRED_JOINTS_KEY = 'joints_54_world'


def ensure_import_paths() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    for rel in (
        'hjlib-evaluation/src',
        'hjlib-dataset-assembly/src',
        'hjlib-skeleton/src',
    ):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


@dataclass(frozen=True)
class Person_Metric_Row:
    method: str
    scene: str
    frame_index: int
    id_person: int
    segment_file: str
    mpjpe_mm: float
    tmpjpe_mm: float


@dataclass(frozen=True)
class Method_Summary:
    method: str
    pred_joints_key: str
    num_person_frames: int
    mpjpe_mm: float
    tmpjpe_mm: float


def load_selected_frames(path_selection: Path) -> List[Dict[str, Any]]:
    data = json.loads(path_selection.read_text(encoding='utf-8'))
    assert isinstance(data, list), type(data)
    return data


def maybe_limit_selected_frames(
        selected_frames: List[Dict[str, Any]],
        max_frames: int | None,
    ) -> List[Dict[str, Any]]:
    if max_frames is None:
        return selected_frames
    assert max_frames >= 1, max_frames
    return selected_frames[:max_frames]


def compute_mpjpe_tmpjpe_mm(
        pred_joints_54_world: np.ndarray,
        gt_joints_54_world: np.ndarray,
        joint_indices: Sequence[int],
        root_indices: Sequence[int],
    ) -> tuple[float, float]:
    pred_sub = pred_joints_54_world[list(joint_indices), :]
    gt_sub = gt_joints_54_world[list(joint_indices), :]
    pred_root = pred_joints_54_world[list(root_indices), :].mean(axis=0, keepdims=True)
    gt_root = gt_joints_54_world[list(root_indices), :].mean(axis=0, keepdims=True)

    assert pred_sub.shape == gt_sub.shape, (pred_sub.shape, gt_sub.shape)
    assert not np.isnan(pred_sub).any(), 'NaN in pred joint subset'
    assert not np.isnan(gt_sub).any(), 'NaN in GT joint subset'
    assert not np.isnan(pred_root).any(), 'NaN in pred root subset'
    assert not np.isnan(gt_root).any(), 'NaN in GT root subset'

    mpjpe_mm = float(np.linalg.norm(gt_sub - pred_sub, axis=1).mean()) * 1000.0
    tmpjpe_mm = float(np.linalg.norm(
        (gt_sub - gt_root) - (pred_sub - pred_root), axis=1,
    ).mean()) * 1000.0
    return mpjpe_mm, tmpjpe_mm


def compute_metrics(
        selected_frames: List[Dict[str, Any]],
        pred_joints_key: str,
        path_label_dump_root: Path,
        path_selection: Path,
    ) -> Dict[str, Any]:
    from hjlib_evaluation.dump_reader import load_inference_dump
    from hjlib_evaluation.get_by_dataset import get_gt_provider
    from hjlib_evaluation.per_dataset.wp_eval_meta import WP_EVAL_META
    from select_worldpose_frames import METHOD_DUMP_DIRS, parse_segment_filename

    gt_provider = get_gt_provider('worldpose_smpl', str(path_label_dump_root))
    metric = WP_EVAL_META.metrics_3d[0]
    rows: List[Person_Metric_Row] = []
    summaries: List[Method_Summary] = []

    for method, path_dump_dir in METHOD_DUMP_DIRS.items():
        method_rows: List[Person_Metric_Row] = []
        for frame in selected_frames:
            frame_index = int(frame['frame_index'])
            for segment_file in frame['segment_files']:
                path_pkl = path_dump_dir / str(segment_file)
                assert path_pkl.is_file(), 'missing pred dump: %s' % path_pkl
                seg_info = parse_segment_filename(path_pkl)
                offset = frame_index - seg_info.frame_start
                assert 0 <= offset < seg_info.frame_end - seg_info.frame_start, (
                    frame_index, seg_info)

                seg, pred = load_inference_dump(str(path_pkl))
                if pred_joints_key not in pred:
                    raise KeyError(
                        'pred field %r missing in %s. Available fields: %s'
                        % (pred_joints_key, path_pkl, sorted(pred.keys())))
                pred_joints = np.asarray(pred[pred_joints_key], dtype=np.float64)
                gt_joints = gt_provider.get_smpl_joints_54_world(
                    seg.name_scene,
                    seg.name_seq,
                    (seg.index_frame_original_start, seg.index_frame_original_end),
                ).astype(np.float64)
                assert pred_joints.shape == gt_joints.shape, (
                    pred_joints.shape, gt_joints.shape, path_pkl)

                mpjpe_mm, tmpjpe_mm = compute_mpjpe_tmpjpe_mm(
                    pred_joints[offset],
                    gt_joints[offset],
                    metric.joint_indices_smpl_54,
                    metric.root_indices_smpl_54_for_alignment,
                )
                method_rows.append(Person_Metric_Row(
                    method=method,
                    scene=str(frame['scene']),
                    frame_index=frame_index,
                    id_person=seg_info.id_person,
                    segment_file=str(segment_file),
                    mpjpe_mm=mpjpe_mm,
                    tmpjpe_mm=tmpjpe_mm,
                ))

        rows.extend(method_rows)
        summaries.append(Method_Summary(
            method=method,
            pred_joints_key=pred_joints_key,
            num_person_frames=len(method_rows),
            mpjpe_mm=float(np.mean([row.mpjpe_mm for row in method_rows])),
            tmpjpe_mm=float(np.mean([row.tmpjpe_mm for row in method_rows])),
        ))

    return {
        'metric_name': metric.name,
        'unit': 'mm',
        'label_dump_root': str(path_label_dump_root),
        'selection_path': str(path_selection),
        'num_selected_frames': len(selected_frames),
        'summaries': [asdict(row) for row in summaries],
        'per_person_frame': [asdict(row) for row in rows],
    }


def write_markdown(result: Dict[str, Any], path_md: Path, path_json: Path) -> None:
    lines = [
        '# Spotcheck Metrics',
        '',
        'Metric: `%s`, unit: `%s`' % (result['metric_name'], result['unit']),
        'Selected frames: `%d`' % int(result['num_selected_frames']),
        '',
        '| method | pred key | person-frames | MPJPE mm | T-MPJPE mm |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in result['summaries']:
        lines.append(
            '| `%s` | `%s` | %d | %.2f | %.2f |'
            % (
                row['method'],
                row['pred_joints_key'],
                int(row['num_person_frames']),
                float(row['mpjpe_mm']),
                float(row['tmpjpe_mm']),
            )
        )
    lines.extend([
        '',
        'Per-frame/person details are in `%s`.' % path_json.name,
        '',
    ])
    path_md.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument('--max-frames', type=int, default=None,
                        help='evaluate only the first N selected frames')
    parser.add_argument('--label-dump-root', type=Path, default=DEFAULT_LABEL_DUMP_ROOT)
    parser.add_argument('--pred-joints-key', default=DEFAULT_PRED_JOINTS_KEY)
    parser.add_argument('--out-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--out-md', type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    ensure_import_paths()
    selected_frames = load_selected_frames(args.selection)
    selected_frames = maybe_limit_selected_frames(selected_frames, args.max_frames)
    result = compute_metrics(
        selected_frames,
        args.pred_joints_key,
        args.label_dump_root,
        args.selection,
    )

    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    write_markdown(result, args.out_md, args.out_json)

    for row in result['summaries']:
        print('%s %s: n=%d MPJPE=%.2fmm T-MPJPE=%.2fmm' % (
            row['method'],
            row['pred_joints_key'],
            int(row['num_person_frames']),
            float(row['mpjpe_mm']),
            float(row['tmpjpe_mm']),
        ))
    print('json: %s' % args.out_json)
    print('md: %s' % args.out_md)


if __name__ == '__main__':
    main()
