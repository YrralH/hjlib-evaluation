'''Sequence-level WorldPose quant for monolith abla_v1 and v18agg dumps.

This campaign-local script evaluates every segment dump whose filename starts
with the requested WorldPose scene name. The aggregation matches the standard
protocol reducer's frame weighting: each segment frame contributes one
person-frame to the final MPJPE / T-MPJPE mean.
'''

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import typer


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
DEFAULT_LABEL_DUMP_ROOT = Path('/data2/hj/Data_Process/__As_Single_Bbox_hjlib__')
DEFAULT_PRED_JOINTS_KEY = 'joints_54_world'
DEFAULT_SCENE = 'NET_ARG_231908'
METHOD_DUMP_DIRS: Dict[str, Path] = {
    'abla_v1': Path(
        '/home/hj/Data_Process/protocol_dynamic/inference_dumps/worldpose/full/kp_rtmlib/'
        'ablation_a00_hvip_hipmid_K1_ep0004'),
    'v18agg': Path(
        '/home/hj/Data_Process/protocol_dynamic/inference_dumps/worldpose/full/kp_rtmlib/'
        'ief_global_v18agg_strict_mask_lowcam_K1_lam9e-3_RF31_ep0004__strictoff'),
}


@dataclass(frozen=True)
class Segment_Info:
    scene: str
    seq_id: str
    id_person: int
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class Segment_Metric_Row:
    method: str
    scene: str
    seq_id: str
    id_person: int
    frame_start: int
    frame_end: int
    num_frames: int
    segment_file: str
    mpjpe_mm: float
    tmpjpe_mm: float


@dataclass(frozen=True)
class Scope_Summary:
    method: str
    scope: str
    pred_joints_key: str
    num_segments: int
    num_person_frames: int
    num_unique_persons: int
    num_unique_original_frames: int
    frame_start_min: int | None
    frame_end_max: int | None
    mpjpe_mm: float
    tmpjpe_mm: float


def ensure_import_paths() -> None:
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    for rel in (
        'hjlib-evaluation/src',
        'hjlib-dataset-assembly/src',
        'hjlib-dataset-std/src',
        'hjlib-skeleton/src',
    ):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_segment_filename(path_pkl: Path) -> Segment_Info:
    match = re.match(
        r'^(?P<scene>.+)__(?P<seq>\d{4}_\d{4})__p(?P<pid>\d+)__'
        r'(?P<start>\d{7})_(?P<end>\d{7})$',
        path_pkl.stem,
    )
    if match is None:
        raise ValueError('unexpected segment filename: %s' % path_pkl.name)
    return Segment_Info(
        scene=match.group('scene'),
        seq_id=match.group('seq'),
        id_person=int(match.group('pid')),
        frame_start=int(match.group('start')),
        frame_end=int(match.group('end')),
    )


def compute_frame_metrics_mm(
        pred_joints: np.ndarray,
        gt_joints: np.ndarray,
        joint_indices: Sequence[int],
        root_indices: Sequence[int],
        scale_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
    pred_sub = pred_joints[:, list(joint_indices), :]
    gt_sub = gt_joints[:, list(joint_indices), :]
    pred_root = pred_joints[:, list(root_indices), :].mean(axis=1, keepdims=True)
    gt_root = gt_joints[:, list(root_indices), :].mean(axis=1, keepdims=True)

    assert pred_sub.shape == gt_sub.shape, (pred_sub.shape, gt_sub.shape)
    assert np.isfinite(pred_sub).all(), 'non-finite prediction joints'
    assert np.isfinite(gt_sub).all(), 'non-finite GT joints'
    assert np.isfinite(pred_root).all(), 'non-finite prediction roots'
    assert np.isfinite(gt_root).all(), 'non-finite GT roots'

    frame_mpjpe = np.linalg.norm(gt_sub - pred_sub, axis=2).mean(axis=1) * scale_mm
    frame_tmpjpe = np.linalg.norm(
        (gt_sub - gt_root) - (pred_sub - pred_root),
        axis=2,
    ).mean(axis=1) * scale_mm
    return frame_mpjpe, frame_tmpjpe


def dump_paths_for_scene(path_dump_dir: Path, scene: str) -> List[Path]:
    paths = sorted(path_dump_dir.glob('%s*.pkl' % scene))
    assert paths, 'no segment dumps for scene %s in %s' % (scene, path_dump_dir)
    return paths


def filter_paths_by_names(paths: Iterable[Path], names: Set[str] | None) -> List[Path]:
    if names is None:
        return list(paths)
    return [path for path in paths if path.name in names]


def summarize_rows(
        method: str,
        scope: str,
        pred_joints_key: str,
        rows: List[Segment_Metric_Row],
        sum_mpjpe_frames: float,
        sum_tmpjpe_frames: float,
    ) -> Scope_Summary:
    num_person_frames = sum(row.num_frames for row in rows)
    person_ids = {row.id_person for row in rows}
    frame_indices: Set[int] = set()
    for row in rows:
        frame_indices.update(range(row.frame_start, row.frame_end))

    if rows:
        frame_start_min = min(row.frame_start for row in rows)
        frame_end_max = max(row.frame_end for row in rows)
        mpjpe_mm = sum_mpjpe_frames / float(num_person_frames)
        tmpjpe_mm = sum_tmpjpe_frames / float(num_person_frames)
    else:
        frame_start_min = None
        frame_end_max = None
        mpjpe_mm = float('nan')
        tmpjpe_mm = float('nan')

    return Scope_Summary(
        method=method,
        scope=scope,
        pred_joints_key=pred_joints_key,
        num_segments=len(rows),
        num_person_frames=num_person_frames,
        num_unique_persons=len(person_ids),
        num_unique_original_frames=len(frame_indices),
        frame_start_min=frame_start_min,
        frame_end_max=frame_end_max,
        mpjpe_mm=float(mpjpe_mm),
        tmpjpe_mm=float(tmpjpe_mm),
    )


def compute_method_scope(
        method: str,
        scope: str,
        paths: List[Path],
        pred_joints_key: str,
        path_label_dump_root: Path,
    ) -> Dict[str, Any]:
    from hjlib_evaluation.dump_reader import load_inference_dump
    from hjlib_evaluation.get_by_dataset import get_gt_provider

    gt_provider = get_gt_provider('worldpose_smpl', str(path_label_dump_root))
    meta = gt_provider.get_eval_meta()
    metric = meta.metrics_3d[0]
    scale_mm = {'m': 1000.0, 'mm': 1.0}[meta.unit_world]

    rows: List[Segment_Metric_Row] = []
    sum_mpjpe_frames = 0.0
    sum_tmpjpe_frames = 0.0

    for path_pkl in paths:
        seg_info = parse_segment_filename(path_pkl)
        seg, pred = load_inference_dump(str(path_pkl))
        assert seg.name_scene == seg_info.scene, (seg.name_scene, seg_info.scene, path_pkl)
        assert seg.name_seq == seg_info.seq_id, (seg.name_seq, seg_info.seq_id, path_pkl)
        assert seg.id_person == seg_info.id_person, (seg.id_person, seg_info.id_person, path_pkl)
        assert seg.index_frame_original_start == seg_info.frame_start, (
            seg.index_frame_original_start, seg_info.frame_start, path_pkl)
        assert seg.index_frame_original_end == seg_info.frame_end, (
            seg.index_frame_original_end, seg_info.frame_end, path_pkl)

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

        frame_mpjpe, frame_tmpjpe = compute_frame_metrics_mm(
            pred_joints,
            gt_joints,
            metric.joint_indices_smpl_54,
            metric.root_indices_smpl_54_for_alignment,
            scale_mm,
        )
        num_frames = int(pred_joints.shape[0])
        sum_mpjpe_frames += float(frame_mpjpe.sum())
        sum_tmpjpe_frames += float(frame_tmpjpe.sum())
        rows.append(Segment_Metric_Row(
            method=method,
            scene=seg_info.scene,
            seq_id=seg_info.seq_id,
            id_person=seg_info.id_person,
            frame_start=seg_info.frame_start,
            frame_end=seg_info.frame_end,
            num_frames=num_frames,
            segment_file=path_pkl.name,
            mpjpe_mm=float(frame_mpjpe.mean()),
            tmpjpe_mm=float(frame_tmpjpe.mean()),
        ))

    summary = summarize_rows(
        method,
        scope,
        pred_joints_key,
        rows,
        sum_mpjpe_frames,
        sum_tmpjpe_frames,
    )
    return {
        'summary': asdict(summary),
        'segments': [asdict(row) for row in rows],
    }


def top_rows_by_mpjpe(rows: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row['mpjpe_mm']), reverse=True)[:limit]


def write_markdown(result: Dict[str, Any], path_md: Path) -> None:
    lines = [
        '# Sequence Monolith Quant',
        '',
        'Scene: `%s`' % result['scene'],
        'Metric: `%s`, unit: `%s`' % (result['metric_name'], result['unit']),
        'Aggregation: frame-weighted person-frame mean over segment dumps.',
        '',
        '## Own Dump Scope',
        '',
        '| method | segments | person-frames | unique persons | frame span | MPJPE mm | T-MPJPE mm |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: |',
    ]
    for row in result['own_scope']['summaries']:
        lines.append(format_summary_row(row))

    lines.extend([
        '',
        '## Common Segment Scope',
        '',
        '| method | segments | person-frames | unique persons | frame span | MPJPE mm | T-MPJPE mm |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: |',
    ])
    for row in result['common_segment_scope']['summaries']:
        lines.append(format_summary_row(row))

    lines.extend([
        '',
        'Common segment filenames: `%d`.' % int(result['common_segment_scope']['num_common_segments']),
        '',
        '## Largest MPJPE Segments',
        '',
    ])
    for method, rows in result['own_scope']['largest_mpjpe_segments'].items():
        lines.extend([
            '### `%s`' % method,
            '',
            '| segment | frames | MPJPE mm | T-MPJPE mm |',
            '| --- | ---: | ---: | ---: |',
        ])
        for row in rows:
            lines.append(
                '| `%s` | %d | %.2f | %.2f |'
                % (
                    row['segment_file'],
                    int(row['num_frames']),
                    float(row['mpjpe_mm']),
                    float(row['tmpjpe_mm']),
                )
            )
        lines.append('')

    lines.append('Full per-segment rows are in `%s`.' % result['json_name'])
    lines.append('')
    path_md.write_text('\n'.join(lines), encoding='utf-8')


def format_summary_row(row: Dict[str, Any]) -> str:
    if row['frame_start_min'] is None:
        frame_span = 'n/a'
    else:
        frame_span = '[%d,%d), %d frames' % (
            int(row['frame_start_min']),
            int(row['frame_end_max']),
            int(row['num_unique_original_frames']),
        )
    return (
        '| `%s` | %d | %d | %d | `%s` | %.2f | %.2f |'
        % (
            row['method'],
            int(row['num_segments']),
            int(row['num_person_frames']),
            int(row['num_unique_persons']),
            frame_span,
            float(row['mpjpe_mm']),
            float(row['tmpjpe_mm']),
        )
    )


def build_result(
        scene: str,
        pred_joints_key: str,
        path_label_dump_root: Path,
    ) -> Dict[str, Any]:
    from hjlib_evaluation.get_by_dataset import get_gt_provider

    gt_provider = get_gt_provider('worldpose_smpl', str(path_label_dump_root))
    meta = gt_provider.get_eval_meta()
    metric = meta.metrics_3d[0]

    method_paths: Dict[str, List[Path]] = {
        method: dump_paths_for_scene(path_dump_dir, scene)
        for method, path_dump_dir in METHOD_DUMP_DIRS.items()
        if method in ('abla_v1', 'v18agg')
    }
    method_names: Dict[str, Set[str]] = {
        method: {path.name for path in paths}
        for method, paths in method_paths.items()
    }
    common_names: Set[str] | None = None
    all_names: Set[str] = set()
    for names in method_names.values():
        all_names |= names
        if common_names is None:
            common_names = set(names)
        else:
            common_names &= names
    assert common_names is not None, 'no methods configured'

    own_scope: Dict[str, Any] = {}
    common_scope: Dict[str, Any] = {}
    for method, paths in method_paths.items():
        own_scope[method] = compute_method_scope(
            method,
            'own_dump_scope',
            paths,
            pred_joints_key,
            path_label_dump_root,
        )
        common_scope[method] = compute_method_scope(
            method,
            'common_segment_scope',
            filter_paths_by_names(paths, common_names),
            pred_joints_key,
            path_label_dump_root,
        )

    return {
        'scene': scene,
        'metric_name': metric.name,
        'unit': 'mm',
        'label_dump_root': str(path_label_dump_root),
        'pred_joints_key': pred_joints_key,
        'method_dump_dirs': {
            method: str(path_dump_dir)
            for method, path_dump_dir in METHOD_DUMP_DIRS.items()
            if method in ('abla_v1', 'v18agg')
        },
        'own_scope': {
            'summaries': [
                own_scope['abla_v1']['summary'],
                own_scope['v18agg']['summary'],
            ],
            'segments': {
                method: data['segments']
                for method, data in own_scope.items()
            },
            'largest_mpjpe_segments': {
                method: top_rows_by_mpjpe(data['segments'])
                for method, data in own_scope.items()
            },
        },
        'common_segment_scope': {
            'num_common_segments': len(common_names),
            'summaries': [
                common_scope['abla_v1']['summary'],
                common_scope['v18agg']['summary'],
            ],
            'segments': {
                method: data['segments']
                for method, data in common_scope.items()
            },
            'missing_by_method': {
                method: sorted(all_names - names)
                for method, names in method_names.items()
            },
        },
    }


def main(
        scene: Annotated[str, typer.Option(help='WorldPose scene name.')] = DEFAULT_SCENE,
        label_dump_root: Annotated[Path, typer.Option(help='WorldPose label dump root.')] = DEFAULT_LABEL_DUMP_ROOT,
        pred_joints_key: Annotated[str, typer.Option(help='Prediction joint field to evaluate.')] = DEFAULT_PRED_JOINTS_KEY,
        out_root: Annotated[Path | None, typer.Option(help='Output root; defaults under campaign artifacts.')] = None,
    ) -> None:
    ensure_import_paths()
    if out_root is None:
        out_root = ARTIFACT_ROOT / 'sequence_quant' / scene
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / 'metrics.json'
    out_md = out_root / 'metrics.md'

    result = build_result(scene, pred_joints_key, label_dump_root)
    result['json_name'] = out_json.name

    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    write_markdown(result, out_md)

    for row in result['own_scope']['summaries']:
        print('%s own n=%d MPJPE=%.2fmm T-MPJPE=%.2fmm' % (
            row['method'],
            int(row['num_person_frames']),
            float(row['mpjpe_mm']),
            float(row['tmpjpe_mm']),
        ))
    print('common segments: %d' % int(result['common_segment_scope']['num_common_segments']))
    for row in result['common_segment_scope']['summaries']:
        print('%s common n=%d MPJPE=%.2fmm T-MPJPE=%.2fmm' % (
            row['method'],
            int(row['num_person_frames']),
            float(row['mpjpe_mm']),
            float(row['tmpjpe_mm']),
        ))
    print('json: %s' % out_json)
    print('md: %s' % out_md)


if __name__ == '__main__':
    typer.run(main)
