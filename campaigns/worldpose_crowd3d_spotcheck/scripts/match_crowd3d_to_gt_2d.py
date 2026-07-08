'''Match Crowd3D merged people to WorldPose GT people using 2D joints.

This is a campaign-local exploratory matcher. It intentionally evaluates only
2D geometry and leaves metric computation for a later step after visual review.
'''

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
DEFAULT_PACKAGE_ROOT = Path(
    '/home/hj/Data_Process/protocol_dynamic/external_results/worldpose/crowd3d/'
    'NET_ARG_231908_downstream')
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / 'crowd3d_matching'
DEFAULT_WORLDPOSE_ROOT = Path('/mnt/hj_exosX18_data0/hj/datasets/worldpose')
DEFAULT_WORLDPOSE_UNDISTORT_FRAMES = Path(
    '/mnt/hj_exosX18_data0/hj/datasets/worldpose_frames/from_raw_fixed_K')
DEFAULT_SCENE = 'NET_ARG_231908'


@dataclass(frozen=True)
class Match_Row:
    frame: str
    frame_index: int
    gt_id: int
    crowd_index: int
    cost_px: float
    chamfer_px: float
    center_px: float
    gt_center_uv: Tuple[float, float]
    crowd_center_uv: Tuple[float, float]


def ensure_import_paths() -> None:
    for rel in (
        'hjlib-dataset-std/src',
        'hjlib-dataset-raw/src',
        'hjlib-camera/src',
        'hjlib-streamer/src',
        'hjlib-smpl/src',
        'hjlib-vis-2d/src',
        'hjlib-geometry/src',
        'hjlib-cache/src',
    ):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_frame_index(frame_name: str) -> int:
    match = re.match(r'^NET_ARG_231908_frame(?P<idx>\d+)$', frame_name)
    assert match is not None, 'unexpected Crowd3D frame name: %s' % frame_name
    return int(match.group('idx'))


def read_manifest(path_manifest: Path) -> List[Dict[str, str]]:
    with path_manifest.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def selected_manifest_rows(
        rows: List[Dict[str, str]],
        frame_start: int,
        frame_end: int,
    ) -> List[Dict[str, str]]:
    selected = [
        row for row in rows
        if frame_start <= parse_frame_index(row['frame']) < frame_end
    ]
    selected.sort(key=lambda row: parse_frame_index(row['frame']))
    assert selected, 'no manifest rows selected for [%d, %d)' % (frame_start, frame_end)
    return selected


def finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 2, points.shape
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def trimmed_mean(values: np.ndarray, trim_ratio: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    assert values.ndim == 1, values.shape
    if values.size == 0:
        return float('inf')
    keep = max(1, int(np.ceil(float(values.size) * trim_ratio)))
    return float(values[:keep].mean())


def center_of_points(points: np.ndarray) -> np.ndarray:
    pts = finite_points(points)
    if pts.shape[0] == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return pts.mean(axis=0)


def pair_cost(
        gt_points: np.ndarray,
        crowd_points: np.ndarray,
        trim_ratio: float,
        center_weight: float,
    ) -> Tuple[float, float, float]:
    gt = finite_points(gt_points)
    crowd = finite_points(crowd_points)
    if gt.shape[0] == 0 or crowd.shape[0] == 0:
        return float('inf'), float('inf'), float('inf')

    diff = gt[:, None, :] - crowd[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    gt_to_crowd = distances.min(axis=1)
    crowd_to_gt = distances.min(axis=0)
    chamfer = 0.5 * (
        trimmed_mean(gt_to_crowd, trim_ratio)
        + trimmed_mean(crowd_to_gt, trim_ratio)
    )

    center_gt = gt.mean(axis=0)
    center_crowd = crowd.mean(axis=0)
    center = float(np.linalg.norm(center_gt - center_crowd))
    cost = chamfer + center_weight * center
    return float(cost), float(chamfer), center


def compute_cost_matrices(
        gt_points: np.ndarray,
        crowd_points: np.ndarray,
        trim_ratio: float,
        center_weight: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_gt = gt_points.shape[0]
    num_crowd = crowd_points.shape[0]
    cost = np.full((num_gt, num_crowd), np.inf, dtype=np.float64)
    chamfer = np.full((num_gt, num_crowd), np.inf, dtype=np.float64)
    center = np.full((num_gt, num_crowd), np.inf, dtype=np.float64)
    for i in range(num_gt):
        for j in range(num_crowd):
            c, ch, ce = pair_cost(gt_points[i], crowd_points[j], trim_ratio, center_weight)
            cost[i, j] = c
            chamfer[i, j] = ch
            center[i, j] = ce
    return cost, chamfer, center


def color_for_index(index: int) -> Tuple[int, int, int]:
    palette = (
        (0, 220, 255),
        (0, 180, 0),
        (255, 120, 0),
        (180, 80, 255),
        (255, 0, 160),
        (80, 220, 120),
        (255, 220, 0),
        (120, 140, 255),
    )
    return palette[index % len(palette)]


def draw_bbox_h1h2w1w2(
        image: np.ndarray,
        bbox: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
    h1, h2, w1, w2 = [float(x) for x in bbox]
    if not np.isfinite([h1, h2, w1, w2]).all():
        return
    cv2.rectangle(
        image,
        (int(round(w1)), int(round(h1))),
        (int(round(w2)), int(round(h2))),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_match_overlay(
        image: np.ndarray,
        frame_name: str,
        gt_ids: Sequence[int],
        gt_points: np.ndarray,
        gt_bbox: np.ndarray,
        crowd_points: np.ndarray,
        matches: Sequence[Match_Row],
        unmatched_gt_ids: Sequence[int],
        unmatched_crowd_indices: Sequence[int],
        path_out: Path,
    ) -> None:
    out = image.copy()
    match_by_gt = {row.gt_id: row for row in matches}
    match_by_crowd = {row.crowd_index: row for row in matches}

    for crowd_index in unmatched_crowd_indices:
        center = center_of_points(crowd_points[crowd_index])
        if np.isfinite(center).all():
            cv2.circle(out, tuple(np.round(center).astype(int)), 4, (120, 120, 120), -1, cv2.LINE_AA)

    for row in matches:
        color = color_for_index(row.gt_id)
        gt_center = tuple(np.round(row.gt_center_uv).astype(int))
        crowd_center = tuple(np.round(row.crowd_center_uv).astype(int))
        cv2.line(out, gt_center, crowd_center, color, 2, cv2.LINE_AA)
        cv2.circle(out, gt_center, 7, color, 2, cv2.LINE_AA)
        cv2.circle(out, crowd_center, 5, color, -1, cv2.LINE_AA)
        mid = ((gt_center[0] + crowd_center[0]) // 2, (gt_center[1] + crowd_center[1]) // 2)
        cv2.putText(
            out, 'g%d-c%d %.0f' % (row.gt_id, row.crowd_index, row.cost_px),
            mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for idx, gt_id in enumerate(gt_ids):
        color = color_for_index(int(gt_id)) if int(gt_id) in match_by_gt else (80, 80, 255)
        thickness = 2 if int(gt_id) in match_by_gt else 1
        draw_bbox_h1h2w1w2(out, gt_bbox[idx], color, thickness)
        center = center_of_points(gt_points[idx])
        if np.isfinite(center).all():
            text = 'g%d' % int(gt_id)
            if int(gt_id) in unmatched_gt_ids:
                text += ' unmatched'
            cv2.putText(
                out, text, tuple(np.round(center).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for crowd_index, row in match_by_crowd.items():
        center = tuple(np.round(row.crowd_center_uv).astype(int))
        cv2.putText(
            out, 'c%d' % crowd_index, (center[0] + 6, center[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_for_index(row.gt_id), 2, cv2.LINE_AA)

    cv2.putText(
        out,
        '%s matched=%d unmatched_gt=%d unmatched_crowd=%d'
        % (frame_name, len(matches), len(unmatched_gt_ids), len(unmatched_crowd_indices)),
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    ok = cv2.imwrite(str(path_out), out)
    assert ok, 'cv2.imwrite failed: %s' % path_out


def build_contact_sheet(paths: Sequence[Path], path_out: Path, tile_width: int = 900) -> None:
    tiles: List[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, 'cv2.imread failed: %s' % path
        height, width = image.shape[:2]
        tile_height = max(1, int(round(float(height) * float(tile_width) / float(width))))
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    assert tiles, 'no overlay paths'
    max_height = max(tile.shape[0] for tile in tiles)
    rows: List[np.ndarray] = []
    for start in range(0, len(tiles), 2):
        chunk = tiles[start:start + 2]
        padded_tiles: List[np.ndarray] = []
        for tile in chunk:
            padded = np.full((max_height, tile_width, 3), 20, dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            padded_tiles.append(padded)
        if len(padded_tiles) == 1:
            padded_tiles.append(np.full((max_height, tile_width, 3), 20, dtype=np.uint8))
        rows.append(np.concatenate(padded_tiles, axis=1))
    gap = np.full((10, rows[0].shape[1], 3), 0, dtype=np.uint8)
    out_rows: List[np.ndarray] = []
    for idx, row in enumerate(rows):
        if idx > 0:
            out_rows.append(gap)
        out_rows.append(row)
    out = np.concatenate(out_rows, axis=0)
    ok = cv2.imwrite(str(path_out), out)
    assert ok, 'cv2.imwrite failed: %s' % path_out


def compute_matches(
        package_root: Path,
        output_root: Path,
        frame_start: int,
        frame_end: int,
        max_cost_px: float,
        trim_ratio: float,
        center_weight: float,
        worldpose_root: Path,
        worldpose_undistort_frames: Path,
    ) -> Dict[str, Any]:
    from hjlib_dataset_std.datasets.worldpose.worldpose import WorldPose_Std

    ds = WorldPose_Std(
        path_data_root=str(worldpose_root),
        path_undistort_frames=str(worldpose_undistort_frames),
    )
    kpts_label, bbox_label = ds.get_keypoints_2d_and_bbox_by_name_scene(DEFAULT_SCENE)

    rows = selected_manifest_rows(
        read_manifest(package_root / 'manifest.csv'),
        frame_start,
        frame_end,
    )
    overlay_root = output_root / 'overlays'
    overlay_root.mkdir(parents=True, exist_ok=True)

    frames_out: List[Dict[str, Any]] = []
    all_matches: List[Match_Row] = []
    overlay_paths: List[Path] = []

    for row in rows:
        frame_name = row['frame']
        frame_index = parse_frame_index(frame_name)
        gt_points, gt_ids_np = kpts_label.get_one_frame(frame_index)
        gt_bbox, bbox_ids_np = bbox_label.get_one_frame(frame_index)
        gt_ids = [int(x) for x in gt_ids_np]
        bbox_ids = [int(x) for x in bbox_ids_np]
        assert gt_ids == bbox_ids, (frame_name, gt_ids, bbox_ids)
        gt_points_np = np.asarray(gt_points, dtype=np.float64)
        gt_bbox_np = np.asarray(gt_bbox, dtype=np.float64)

        crowd_npz = np.load(package_root / row['joints_npz'])
        crowd_points_np = np.asarray(crowd_npz['pj2d_71_scene_px'], dtype=np.float64)

        cost, chamfer, center = compute_cost_matrices(
            gt_points_np,
            crowd_points_np,
            trim_ratio,
            center_weight,
        )
        assigned_rows, assigned_cols = linear_sum_assignment(cost)

        matches: List[Match_Row] = []
        rejected: List[Dict[str, Any]] = []
        matched_gt_indices = set()
        matched_crowd_indices = set()
        for gt_row, crowd_col in zip(assigned_rows, assigned_cols):
            c = float(cost[gt_row, crowd_col])
            row_info = {
                'gt_id': gt_ids[int(gt_row)],
                'crowd_index': int(crowd_col),
                'cost_px': c,
                'chamfer_px': float(chamfer[gt_row, crowd_col]),
                'center_px': float(center[gt_row, crowd_col]),
            }
            if c > max_cost_px:
                rejected.append(row_info)
                continue
            gt_center = center_of_points(gt_points_np[gt_row])
            crowd_center = center_of_points(crowd_points_np[crowd_col])
            match = Match_Row(
                frame=frame_name,
                frame_index=frame_index,
                gt_id=gt_ids[int(gt_row)],
                crowd_index=int(crowd_col),
                cost_px=c,
                chamfer_px=float(chamfer[gt_row, crowd_col]),
                center_px=float(center[gt_row, crowd_col]),
                gt_center_uv=(float(gt_center[0]), float(gt_center[1])),
                crowd_center_uv=(float(crowd_center[0]), float(crowd_center[1])),
            )
            matches.append(match)
            matched_gt_indices.add(int(gt_row))
            matched_crowd_indices.add(int(crowd_col))

        unmatched_gt_indices = [
            idx for idx in range(len(gt_ids))
            if idx not in matched_gt_indices
        ]
        unmatched_crowd_indices = [
            idx for idx in range(crowd_points_np.shape[0])
            if idx not in matched_crowd_indices
        ]
        all_matches.extend(matches)

        image = cv2.imread(str(package_root / row['scene_image']), cv2.IMREAD_COLOR)
        assert image is not None, 'cv2.imread failed: %s' % (package_root / row['scene_image'])
        path_overlay = overlay_root / ('%s_matching.jpg' % frame_name)
        draw_match_overlay(
            image,
            frame_name,
            gt_ids,
            gt_points_np,
            gt_bbox_np,
            crowd_points_np,
            matches,
            [gt_ids[idx] for idx in unmatched_gt_indices],
            unmatched_crowd_indices,
            path_overlay,
        )
        overlay_paths.append(path_overlay)

        frames_out.append({
            'frame': frame_name,
            'frame_index': frame_index,
            'gt_count': len(gt_ids),
            'crowd_count': int(crowd_points_np.shape[0]),
            'accepted_count': len(matches),
            'rejected_count': len(rejected),
            'unmatched_gt_ids': [gt_ids[idx] for idx in unmatched_gt_indices],
            'unmatched_crowd_indices': unmatched_crowd_indices,
            'matches': [asdict(match) for match in matches],
            'rejected_assignments': rejected,
            'overlay': str(path_overlay),
        })

    path_contact = output_root / 'contact_sheet.jpg'
    build_contact_sheet(overlay_paths, path_contact)

    accepted_costs = [match.cost_px for match in all_matches]
    summary = {
        'algorithm': 'scipy.optimize.linear_sum_assignment',
        'assignment_name': 'linear sum assignment / Hungarian-style bipartite matching',
        'cost': {
            'type': '2D robust bidirectional nearest-joint distance',
            'trim_ratio': trim_ratio,
            'center_weight': center_weight,
            'max_cost_px': max_cost_px,
        },
        'package_root': str(package_root),
        'scene': DEFAULT_SCENE,
        'frame_range': [frame_start, frame_end],
        'num_frames': len(frames_out),
        'total_gt': int(sum(frame['gt_count'] for frame in frames_out)),
        'total_crowd': int(sum(frame['crowd_count'] for frame in frames_out)),
        'total_accepted': int(sum(frame['accepted_count'] for frame in frames_out)),
        'mean_accepted_cost_px': float(np.mean(accepted_costs)) if accepted_costs else float('nan'),
        'median_accepted_cost_px': float(np.median(accepted_costs)) if accepted_costs else float('nan'),
        'frames': frames_out,
        'contact_sheet': str(path_contact),
        'overlay_root': str(overlay_root),
    }
    return summary


def write_csv(summary: Dict[str, Any], path_csv: Path) -> None:
    with path_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'frame', 'frame_index', 'gt_id', 'crowd_index',
                'cost_px', 'chamfer_px', 'center_px',
                'gt_center_u', 'gt_center_v', 'crowd_center_u', 'crowd_center_v',
            ],
        )
        writer.writeheader()
        for frame in summary['frames']:
            for match in frame['matches']:
                writer.writerow({
                    'frame': match['frame'],
                    'frame_index': match['frame_index'],
                    'gt_id': match['gt_id'],
                    'crowd_index': match['crowd_index'],
                    'cost_px': '%.6f' % float(match['cost_px']),
                    'chamfer_px': '%.6f' % float(match['chamfer_px']),
                    'center_px': '%.6f' % float(match['center_px']),
                    'gt_center_u': '%.6f' % float(match['gt_center_uv'][0]),
                    'gt_center_v': '%.6f' % float(match['gt_center_uv'][1]),
                    'crowd_center_u': '%.6f' % float(match['crowd_center_uv'][0]),
                    'crowd_center_v': '%.6f' % float(match['crowd_center_uv'][1]),
                })


def write_markdown(summary: Dict[str, Any], path_md: Path) -> None:
    lines = [
        '# Crowd3D 2D Matching',
        '',
        'Algorithm: `scipy.optimize.linear_sum_assignment`.',
        'Cost: robust bidirectional nearest-joint 2D distance plus center distance.',
        '',
        '| frames | GT | Crowd3D | accepted | mean cost px | median cost px |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
        '| %d | %d | %d | %d | %.2f | %.2f |'
        % (
            int(summary['num_frames']),
            int(summary['total_gt']),
            int(summary['total_crowd']),
            int(summary['total_accepted']),
            float(summary['mean_accepted_cost_px']),
            float(summary['median_accepted_cost_px']),
        ),
        '',
        '| frame | GT | Crowd3D | accepted | rejected | unmatched GT | unmatched Crowd |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |',
    ]
    for frame in summary['frames']:
        lines.append(
            '| `%s` | %d | %d | %d | %d | `%s` | %d |'
            % (
                frame['frame'],
                int(frame['gt_count']),
                int(frame['crowd_count']),
                int(frame['accepted_count']),
                int(frame['rejected_count']),
                ','.join('%d' % x for x in frame['unmatched_gt_ids']),
                len(frame['unmatched_crowd_indices']),
            )
        )
    lines.extend([
        '',
        'Review overlays in `overlays/` and `contact_sheet.jpg` before using these matches for metrics.',
        '',
    ])
    path_md.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-end', type=int, default=10,
                        help='exclusive end; default evaluates frame0-frame9')
    parser.add_argument('--max-cost-px', type=float, default=90.0)
    parser.add_argument('--trim-ratio', type=float, default=0.8)
    parser.add_argument('--center-weight', type=float, default=0.25)
    parser.add_argument('--worldpose-root', type=Path, default=DEFAULT_WORLDPOSE_ROOT)
    parser.add_argument('--worldpose-undistort-frames', type=Path,
                        default=DEFAULT_WORLDPOSE_UNDISTORT_FRAMES)
    args = parser.parse_args()

    ensure_import_paths()
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = compute_matches(
        package_root=args.package_root,
        output_root=args.out_root,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        max_cost_px=args.max_cost_px,
        trim_ratio=args.trim_ratio,
        center_weight=args.center_weight,
        worldpose_root=args.worldpose_root,
        worldpose_undistort_frames=args.worldpose_undistort_frames,
    )

    path_json = args.out_root / 'matches.json'
    path_csv = args.out_root / 'matches.csv'
    path_md = args.out_root / 'matches.md'
    path_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
    write_csv(summary, path_csv)
    write_markdown(summary, path_md)

    print('frames: %d' % int(summary['num_frames']))
    print('gt/crowd/accepted: %d/%d/%d' % (
        int(summary['total_gt']),
        int(summary['total_crowd']),
        int(summary['total_accepted']),
    ))
    print('mean cost px: %.2f' % float(summary['mean_accepted_cost_px']))
    print('json: %s' % path_json)
    print('csv: %s' % path_csv)
    print('md: %s' % path_md)
    print('overlays: %s' % summary['overlay_root'])
    print('contact_sheet: %s' % summary['contact_sheet'])


if __name__ == '__main__':
    main()
