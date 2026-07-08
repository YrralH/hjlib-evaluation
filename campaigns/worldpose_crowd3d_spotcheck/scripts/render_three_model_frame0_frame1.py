'''Render the three-method qualitative check for NET_ARG_231908 frame0/frame1.

Outputs exactly three method folders under ``artifacts/overlays``:
``crowd3d``, ``abla_v1``, and ``v18agg``. Side-by-side review panels are written
outside that folder so ``overlays`` stays method-only.
'''

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
OVERLAY_ROOT = ARTIFACT_ROOT / 'overlays'
VIS_ROOT = ARTIFACT_ROOT / 'three_model_visual'
SIDE_BY_SIDE_ROOT = VIS_ROOT / 'side_by_side'
SUMMARY_PATH = VIS_ROOT / 'summary.json'
CROWD3D_PACKAGE_ROOT = Path(
    '/home/hj/Data_Process/protocol_dynamic/external_results/worldpose/crowd3d/'
    'NET_ARG_231908_downstream')
WORLDPOSE_ROOT = Path('/mnt/hj_exosX18_data0/hj/datasets/worldpose')
WORLDPOSE_UNDISTORT_FRAMES = Path(
    '/mnt/hj_exosX18_data0/hj/datasets/worldpose_frames/from_raw_fixed_K')

SCENE = 'NET_ARG_231908'
FRAME_INDICES = (0, 1)


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


def frame_name(frame_index: int) -> str:
    return '%s_frame%d' % (SCENE, frame_index)


def output_name(frame_index: int) -> str:
    return '%s.png' % frame_name(frame_index)


def prepare_output_dirs() -> None:
    if OVERLAY_ROOT.exists():
        shutil.rmtree(OVERLAY_ROOT)
    for method in ('crowd3d', 'abla_v1', 'v18agg'):
        (OVERLAY_ROOT / method).mkdir(parents=True, exist_ok=True)
    if VIS_ROOT.exists():
        shutil.rmtree(VIS_ROOT)
    SIDE_BY_SIDE_ROOT.mkdir(parents=True, exist_ok=True)


def load_segments_for_frame(scene: str, frame_index: int) -> Tuple[List[int], List[str]]:
    from select_worldpose_frames import METHOD_DUMP_DIRS, load_segments

    segments = load_segments(METHOD_DUMP_DIRS['abla_v1'])
    covering = [
        seg for seg in segments
        if seg.scene == scene and seg.frame_start <= frame_index < seg.frame_end
    ]
    assert covering, 'no abla_v1 segments cover %s frame %d' % (scene, frame_index)
    return (
        sorted({seg.id_person for seg in covering}),
        [Path(seg.path_pkl).name for seg in covering],
    )


def make_candidate(scene: str, frame_index: int) -> object:
    from select_worldpose_frames import Frame_Candidate

    person_ids, segment_files = load_segments_for_frame(scene, frame_index)
    return Frame_Candidate(
        scene=scene,
        frame_index=frame_index,
        num_covering_segments=len(segment_files),
        person_ids=person_ids,
        segment_files=segment_files,
    )


def read_crowd3d_render(frame_index: int) -> np.ndarray:
    name = frame_name(frame_index)
    paths = [
        CROWD3D_PACKAGE_ROOT / 'visual' / 'render' / ('%s_result.jpg' % name),
        CROWD3D_PACKAGE_ROOT / 'reproduced_visual' / 'render' / name / ('%s_result.jpg' % name),
    ]
    for path in paths:
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, 'cv2.imread failed: %s' % path
        return np.ascontiguousarray(image)
    raise FileNotFoundError('Crowd3D render not found for %s; tried %s' % (name, paths))


def draw_title(image: np.ndarray, lines: List[str]) -> np.ndarray:
    out = image.copy()
    y = 38
    for line in lines:
        cv2.putText(out, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        y += 34
    return out


def render_crowd3d(frame_index: int) -> np.ndarray:
    image = read_crowd3d_render(frame_index)
    return draw_title(image, ['Crowd3D render', frame_name(frame_index)])


def render_monolith(
        ds: object,
        path_undistort_frames: str,
        frame_index: int,
        method_name: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
    from select_worldpose_frames import METHOD_DUMP_DIRS, render_pred2d_overlay

    candidate = make_candidate(SCENE, frame_index)
    image, stats = render_pred2d_overlay(
        ds,
        path_undistort_frames,
        candidate,
        method_name,
        METHOD_DUMP_DIRS[method_name],
    )
    image = draw_title(
        image,
        [
            '%s projected joints_54_world' % method_name,
            frame_name(frame_index),
        ],
    )
    stats['method'] = method_name
    return image, stats


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    width = max(1, int(round(float(w) * float(height) / float(h))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_side_by_side(images: List[Tuple[str, np.ndarray]], height: int = 430) -> np.ndarray:
    panels: List[np.ndarray] = []
    for title, image in images:
        panel = resize_to_height(image, height)
        header = np.full((36, panel.shape[1], 3), 16, dtype=np.uint8)
        cv2.putText(
            header, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (245, 245, 245), 2, cv2.LINE_AA)
        panels.append(np.concatenate([header, panel], axis=0))
    return np.concatenate(panels, axis=1)


def make_contact_sheet(paths: List[Path], path_out: Path) -> None:
    rows: List[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None, 'cv2.imread failed: %s' % path
        rows.append(resize_to_height(image, 430))
    width = max(row.shape[1] for row in rows)
    padded_rows: List[np.ndarray] = []
    for row in rows:
        canvas = np.full((row.shape[0], width, 3), 16, dtype=np.uint8)
        canvas[:, :row.shape[1]] = row
        padded_rows.append(canvas)
    gap = np.full((10, width, 3), 0, dtype=np.uint8)
    out_rows: List[np.ndarray] = []
    for index, row in enumerate(padded_rows):
        if index > 0:
            out_rows.append(gap)
        out_rows.append(row)
    ok = cv2.imwrite(str(path_out), np.concatenate(out_rows, axis=0))
    assert ok, 'cv2.imwrite failed: %s' % path_out


def main() -> None:
    ensure_import_paths()
    from hjlib_dataset_std.datasets.worldpose.worldpose import WorldPose_Std

    prepare_output_dirs()
    path_root = str(WORLDPOSE_ROOT)
    path_undistort = str(WORLDPOSE_UNDISTORT_FRAMES)
    ds = WorldPose_Std(path_data_root=path_root, path_undistort_frames=path_undistort)

    summary: Dict[str, Any] = {
        'scene': SCENE,
        'frames': list(FRAME_INDICES),
        'crowd3d_package_root': str(CROWD3D_PACKAGE_ROOT),
        'overlays': {},
        'projection_stats': [],
        'side_by_side': [],
    }
    side_by_side_paths: List[Path] = []

    for frame_index in FRAME_INDICES:
        crowd = render_crowd3d(frame_index)
        path_crowd = OVERLAY_ROOT / 'crowd3d' / output_name(frame_index)
        ok = cv2.imwrite(str(path_crowd), crowd)
        assert ok, 'cv2.imwrite failed: %s' % path_crowd

        abla, stats_abla = render_monolith(ds, path_undistort, frame_index, 'abla_v1')
        path_abla = OVERLAY_ROOT / 'abla_v1' / output_name(frame_index)
        ok = cv2.imwrite(str(path_abla), abla)
        assert ok, 'cv2.imwrite failed: %s' % path_abla

        v18, stats_v18 = render_monolith(ds, path_undistort, frame_index, 'v18agg')
        path_v18 = OVERLAY_ROOT / 'v18agg' / output_name(frame_index)
        ok = cv2.imwrite(str(path_v18), v18)
        assert ok, 'cv2.imwrite failed: %s' % path_v18

        path_sbs = SIDE_BY_SIDE_ROOT / ('%s_three_models.jpg' % frame_name(frame_index))
        sbs = make_side_by_side([
            ('Crowd3D', crowd),
            ('abla_v1', abla),
            ('v18agg', v18),
        ])
        ok = cv2.imwrite(str(path_sbs), sbs)
        assert ok, 'cv2.imwrite failed: %s' % path_sbs
        side_by_side_paths.append(path_sbs)

        summary['overlays'][frame_name(frame_index)] = {
            'crowd3d': str(path_crowd),
            'abla_v1': str(path_abla),
            'v18agg': str(path_v18),
        }
        summary['projection_stats'].extend([stats_abla, stats_v18])
        summary['side_by_side'].append(str(path_sbs))

    path_contact = VIS_ROOT / 'contact_sheet.jpg'
    make_contact_sheet(side_by_side_paths, path_contact)
    summary['contact_sheet'] = str(path_contact)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

    print('overlays: %s' % OVERLAY_ROOT)
    print('side_by_side: %s' % SIDE_BY_SIDE_ROOT)
    print('contact_sheet: %s' % path_contact)
    print('summary: %s' % SUMMARY_PATH)


if __name__ == '__main__':
    main()
