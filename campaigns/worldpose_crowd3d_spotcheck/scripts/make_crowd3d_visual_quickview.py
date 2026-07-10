'''Build quick visual panels for the pulled Crowd3D downstream package.

This deliberately does not match Crowd3D people to GT/monolith people. It only
places the package-provided input, render, and depth views next to each other so
we can inspect the result set before choosing a matching policy.
'''

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated, Dict, List

import cv2
import numpy as np
import typer


WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
DEFAULT_PACKAGE_ROOT = Path(
    '/home/hj/Data_Process/protocol_dynamic/external_results/worldpose/crowd3d/'
    'NET_ARG_231908_downstream')
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / 'crowd3d_visual'


def read_manifest(path_manifest: Path) -> List[Dict[str, str]]:
    with path_manifest.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def resize_to_box(image: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    scale = min(float(width) / float(iw), float(height) / float(ih))
    nw = max(1, int(round(float(iw) * scale)))
    nh = max(1, int(round(float(ih) * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def labeled_panel(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:
    body = resize_to_box(image, width, height)
    header_h = 38
    out = np.full((height + header_h, width, 3), 16, dtype=np.uint8)
    out[header_h:] = body
    cv2.putText(
        out, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        (245, 245, 245), 2, cv2.LINE_AA)
    return out


def read_image_or_placeholder(path_image: Path, title: str, width: int, height: int) -> np.ndarray:
    if path_image.is_file():
        image = cv2.imread(str(path_image), cv2.IMREAD_COLOR)
        assert image is not None, 'cv2.imread failed: %s' % path_image
        return image
    image = np.full((height, width, 3), 38, dtype=np.uint8)
    cv2.putText(
        image, 'missing: %s' % title, (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2, cv2.LINE_AA)
    return image


def resolve_visual_path(package_root: Path, frame: str, rel_path: str, kind: str) -> Path:
    path = package_root / rel_path if rel_path else package_root / '__missing__.jpg'
    if path.is_file():
        return path
    if kind == 'render':
        fallback = package_root / 'reproduced_visual' / 'render' / frame / ('%s_result.jpg' % frame)
        if fallback.is_file():
            return fallback
    if kind == 'depth':
        fallback = package_root / 'reproduced_visual' / 'depth' / frame / ('%s_with_depth.jpg' % frame)
        if fallback.is_file():
            return fallback
    return path


def build_frame_panel(
        package_root: Path,
        row: Dict[str, str],
        panel_width: int,
        panel_height: int,
    ) -> np.ndarray:
    frame = row['frame']
    input_image = read_image_or_placeholder(
        package_root / row['scene_image'], 'input', panel_width, panel_height)
    render_path = resolve_visual_path(package_root, frame, row['render_visual'], 'render')
    render_image = read_image_or_placeholder(
        render_path, 'render', panel_width, panel_height)
    depth_rel = row.get('depth_visual', '')
    depth_path = resolve_visual_path(package_root, frame, depth_rel, 'depth')
    depth_image = read_image_or_placeholder(
        depth_path, 'depth', panel_width, panel_height)

    meta = 'patch=%s patch-person=%s merged-person=%s' % (
        row.get('patch_count', ''),
        row.get('person_result_count_patch_level', ''),
        row.get('person_count_after_scene_merge', ''),
    )
    panels = [
        labeled_panel(input_image, '%s input' % frame, panel_width, panel_height),
        labeled_panel(render_image, 'Crowd3D render', panel_width, panel_height),
        labeled_panel(depth_image, 'Crowd3D depth', panel_width, panel_height),
    ]
    body = np.concatenate(panels, axis=1)
    footer_h = 34
    out = np.full((body.shape[0] + footer_h, body.shape[1], 3), 16, dtype=np.uint8)
    out[:body.shape[0]] = body
    cv2.putText(
        out, meta, (14, body.shape[0] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (245, 245, 245), 2, cv2.LINE_AA)
    return out


def build_contact_sheet(frame_panels: List[np.ndarray]) -> np.ndarray:
    assert frame_panels, 'no frame panels'
    width = max(panel.shape[1] for panel in frame_panels)
    rows: List[np.ndarray] = []
    for panel in frame_panels:
        if panel.shape[1] == width:
            rows.append(panel)
            continue
        padded = np.full((panel.shape[0], width, 3), 16, dtype=np.uint8)
        padded[:, :panel.shape[1]] = panel
        rows.append(padded)
    gap = np.full((10, width, 3), 0, dtype=np.uint8)
    out_rows: List[np.ndarray] = []
    for index, row in enumerate(rows):
        if index > 0:
            out_rows.append(gap)
        out_rows.append(row)
    return np.concatenate(out_rows, axis=0)


def main(
        package_root: Annotated[Path, typer.Option(help='Crowd3D downstream package root.')] = DEFAULT_PACKAGE_ROOT,
        out_root: Annotated[Path, typer.Option(help='Output directory.')] = DEFAULT_OUTPUT_ROOT,
        panel_width: Annotated[int, typer.Option(help='Panel width in pixels.')] = 520,
        panel_height: Annotated[int, typer.Option(help='Panel height in pixels.')] = 292,
    ) -> None:
    manifest_path = package_root / 'manifest.csv'
    rows = read_manifest(manifest_path)
    side_by_side_root = out_root / 'side_by_side'
    side_by_side_root.mkdir(parents=True, exist_ok=True)

    frame_panels: List[np.ndarray] = []
    for row in rows:
        panel = build_frame_panel(
            package_root,
            row,
            panel_width,
            panel_height,
        )
        frame_panels.append(panel)
        path_out = side_by_side_root / ('%s_side_by_side.jpg' % row['frame'])
        ok = cv2.imwrite(str(path_out), panel)
        assert ok, 'cv2.imwrite failed: %s' % path_out

    contact = build_contact_sheet(frame_panels)
    path_contact = out_root / 'contact_sheet.jpg'
    ok = cv2.imwrite(str(path_contact), contact)
    assert ok, 'cv2.imwrite failed: %s' % path_contact

    summary = {
        'package_root': str(package_root),
        'manifest': str(manifest_path),
        'num_frames': len(rows),
        'side_by_side_root': str(side_by_side_root),
        'contact_sheet': str(path_contact),
        'frames': [
            {
                'frame': row['frame'],
                'person_count_after_scene_merge': int(row['person_count_after_scene_merge']),
                'patch_count': int(row['patch_count']),
                'has_depth': bool(row.get('depth_visual', '')),
            }
            for row in rows
        ],
    }
    path_summary = out_root / 'summary.json'
    path_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')

    print('frames: %d' % len(rows))
    print('side_by_side: %s' % side_by_side_root)
    print('contact_sheet: %s' % path_contact)
    print('summary: %s' % path_summary)


if __name__ == '__main__':
    typer.run(main)
