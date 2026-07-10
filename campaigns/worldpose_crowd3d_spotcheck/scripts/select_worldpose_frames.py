'''Select and visualize original multi-person WorldPose frames for spot checks.

This is campaign-local glue: choose frames covered by the monolith abla v1 dump
using segment filenames, then render original WorldPose frames with GT 2D
keypoints/bboxes or projected prediction joints.
'''

from __future__ import annotations

import json
import os
import os.path as osp
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Literal, Tuple

import cv2
import numpy as np
import typer


REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = WORK_ROOT / 'artifacts'
OVERLAY_ROOT = ARTIFACT_ROOT / 'overlays'

METHOD_DUMP_DIRS = {
    'abla_v1': Path(
        '/home/hj/Data_Process/protocol_dynamic/inference_dumps/worldpose/full/kp_rtmlib/'
        'ablation_a00_hvip_hipmid_K1_ep0004'),
    'v18agg': Path(
        '/home/hj/Data_Process/protocol_dynamic/inference_dumps/worldpose/full/kp_rtmlib/'
        'ief_global_v18agg_strict_mask_lowcam_K1_lam9e-3_RF31_ep0004__strictoff'),
}
SELECTION_METHOD = 'abla_v1'

DEFAULT_NUM_FRAMES = 3
DEFAULT_MIN_PERSONS = 3


@dataclass(frozen=True)
class Segment_Info:
    scene: str
    seq_id: str
    id_person: int
    frame_start: int
    frame_end: int
    path_pkl: str


@dataclass(frozen=True)
class Frame_Candidate:
    scene: str
    frame_index: int
    num_covering_segments: int
    person_ids: List[int]
    segment_files: List[str]


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
        'hjlib-evaluation/src',
    ):
        path = str(REPO_ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)
    if str(REPO_ROOT / 'hjlib-experiments') not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / 'hjlib-experiments'))
    try:
        import local_setting  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return
    path_monolith = getattr(local_setting, 'PATH_MONOLITH', None)
    if path_monolith is not None and str(path_monolith) not in sys.path:
        sys.path.insert(0, str(path_monolith))


def parse_segment_filename(path_pkl: Path) -> Segment_Info:
    stem = path_pkl.stem
    match = re.match(
        r'^(?P<scene>.+)__(?P<seq>\d{4}_\d{4})__p(?P<pid>\d+)__'
        r'(?P<start>\d{7})_(?P<end>\d{7})$',
        stem,
    )
    if match is None:
        raise ValueError('unexpected segment filename: %s' % path_pkl.name)
    return Segment_Info(
        scene=match.group('scene'),
        seq_id=match.group('seq'),
        id_person=int(match.group('pid')),
        frame_start=int(match.group('start')),
        frame_end=int(match.group('end')),
        path_pkl=str(path_pkl),
    )


def load_segments(path_dump_dir: Path) -> List[Segment_Info]:
    paths = sorted(path_dump_dir.glob('*.pkl'))
    assert paths, 'no pkl files found: %s' % path_dump_dir
    return [parse_segment_filename(path) for path in paths]


def candidate_frames_from_segments(
        segments: Iterable[Segment_Info],
        min_persons: int,
    ) -> List[Frame_Candidate]:
    per_scene: Dict[str, List[Segment_Info]] = {}
    for seg in segments:
        per_scene.setdefault(seg.scene, []).append(seg)

    candidates: List[Frame_Candidate] = []
    for scene, scene_segments in sorted(per_scene.items()):
        # Evaluate only event boundaries and midpoints. That is enough to find
        # dense frames without scanning every video frame.
        probe_frames = set()
        for seg in scene_segments:
            probe_frames.add(seg.frame_start)
            probe_frames.add(max(seg.frame_start, seg.frame_end - 1))
            probe_frames.add((seg.frame_start + seg.frame_end - 1) // 2)
        for frame_index in sorted(probe_frames):
            covering = [
                seg for seg in scene_segments
                if seg.frame_start <= frame_index < seg.frame_end
            ]
            person_ids = sorted({seg.id_person for seg in covering})
            if len(person_ids) < min_persons:
                continue
            candidates.append(Frame_Candidate(
                scene=scene,
                frame_index=frame_index,
                num_covering_segments=len(covering),
                person_ids=person_ids,
                segment_files=[osp.basename(seg.path_pkl) for seg in covering],
            ))

    candidates.sort(
        key=lambda x: (-len(x.person_ids), -x.num_covering_segments, x.scene, x.frame_index))
    return candidates


def choose_diverse_candidates(
        candidates: List[Frame_Candidate],
        num_frames: int,
    ) -> List[Frame_Candidate]:
    chosen: List[Frame_Candidate] = []
    used_scenes: set[str] = set()
    for cand in candidates:
        if cand.scene in used_scenes:
            continue
        chosen.append(cand)
        used_scenes.add(cand.scene)
        if len(chosen) >= num_frames:
            return chosen
    for cand in candidates:
        if cand in chosen:
            continue
        chosen.append(cand)
        if len(chosen) >= num_frames:
            break
    return chosen


def resolve_worldpose_paths() -> Tuple[str, str | None]:
    # Prefer the existing test local setting when available; it already records
    # the machine's raw root and pre-extracted undistorted frame cache.
    root_from_settings: str | None = None
    for path_settings in (
        REPO_ROOT / 'hjlib-experiments',
        REPO_ROOT / 'hjlib-integration-tests',
        REPO_ROOT / 'hjlib-migration-tests',
    ):
        if str(path_settings) not in sys.path:
            sys.path.insert(0, str(path_settings))
        try:
            import local_setting_test  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            local_setting_test = None  # type: ignore[assignment]
        if local_setting_test is not None and hasattr(local_setting_test, 'PATH_WORLDPOSE_DATA_ROOT'):
            root = getattr(local_setting_test, 'PATH_WORLDPOSE_DATA_ROOT')
            undistort = getattr(local_setting_test, 'PATH_WORLDPOSE_UNDISTORT_FRAMES', None)
            root_from_settings = str(root)
            if undistort is not None:
                return root_from_settings, str(undistort)
            break
        sys.modules.pop('local_setting_test', None)

    try:
        from lib_dynamic_hvip import local_setting as legacy_setting  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Could not resolve WorldPose root. Add local_setting_test.PATH_WORLDPOSE_DATA_ROOT '
            'or make dynamic_hvip local_setting importable.') from exc

    root = root_from_settings or getattr(legacy_setting, 'PATH_ROOT_WORLDPOSE', None)
    if root is None:
        labels = getattr(legacy_setting, 'PATH_ROOT_WORLDPOSE_LABELS', None)
        if labels is not None:
            # Monolith PATH_ROOT_WORLDPOSE_LABELS points at <root>/WorldPoseParam.
            root = str(Path(labels).parent)
    if root is None:
        raise RuntimeError('dynamic_hvip local_setting has no PATH_ROOT_WORLDPOSE root')
    undistort = getattr(legacy_setting, 'PATH_ROOT_WORLDPOSE_FRAME_EXTRACT', None)
    if undistort is not None:
        fixed_k = Path(str(undistort)) / 'from_raw_fixed_K'
        if fixed_k.is_dir():
            undistort = str(fixed_k)
    return str(root), None if undistort is None else str(undistort)


def load_prediction_joints_for_candidate(
        candidate: Frame_Candidate,
        path_dump_dir: Path,
    ) -> Dict[int, np.ndarray]:
    from hjlib_evaluation.dump_reader import load_inference_dump

    out: Dict[int, np.ndarray] = {}
    for filename in candidate.segment_files:
        path_pkl = path_dump_dir / filename
        assert path_pkl.is_file(), 'prediction dump file not found: %s' % path_pkl
        seg_info = parse_segment_filename(path_pkl)
        offset = candidate.frame_index - seg_info.frame_start
        assert 0 <= offset < seg_info.frame_end - seg_info.frame_start, (
            candidate.frame_index, seg_info)
        _segment, pred = load_inference_dump(seg_info.path_pkl)
        joints = np.asarray(pred['joints_54_world'], dtype=np.float64)
        assert joints.ndim == 3 and joints.shape[1:] == (54, 3), joints.shape
        out[seg_info.id_person] = joints[offset]
    return out


def projection_stats(
        pixels: np.ndarray,
        depth: np.ndarray,
        image_shape_hw: Tuple[int, int],
    ) -> Dict[str, Any]:
    height, width = image_shape_hw
    finite = np.isfinite(pixels).all(axis=1) & np.isfinite(depth)
    front = finite & (depth > 0.05)
    in_image = (
        front
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height))
    )
    finite_pixels = pixels[finite]
    finite_depth = depth[finite]
    if finite_pixels.size == 0:
        xy_min = [float('nan'), float('nan')]
        xy_max = [float('nan'), float('nan')]
    else:
        xy_min = [float(x) for x in np.nanmin(finite_pixels, axis=0)]
        xy_max = [float(x) for x in np.nanmax(finite_pixels, axis=0)]
    if finite_depth.size == 0:
        depth_min = float('nan')
        depth_max = float('nan')
    else:
        depth_min = float(np.nanmin(finite_depth))
        depth_max = float(np.nanmax(finite_depth))
    return {
        'num_joints': int(pixels.shape[0]),
        'num_finite': int(finite.sum()),
        'num_front': int(front.sum()),
        'num_in_image': int(in_image.sum()),
        'in_image_ratio': float(in_image.sum() / max(1, pixels.shape[0])),
        'xy_min': xy_min,
        'xy_max': xy_max,
        'depth_min': depth_min,
        'depth_max': depth_max,
    }


def render_pred2d_overlay(
        ds: object,
        path_undistort_frames: str,
        candidate: Frame_Candidate,
        method_name: str,
        path_dump_dir: Path,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
    from hjlib_dataset_std.vis.overlay_2d import sub_camera_one_frame
    from hjlib_dataset_std.vis.person_color import get_person_color_bgr
    from hjlib_vis_2d import vis_points

    image_out = render_raw_overlay(path_undistort_frames, candidate, method_name=method_name)
    height, width = image_out.shape[:2]
    camera = ds.get_camera_by_name_scene(candidate.scene)
    sub_camera = sub_camera_one_frame(camera, candidate.frame_index)
    joints_by_person = load_prediction_joints_for_candidate(candidate, path_dump_dir)

    stats_by_person: Dict[str, Any] = {}
    for id_person, joints_world in sorted(joints_by_person.items()):
        pixels_batch, depth_batch = sub_camera.project_world_points(joints_world[None])
        pixels = np.asarray(pixels_batch[0], dtype=np.float64)
        depth = np.asarray(depth_batch[0], dtype=np.float64)
        stats = projection_stats(pixels, depth, (height, width))
        stats_by_person[str(id_person)] = stats

        visible = (
            np.isfinite(pixels).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0.05)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < float(width))
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < float(height))
        )
        color = get_person_color_bgr(id_person)
        if bool(visible.any()):
            vis_points(
                image_out, pixels[visible], radius=4,
                list_colors=[color])
            center = np.nanmean(pixels[visible], axis=0)
            cv2.putText(
                image_out, 'p%d' % id_person,
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    frame_stats = {
        'method': method_name,
        'scene': candidate.scene,
        'frame_index': candidate.frame_index,
        'image_width': int(width),
        'image_height': int(height),
        'persons': stats_by_person,
    }
    return image_out, frame_stats


def render_overlay(ds: object, candidate: Frame_Candidate) -> np.ndarray:
    from hjlib_vis_2d import vis_bbox, vis_points
    from hjlib_dataset_std.vis.overlay_2d import draw_ground_grid_2d, sub_camera_one_frame
    from hjlib_dataset_std.vis.person_color import get_person_color_bgr

    camera = ds.get_camera_by_name_scene(candidate.scene)
    streamer = ds.get_streamer_by_name_scene(candidate.scene)
    image_out = np.ascontiguousarray(
        np.asarray(streamer.get_frame_certain_index(candidate.frame_index), dtype=np.uint8))

    draw_ground_grid_2d(
        image_out,
        sub_camera_one_frame(camera, candidate.frame_index),
        ds.get_ground_param_by_name_scene(candidate.scene),
    )

    kpts_label, bbox_label = ds.get_keypoints_2d_and_bbox_by_name_scene(candidate.scene)
    kpts_frame, kpts_ids = kpts_label.get_one_frame(candidate.frame_index)
    bbox_frame, bbox_ids = bbox_label.get_one_frame(candidate.frame_index)
    wanted = set(candidate.person_ids)

    for person_kpts, id_person in zip(kpts_frame, kpts_ids):
        if int(id_person) not in wanted:
            continue
        color = get_person_color_bgr(int(id_person))
        vis_points(image_out, np.asarray(person_kpts, dtype=np.float64), radius=3,
                   list_colors=[color])
    for person_bbox, id_person in zip(bbox_frame, bbox_ids):
        if int(id_person) not in wanted:
            continue
        color = get_person_color_bgr(int(id_person))
        vis_bbox(image_out, np.asarray(person_bbox, dtype=np.float64), color=color, thickness=3)
        h1, _h2, w1, _w2 = np.asarray(person_bbox, dtype=np.float64)
        cv2.putText(
            image_out, 'p%d' % int(id_person), (int(w1), max(20, int(h1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    cv2.putText(
        image_out,
        '%s frame %d, a00-covered persons=%s'
        % (candidate.scene, candidate.frame_index, ','.join('%d' % x for x in candidate.person_ids)),
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return image_out


def render_raw_overlay(
        path_undistort_frames: str,
        candidate: Frame_Candidate,
        method_name: str | None = None,
    ) -> np.ndarray:
    path_scene = Path(path_undistort_frames) / candidate.scene
    path_frame = path_scene / ('%06d.png' % candidate.frame_index)
    if not path_frame.is_file():
        path_frame = path_scene / ('%06d.jpg' % candidate.frame_index)
    assert path_frame.is_file(), 'frame image not found: %s' % path_frame
    image_out = cv2.imread(str(path_frame), cv2.IMREAD_COLOR)
    assert image_out is not None, 'cv2.imread failed: %s' % path_frame
    image_out = np.ascontiguousarray(image_out)

    lines = [
        '%s frame %d' % (candidate.scene, candidate.frame_index),
        'a00 covered persons: %s' % ','.join('%d' % x for x in candidate.person_ids),
        'segments: %d' % candidate.num_covering_segments,
    ]
    if method_name is not None:
        lines.insert(1, 'method: %s' % method_name)
    y = 42
    for line in lines:
        cv2.putText(
            image_out, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            image_out, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y += 40
    return image_out


def write_markdown(chosen: List[Frame_Candidate], path_md: Path) -> None:
    lines = [
        '# Selected WorldPose Frames',
        '',
        'Anchor: `ours__ablation_a00_hvip_hipmid_K1_ep0004`',
        '',
        '| scene | frame | persons | abla v1 | v18agg |',
        '| --- | ---: | --- | --- | --- |',
    ]
    for cand in chosen:
        filename = '%s__frame_%07d.png' % (cand.scene, cand.frame_index)
        lines.append(
            '| `%s` | %d | `%s` | [overlay](overlays/abla_v1/%s) | [overlay](overlays/v18agg/%s) |'
            % (
                cand.scene,
                cand.frame_index,
                ','.join('%d' % x for x in cand.person_ids),
                filename,
                filename,
            ))
    path_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(
        num_frames: Annotated[int, typer.Option(help='Number of diverse frames to select.')] = DEFAULT_NUM_FRAMES,
        min_persons: Annotated[int, typer.Option(help='Minimum covered people required per candidate frame.')] = DEFAULT_MIN_PERSONS,
        render: Annotated[bool, typer.Option(help='Also render original-frame GT overlay PNGs.')] = False,
        render_mode: Annotated[Literal['raw', 'gt', 'pred2d'], typer.Option(help='Overlay rendering mode.')] = 'raw',
    ) -> None:
    ensure_import_paths()

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)

    path_selection_dump = METHOD_DUMP_DIRS[SELECTION_METHOD]
    segments = load_segments(path_selection_dump)
    candidates = candidate_frames_from_segments(segments, min_persons)
    assert candidates, 'no candidates found at min_persons=%d' % min_persons
    chosen = choose_diverse_candidates(candidates, num_frames)

    if render:
        path_root, path_undistort = resolve_worldpose_paths()
        assert path_undistort is not None, 'render needs an undistorted frame directory'
        ds = None
        if render_mode in ('gt', 'pred2d'):
            from hjlib_dataset_std.datasets.worldpose.worldpose import WorldPose_Std
            ds = WorldPose_Std(path_data_root=path_root, path_undistort_frames=path_undistort)
        projection_stats_by_method: Dict[str, List[Dict[str, Any]]] = {}
        if render_mode == 'pred2d':
            assert ds is not None
            for method_name, path_dump_dir in METHOD_DUMP_DIRS.items():
                method_overlay_root = OVERLAY_ROOT / method_name
                method_overlay_root.mkdir(parents=True, exist_ok=True)
                method_stats: List[Dict[str, Any]] = []
                for cand in chosen:
                    image, stats = render_pred2d_overlay(
                        ds, path_undistort, cand, method_name, path_dump_dir)
                    method_stats.append(stats)
                    path_png = method_overlay_root / (
                        '%s__frame_%07d.png' % (cand.scene, cand.frame_index))
                    ok = cv2.imwrite(str(path_png), image)
                    assert ok, 'cv2.imwrite failed: %s' % path_png
                projection_stats_by_method[method_name] = method_stats
        else:
            for cand in chosen:
                if render_mode == 'gt':
                    assert ds is not None
                    image = render_overlay(ds, cand)
                else:
                    image = render_raw_overlay(path_undistort, cand)
                path_png = OVERLAY_ROOT / ('%s__frame_%07d.png' % (cand.scene, cand.frame_index))
                ok = cv2.imwrite(str(path_png), image)
                assert ok, 'cv2.imwrite failed: %s' % path_png
        if render_mode == 'pred2d':
            for method_name, method_stats in projection_stats_by_method.items():
                path_stats = ARTIFACT_ROOT / ('projection_stats_%s.json' % method_name)
                path_stats.write_text(
                    json.dumps(method_stats, indent=2, sort_keys=True),
                    encoding='utf-8')

    path_json = ARTIFACT_ROOT / 'selected_worldpose_frames.json'
    path_json.write_text(
        json.dumps([asdict(cand) for cand in chosen], indent=2, sort_keys=True),
        encoding='utf-8')
    write_markdown(chosen, ARTIFACT_ROOT / 'selected_worldpose_frames.md')

    print('selected %d frames' % len(chosen))
    print('json: %s' % path_json)
    print('md: %s' % (ARTIFACT_ROOT / 'selected_worldpose_frames.md'))
    if render:
        print('overlays: %s' % OVERLAY_ROOT)
    if render and render_mode == 'pred2d':
        for method_name in METHOD_DUMP_DIRS:
            print('projection stats %s: %s' % (
                method_name, ARTIFACT_ROOT / ('projection_stats_%s.json' % method_name)))


if __name__ == '__main__':
    typer.run(main)
