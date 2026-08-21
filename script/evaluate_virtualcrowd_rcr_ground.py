'''Run the VirtualCrowd fixed-height RCR ground baseline.'''

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import typer

from hjlib_dataset_std import VirtualCrowd_Std
from hjlib_detection import load_tracked_scene
from hjlib_evaluation import (
    Ground_Effect_Support,
    Ground_Estimation_Result,
    Ground_Observation_Set,
    collect_ground_observations,
    compute_same_ray_ground_errors,
    estimate_ground_from_observations,
    sample_ground_observations,
    select_ground_observations_at_frame,
    summarize_ground_errors,
)


TOP_JOINT_PAIR = (5, 6)
BOTTOM_JOINT_PAIR = (15, 16)
CONFIDENCE_THRESHOLD = 3.0
SAMPLE_COUNT = 5_000
H_PRIOR_M = 1.35
STRATEGY_FIRST_FRAME = 'first_frame_high_confidence'
STRATEGY_ALL_FRAMES = 'all_frames_high_confidence_sampled_5000'
STRATEGIES = (STRATEGY_FIRST_FRAME, STRATEGY_ALL_FRAMES)
SUPPORT_ARRAY_NAMES = (
    'frame_id',
    'gt_track_id',
    'image_xy_px',
    'gt_intersection_camera_m',
)


def scene_camera_K(
        dataset: VirtualCrowd_Std,
        scene: str,
        num_frame: int,
    ) -> NDArray[np.float64]:
    '''Require one fixed valid camera across the complete tracked scene.'''
    camera = dataset.get_camera_by_name_scene(scene)
    intrinsics = camera.intrinsics_batch
    if intrinsics.B != num_frame:
        raise ValueError('camera batch length differs from tracked scene: %s' % scene)
    if not bool(intrinsics.valid.all()):
        raise ValueError('camera batch contains invalid rows: %s' % scene)
    Ks = intrinsics.Ks
    if not np.array_equal(Ks, np.broadcast_to(Ks[0], Ks.shape)):
        raise ValueError('RCR evaluation requires one fixed K per scene: %s' % scene)
    return cast(NDArray[np.float64], Ks[0].copy())


def load_support_arrays(path: Path) -> dict[str, NDArray[np.generic]]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as loaded:
        missing = set(SUPPORT_ARRAY_NAMES) - set(loaded.files)
        if missing:
            raise ValueError('ground-effect support arrays are missing: %r' % sorted(missing))
        return {name: loaded[name] for name in SUPPORT_ARRAY_NAMES}


def load_scene_ground_effect_support(
        support_root: Path,
        scene: str,
    ) -> Ground_Effect_Support:
    '''Verify two prior methods expose one exact shared ray support.'''
    crowd4d = load_support_arrays(support_root / 'crowd4d' / ('%s.npz' % scene))
    dycrowd = load_support_arrays(support_root / 'dycrowd' / ('%s.npz' % scene))
    for name in SUPPORT_ARRAY_NAMES:
        if not np.array_equal(crowd4d[name], dycrowd[name]):
            raise ValueError('prior ground-effect support differs for %s/%s' % (scene, name))
    return Ground_Effect_Support(
        cast(NDArray[np.int64], crowd4d['frame_id']),
        cast(NDArray[np.int64], crowd4d['gt_track_id']),
        cast(NDArray[np.float64], crowd4d['image_xy_px']),
        cast(NDArray[np.float64], crowd4d['gt_intersection_camera_m']),
    )


def selected_observations(
        candidates: Ground_Observation_Set,
        strategy: str,
        seed: int,
    ) -> Ground_Observation_Set:
    if strategy == STRATEGY_FIRST_FRAME:
        return select_ground_observations_at_frame(candidates, 0)
    if strategy == STRATEGY_ALL_FRAMES:
        return sample_ground_observations(candidates, SAMPLE_COUNT, seed)
    raise ValueError('unknown RCR evaluation strategy: %s' % strategy)


def write_scene_result(
        path: Path,
        result: Ground_Estimation_Result,
        support: Ground_Effect_Support,
        error_m: NDArray[np.float64] | None,
        invalid_reason: str | None,
    ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = error_m is not None
    errors = (
        error_m
        if error_m is not None
        else np.empty((0,), dtype=np.float64)
    )
    np.savez(
        path,
        selected_frame_index=result.observations.frame_indices,
        selected_person_id=result.observations.person_ids,
        top_xy_px=result.observations.top_xy_px,
        bottom_xy_px=result.observations.bottom_xy_px,
        quality=result.observations.quality,
        plane_camera_abcd=result.plane_camera_abcd,
        rcr_objective=np.asarray(result.objective, dtype=np.float64),
        effect_frame_id=support.frame_ids,
        effect_gt_track_id=support.gt_track_ids,
        ground_effect_valid=np.asarray(valid, dtype=np.bool_),
        ground_effect_error_m=errors,
        ground_effect_invalid_reason=np.asarray(invalid_reason or ''),
    )


def run_virtualcrowd_rcr_ground(
        *,
        path_dataset_root: Path,
        path_tracked_scene_root: Path,
        path_ground_effect_support_root: Path,
        path_output_root: Path,
        base_seed: int = 20_260_818,
        dry_run: bool = False,
    ) -> dict[str, Any]:
    '''Run both strategies for all released scenes and optionally write results.'''
    dataset_root = path_dataset_root.resolve(strict=True)
    tracked_root = path_tracked_scene_root.resolve(strict=True)
    support_root = path_ground_effect_support_root.resolve(strict=True)
    if not tracked_root.is_dir() or not support_root.is_dir():
        raise NotADirectoryError('tracked-scene and support roots must be directories')
    if path_output_root.exists():
        raise FileExistsError('output root already exists: %s' % path_output_root)
    if base_seed < 0:
        raise ValueError('base_seed must be nonnegative')

    dataset = VirtualCrowd_Std(str(dataset_root))
    scenes = dataset.get_list_scene_names()
    if scenes != sorted(scenes) or len(scenes) != 8 or len(set(scenes)) != 8:
        raise ValueError('VirtualCrowd provider must expose eight sorted unique scenes')
    summary: dict[str, Any] = {
        'baseline': 'RCR fixed-height baseline under HJ high-confidence protocol',
        'headline_strategy': STRATEGY_ALL_FRAMES,
        'diagnostic_strategy': STRATEGY_FIRST_FRAME,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'top_joint_pair': list(TOP_JOINT_PAIR),
        'bottom_joint_pair': list(BOTTOM_JOINT_PAIR),
        'sample_count_per_scene': SAMPLE_COUNT,
        'h_prior_m': H_PRIOR_M,
        'base_seed': base_seed,
        'strategies': {},
    }
    errors_by_strategy: dict[str, list[NDArray[np.float64]]] = {
        strategy: [] for strategy in STRATEGIES
    }
    invalid_by_strategy: dict[str, list[str]] = {
        strategy: [] for strategy in STRATEGIES
    }
    if not dry_run:
        path_output_root.mkdir(parents=True)

    strategies_summary = cast(dict[str, Any], summary['strategies'])
    for strategy in STRATEGIES:
        strategies_summary[strategy] = {'scenes': {}}

    for scene_rank, scene in enumerate(scenes):
        path_tracked = tracked_root / ('%s_tracked_scene.bin' % scene)
        tracked_scene = load_tracked_scene(path_tracked)
        K = scene_camera_K(dataset, scene, tracked_scene.num_frame)
        support = load_scene_ground_effect_support(support_root, scene)
        candidates = collect_ground_observations(
            tracked_scene,
            TOP_JOINT_PAIR,
            BOTTOM_JOINT_PAIR,
            CONFIDENCE_THRESHOLD,
        )
        for strategy in STRATEGIES:
            selected = selected_observations(
                candidates,
                strategy,
                base_seed + scene_rank,
            )
            scene_summary: dict[str, Any] = {
                'candidate_count': candidates.count,
                'selected_count': selected.count,
            }
            if dry_run:
                cast(dict[str, Any], strategies_summary[strategy]['scenes'])[scene] = scene_summary
                continue
            result = estimate_ground_from_observations(selected, K)
            invalid_reason: str | None = None
            try:
                errors = compute_same_ray_ground_errors(
                    support,
                    K,
                    result.plane_camera_abcd,
                )
            except ValueError as error:
                if strategy != STRATEGY_FIRST_FRAME:
                    raise
                errors = None
                invalid_reason = str(error)
                invalid_by_strategy[strategy].append(scene)
            scene_summary.update({
                'plane_camera_abcd': result.plane_camera_abcd.tolist(),
                'rcr_objective': result.objective,
            })
            if errors is None:
                scene_summary['ground_effect'] = {
                    'status': 'invalid',
                    'support_count': support.count,
                    'reason': invalid_reason,
                }
            else:
                errors_by_strategy[strategy].append(errors)
                scene_summary['ground_effect'] = {
                    'status': 'valid',
                    **summarize_ground_errors(errors),
                }
            cast(dict[str, Any], strategies_summary[strategy]['scenes'])[scene] = scene_summary
            write_scene_result(
                path_output_root / strategy / ('%s.npz' % scene),
                result,
                support,
                errors,
                invalid_reason,
            )

    for strategy in STRATEGIES:
        strategy_summary = cast(dict[str, Any], strategies_summary[strategy])
        if dry_run:
            continue
        invalid_scenes = invalid_by_strategy[strategy]
        if invalid_scenes:
            strategy_summary['global_ground_effect'] = {
                'status': 'invalid',
                'invalid_scenes': invalid_scenes,
                'reason': 'at least one complete scene support is invalid; no partial reduction',
            }
        else:
            global_errors = np.concatenate(errors_by_strategy[strategy])
            strategy_summary['global_ground_effect'] = {
                'status': 'valid',
                **summarize_ground_errors(global_errors),
            }
    if not dry_run:
        (path_output_root / 'summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    return summary


def main(
        path_dataset_root: Path = typer.Option(..., exists=True, file_okay=False),
        path_tracked_scene_root: Path = typer.Option(..., exists=True, file_okay=False),
        path_ground_effect_support_root: Path = typer.Option(
            ...,
            exists=True,
            file_okay=False,
        ),
        path_output_root: Path = typer.Option(...),
        base_seed: int = typer.Option(20_260_818, min=0),
        dry_run: bool = typer.Option(False),
    ) -> None:
    '''Estimate and evaluate VirtualCrowd ground planes from tracked keypoints.'''
    summary = run_virtualcrowd_rcr_ground(
        path_dataset_root=path_dataset_root,
        path_tracked_scene_root=path_tracked_scene_root,
        path_ground_effect_support_root=path_ground_effect_support_root,
        path_output_root=path_output_root,
        base_seed=base_seed,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    typer.run(main)
