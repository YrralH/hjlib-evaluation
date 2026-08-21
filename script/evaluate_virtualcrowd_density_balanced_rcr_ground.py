'''Run one KDE density-balanced RCR arm on VirtualCrowd detections.'''

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import typer

from evaluate_virtualcrowd_rcr_ground import (
    load_scene_ground_effect_support,
    scene_camera_K,
)
from hjlib_dataset_std import VirtualCrowd_Std
from hjlib_detection import load_tracked_scene
from hjlib_evaluation import (
    Ground_Effect_Decomposition,
    Ground_Effect_Support,
    Ground_Estimation_Result,
    Ground_Observation_Set,
    Ground_Plane_Diagnostics,
    collect_ground_observations,
    compute_ground_effect_decomposition,
    compute_ground_plane_diagnostics,
    compute_same_ray_ground_errors,
    estimate_ground_from_observations,
    summarize_ground_errors,
)
from hjlib_ground_solver import (
    Ground_Observation_KDE_Density,
    compute_ground_observation_kde_density,
    solve_ground_param_by_top_bottom_given_K,
)


TOP_JOINT_PAIR = (5, 6)
BOTTOM_JOINT_PAIR = (15, 16)
CONFIDENCE_THRESHOLD = 4.0
MAXIMUM_ANKLE_BBOX_WIDTH_RATIO = 0.20
H_PRIOR_M = 1.35
MINIMUM_PRE_NORMALIZATION_WEIGHT = 0.25
MAXIMUM_PRE_NORMALIZATION_WEIGHT = 4.0
VARIANT_UNWEIGHTED = 'filtered_unweighted'
VARIANT_KDE = 'density_kde_scott_loo'
VARIANTS = (VARIANT_UNWEIGHTED, VARIANT_KDE)
CARTESIAN_CONFIDENCE_THRESHOLDS = (4.0, 4.5, 5.0)
CARTESIAN_ANKLE_RATIO_THRESHOLDS = (0.15, 0.20)
EXPECTED_COUNTS = {
    'scene1': 885,
    'scene1_view2': 1_397,
    'scene2': 3_370,
    'scene2_view2': 1_486,
    'scene3': 1_638,
    'scene3_view2': 1_220,
    'scene4': 3_046,
    'scene4_view2': 4_950,
}
EXPECTED_EFFECT_SUPPORT_TOTAL = 167_243
COMMON_NPZ_KEYS = {
    'selected_frame_index',
    'selected_person_id',
    'top_xy_px',
    'bottom_xy_px',
    'quality',
    'bottom_pair_bbox_width_ratio',
    'plane_camera_abcd',
    'rcr_objective',
    'effect_frame_id',
    'effect_gt_track_id',
    'ground_effect_error_m',
    'normal_oracle_error_m',
    'distance_only_error_m',
    'normalized_pred_plane_camera_abcd',
    'normalized_gt_plane_camera_abcd',
    'normal_angle_deg',
    'distance_ratio',
    'oracle_distance_m',
}
KDE_NPZ_KEYS = {
    'provisional_unit_plane_xy',
    'kernel_covariance_unit_plane',
    'loo_log_density_per_unit_area',
    'log_relative_inverse_density',
    'clipped_relative_inverse_density',
    'normalized_observation_weights',
    'scott_bandwidth_factor',
    'minimum_pre_normalization_weight',
    'maximum_pre_normalization_weight',
    'weight_normalization_factor',
    'effective_sample_size',
}


def threshold_token(value: float, decimal_places: int) -> str:
    return (('%.*f' % (decimal_places, value)).replace('.', 'p'))


@dataclass(frozen=True, slots=True)
class VirtualCrowd_RCR_Cartesian_Config:
    '''One canonical VirtualCrowd selection and density configuration.'''

    name: str
    population_name: str
    confidence_threshold_strict_gt: float
    maximum_ankle_bbox_width_ratio_strict_lt: float
    density_mode: str

    def __post_init__(self) -> None:
        confidence = self.confidence_threshold_strict_gt
        ankle_ratio = self.maximum_ankle_bbox_width_ratio_strict_lt
        density_mode = self.density_mode
        if type(confidence) is not float or confidence not in CARTESIAN_CONFIDENCE_THRESHOLDS:
            raise ValueError('confidence threshold is outside the Cartesian axis')
        if type(ankle_ratio) is not float or ankle_ratio not in CARTESIAN_ANKLE_RATIO_THRESHOLDS:
            raise ValueError('ankle-ratio threshold is outside the Cartesian axis')
        if density_mode not in VARIANTS:
            raise ValueError('density mode is outside the Cartesian axis')
        population_name = 'conf_gt_%s__ankle_lt_%s' % (
            threshold_token(confidence, 1),
            threshold_token(ankle_ratio, 2),
        )
        if self.population_name != population_name:
            raise ValueError('population_name is inconsistent with thresholds')
        if self.name != '%s__%s' % (population_name, density_mode):
            raise ValueError('config name is inconsistent with its fields')


def virtualcrowd_rcr_cartesian_configs(
    ) -> tuple[VirtualCrowd_RCR_Cartesian_Config, ...]:
    '''Return the reviewed 12 Cartesian configs in canonical order.'''
    configs = tuple(
        VirtualCrowd_RCR_Cartesian_Config(
            name='%s__%s' % (population_name, density_mode),
            population_name=population_name,
            confidence_threshold_strict_gt=confidence,
            maximum_ankle_bbox_width_ratio_strict_lt=ankle_ratio,
            density_mode=density_mode,
        )
        for confidence in CARTESIAN_CONFIDENCE_THRESHOLDS
        for ankle_ratio in CARTESIAN_ANKLE_RATIO_THRESHOLDS
        for density_mode in VARIANTS
        for population_name in (
            'conf_gt_%s__ankle_lt_%s' % (
                threshold_token(confidence, 1),
                threshold_token(ankle_ratio, 2),
            ),
        )
    )
    if len(configs) != 12 or len({config.name for config in configs}) != 12:
        raise RuntimeError('Cartesian config names are not exactly 12 unique values')
    populations: dict[str, list[str]] = {}
    for config in configs:
        populations.setdefault(config.population_name, []).append(config.density_mode)
    if (
            len(populations) != 6
            or any(tuple(modes) != VARIANTS for modes in populations.values())
        ):
        raise RuntimeError('Cartesian populations do not map to two density modes')
    return configs


def weighted_estimator(
        weights: NDArray[np.float64],
    ) -> Callable[
        [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        tuple[np.ndarray, np.ndarray],
    ]:
    def estimate(
            top_xy_px: NDArray[np.float64],
            bottom_xy_px: NDArray[np.float64],
            K: NDArray[np.float64],
        ) -> tuple[np.ndarray, np.ndarray]:
        return solve_ground_param_by_top_bottom_given_K(
            top_xy_px,
            bottom_xy_px,
            K,
            H_prior=H_PRIOR_M,
            observation_weights=weights,
        )

    return estimate


def density_arrays(
        density: Ground_Observation_KDE_Density,
    ) -> dict[str, np.ndarray]:
    return {
        'provisional_unit_plane_xy': density.provisional_unit_plane_xy,
        'kernel_covariance_unit_plane': density.kernel_covariance_unit_plane,
        'loo_log_density_per_unit_area': density.loo_log_density_per_unit_area,
        'log_relative_inverse_density': density.log_relative_inverse_density,
        'clipped_relative_inverse_density': density.clipped_relative_inverse_density,
        'normalized_observation_weights': density.normalized_observation_weights,
        'scott_bandwidth_factor': np.asarray(
            density.scott_bandwidth_factor,
            dtype=np.float64,
        ),
        'minimum_pre_normalization_weight': np.asarray(
            density.minimum_pre_normalization_weight,
            dtype=np.float64,
        ),
        'maximum_pre_normalization_weight': np.asarray(
            density.maximum_pre_normalization_weight,
            dtype=np.float64,
        ),
        'weight_normalization_factor': np.asarray(
            density.weight_normalization_factor,
            dtype=np.float64,
        ),
        'effective_sample_size': np.asarray(
            density.effective_sample_size,
            dtype=np.float64,
        ),
    }


def scene_result_summary(
        observations: Ground_Observation_Set,
        result: Ground_Estimation_Result,
        density: Ground_Observation_KDE_Density | None,
        diagnostics: Ground_Plane_Diagnostics,
        headline_error_m: NDArray[np.float64],
        decomposition: Ground_Effect_Decomposition,
    ) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'selected_count': observations.count,
        'plane_camera_abcd': result.plane_camera_abcd.tolist(),
        'rcr_objective': result.objective,
        'normalized_pred_plane_camera_abcd': (
            diagnostics.normalized_pred_plane_camera_abcd.tolist()
        ),
        'normalized_gt_plane_camera_abcd': (
            diagnostics.normalized_gt_plane_camera_abcd.tolist()
        ),
        'normal_angle_deg': diagnostics.normal_angle_deg,
        'distance_ratio': diagnostics.distance_ratio,
        'oracle_distance_m': decomposition.oracle_distance_m,
        'ground_effect': summarize_ground_errors(headline_error_m),
        'normal_oracle': summarize_ground_errors(
            decomposition.normal_oracle_error_m,
        ),
        'distance_only': summarize_ground_errors(
            decomposition.distance_only_error_m,
        ),
    }
    if density is not None:
        summary['density'] = {
            'algorithm': 'gaussian_kde_scott_exact_loo',
            'scott_bandwidth_factor': density.scott_bandwidth_factor,
            'effective_sample_size': density.effective_sample_size,
            'minimum_weight': float(np.min(density.normalized_observation_weights)),
            'maximum_weight': float(np.max(density.normalized_observation_weights)),
            'minimum_pre_normalization_weight': (
                density.minimum_pre_normalization_weight
            ),
            'maximum_pre_normalization_weight': (
                density.maximum_pre_normalization_weight
            ),
            'weight_normalization_factor': density.weight_normalization_factor,
        }
    return summary


def scene_payload(
        observations: Ground_Observation_Set,
        result: Ground_Estimation_Result,
        support: Ground_Effect_Support,
        density: Ground_Observation_KDE_Density | None,
        diagnostics: Ground_Plane_Diagnostics,
        headline_error_m: NDArray[np.float64],
        decomposition: Ground_Effect_Decomposition,
    ) -> dict[str, np.ndarray]:
    ratio = observations.bottom_pair_bbox_width_ratio
    if ratio is None:
        raise ValueError('density-balanced observations must retain bbox ratio')
    arrays: dict[str, np.ndarray] = {
        'selected_frame_index': observations.frame_indices,
        'selected_person_id': observations.person_ids,
        'top_xy_px': observations.top_xy_px,
        'bottom_xy_px': observations.bottom_xy_px,
        'quality': observations.quality,
        'bottom_pair_bbox_width_ratio': ratio,
        'plane_camera_abcd': result.plane_camera_abcd,
        'rcr_objective': np.asarray(result.objective, dtype=np.float64),
        'effect_frame_id': support.frame_ids,
        'effect_gt_track_id': support.gt_track_ids,
        'ground_effect_error_m': headline_error_m,
        'normal_oracle_error_m': decomposition.normal_oracle_error_m,
        'distance_only_error_m': decomposition.distance_only_error_m,
        'normalized_pred_plane_camera_abcd': (
            diagnostics.normalized_pred_plane_camera_abcd
        ),
        'normalized_gt_plane_camera_abcd': diagnostics.normalized_gt_plane_camera_abcd,
        'normal_angle_deg': np.asarray(diagnostics.normal_angle_deg, dtype=np.float64),
        'distance_ratio': np.asarray(diagnostics.distance_ratio, dtype=np.float64),
        'oracle_distance_m': np.asarray(
            decomposition.oracle_distance_m,
            dtype=np.float64,
        ),
    }
    if density is not None:
        arrays.update(density_arrays(density))
    return arrays


def evaluate_scene(
        observations: Ground_Observation_Set,
        K: NDArray[np.float64],
        support: Ground_Effect_Support,
        gt_plane: NDArray[np.float64],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, dict[str, Any]],
    ]:
    unweighted = estimate_ground_from_observations(observations, K)
    density = compute_ground_observation_kde_density(
        observations.bottom_xy_px,
        K,
        unweighted.plane_camera_abcd[:3],
        minimum_pre_normalization_weight=MINIMUM_PRE_NORMALIZATION_WEIGHT,
        maximum_pre_normalization_weight=MAXIMUM_PRE_NORMALIZATION_WEIGHT,
    )
    weighted = estimate_ground_from_observations(
        observations,
        K,
        weighted_estimator(density.normalized_observation_weights),
    )
    results: dict[
        str,
        tuple[Ground_Estimation_Result, Ground_Observation_KDE_Density | None],
    ] = {
        VARIANT_UNWEIGHTED: (unweighted, None),
        VARIANT_KDE: (weighted, density),
    }
    payloads: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for variant, (result, density_record) in results.items():
        diagnostics = compute_ground_plane_diagnostics(result.plane_camera_abcd, gt_plane)
        headline_error = compute_same_ray_ground_errors(
            support,
            K,
            result.plane_camera_abcd,
        )
        decomposition = compute_ground_effect_decomposition(
            support,
            K,
            result.plane_camera_abcd,
            gt_plane,
        )
        payloads[variant] = scene_payload(
            observations,
            result,
            support,
            density_record,
            diagnostics,
            headline_error,
            decomposition,
        )
        summaries[variant] = scene_result_summary(
            observations,
            result,
            density_record,
            diagnostics,
            headline_error,
            decomposition,
        )
    return payloads, summaries


def load_scene_inputs(
        dataset: VirtualCrowd_Std,
        tracked_root: Path,
        support_root: Path,
        scene: str,
    ) -> tuple[
        Ground_Observation_Set,
        NDArray[np.float64],
        Ground_Effect_Support,
        NDArray[np.float64],
    ]:
    tracked_scene = load_tracked_scene(
        tracked_root / ('%s_tracked_scene.bin' % scene),
    )
    observations = collect_ground_observations(
        tracked_scene,
        TOP_JOINT_PAIR,
        BOTTOM_JOINT_PAIR,
        CONFIDENCE_THRESHOLD,
        maximum_bottom_pair_bbox_width_ratio=MAXIMUM_ANKLE_BBOX_WIDTH_RATIO,
    )
    if observations.count != EXPECTED_COUNTS[scene]:
        raise ValueError('reviewed observation count changed for %s' % scene)
    K = scene_camera_K(dataset, scene, tracked_scene.num_frame)
    support = load_scene_ground_effect_support(support_root, scene)
    gt_plane = dataset.get_ground_param_by_name_scene(scene).astype(np.float64)
    return observations, K, support, gt_plane


def compute_complete_result(
        dataset: VirtualCrowd_Std,
        tracked_root: Path,
        support_root: Path,
        scenes: list[str],
    ) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, np.ndarray]]]:
    summary: dict[str, Any] = {
        'baseline': 'single-arm exact-LOO KDE density-balanced RCR',
        'candidate_headline_variant': VARIANT_KDE,
        'confidence_threshold_strict_gt': CONFIDENCE_THRESHOLD,
        'maximum_ankle_bbox_width_ratio_strict_lt': MAXIMUM_ANKLE_BBOX_WIDTH_RATIO,
        'top_joint_pair': list(TOP_JOINT_PAIR),
        'bottom_joint_pair': list(BOTTOM_JOINT_PAIR),
        'h_prior_m': H_PRIOR_M,
        'selected_total': 0,
        'effect_support_total': 0,
        'variants': {variant: {'scenes': {}} for variant in VARIANTS},
    }
    payloads: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    global_errors: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        variant: {'ground_effect': [], 'normal_oracle': [], 'distance_only': []}
        for variant in VARIANTS
    }
    selected_total = 0
    support_total = 0
    variant_summaries = cast(dict[str, Any], summary['variants'])
    for scene in scenes:
        observations, K, support, gt_plane = load_scene_inputs(
            dataset,
            tracked_root,
            support_root,
            scene,
        )
        scene_payloads, scene_summaries = evaluate_scene(
            observations,
            K,
            support,
            gt_plane,
        )
        selected_total += observations.count
        support_total += support.count
        for variant in VARIANTS:
            payload = scene_payloads[variant]
            payloads[(variant, scene)] = payload
            cast(dict[str, Any], variant_summaries[variant]['scenes'])[scene] = (
                scene_summaries[variant]
            )
            global_errors[variant]['ground_effect'].append(
                cast(NDArray[np.float64], payload['ground_effect_error_m']),
            )
            global_errors[variant]['normal_oracle'].append(
                cast(NDArray[np.float64], payload['normal_oracle_error_m']),
            )
            global_errors[variant]['distance_only'].append(
                cast(NDArray[np.float64], payload['distance_only_error_m']),
            )
    if selected_total != sum(EXPECTED_COUNTS.values()):
        raise ValueError('reviewed total observation count changed')
    if support_total != EXPECTED_EFFECT_SUPPORT_TOTAL:
        raise ValueError('frozen ground-effect support count changed')
    summary['selected_total'] = selected_total
    summary['effect_support_total'] = support_total
    for variant in VARIANTS:
        variant_summary = cast(dict[str, Any], variant_summaries[variant])
        for name, arrays in global_errors[variant].items():
            variant_summary[name] = summarize_ground_errors(np.concatenate(arrays))
    return summary, payloads


def write_plain_result(
        output_root: Path,
        scenes: list[str],
        config_names: tuple[str, ...],
        summary: dict[str, Any],
        payloads: dict[tuple[str, str], dict[str, np.ndarray]],
    ) -> None:
    '''Write one plain summary plus one numeric NPZ per config and scene.'''
    output_root.mkdir(parents=True)
    for config_name in config_names:
        for scene in scenes:
            path = output_root / config_name / ('%s.npz' % scene)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **cast(dict[str, Any], payloads[(config_name, scene)]))
    (output_root / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def validate_plain_result(
        output_root: Path,
        scenes: list[str],
        config_names: tuple[str, ...],
        expected_keys_by_config: dict[str, set[str]],
        expected_summary: dict[str, Any],
        expected_payloads: dict[tuple[str, str], dict[str, np.ndarray]],
    ) -> None:
    '''Compare one plain artifact exactly against reconstructed expectations.'''
    if set(expected_keys_by_config) != set(config_names):
        raise ValueError('plain-result config key mapping is incomplete')
    expected_files = {Path('summary.json')}
    expected_files.update(
        Path(config_name) / ('%s.npz' % scene)
        for config_name in config_names
        for scene in scenes
    )
    actual_files = {
        path.relative_to(output_root)
        for path in output_root.rglob('*')
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError('density-balanced result has unexpected output files')
    loaded_summary = json.loads(
        (output_root / 'summary.json').read_text(encoding='utf-8'),
    )
    if loaded_summary != expected_summary:
        raise ValueError('reloaded summary differs from source reconstruction')
    for config_name in config_names:
        expected_keys = expected_keys_by_config[config_name]
        for scene in scenes:
            with np.load(
                    output_root / config_name / ('%s.npz' % scene),
                    allow_pickle=False,
                ) as loaded:
                if set(loaded.files) != expected_keys:
                    raise ValueError(
                        'scene NPZ key set is invalid: %s/%s'
                        % (config_name, scene),
                    )
                expected = expected_payloads[(config_name, scene)]
                for name in expected_keys:
                    if not np.array_equal(loaded[name], expected[name]):
                        raise ValueError(
                            'scene payload differs from source reconstruction: '
                            '%s/%s/%s' % (config_name, scene, name),
                        )


def validate_written_results(
        output_root: Path,
        path_dataset_root: Path,
        path_tracked_scene_root: Path,
        path_ground_effect_support_root: Path,
    ) -> dict[str, Any]:
    dataset = VirtualCrowd_Std(str(path_dataset_root.resolve(strict=True)))
    tracked_root = path_tracked_scene_root.resolve(strict=True)
    support_root = path_ground_effect_support_root.resolve(strict=True)
    scenes = dataset.get_list_scene_names()
    if scenes != sorted(EXPECTED_COUNTS) or len(scenes) != 8:
        raise ValueError('VirtualCrowd scenes differ from reviewed population')
    expected_summary, expected_payloads = compute_complete_result(
        dataset,
        tracked_root,
        support_root,
        scenes,
    )
    expected_keys_by_config = {
        variant: (
            COMMON_NPZ_KEYS
            if variant == VARIANT_UNWEIGHTED
            else COMMON_NPZ_KEYS | KDE_NPZ_KEYS
        )
        for variant in VARIANTS
    }
    validate_plain_result(
        output_root,
        scenes,
        VARIANTS,
        expected_keys_by_config,
        expected_summary,
        expected_payloads,
    )
    return expected_summary


def dry_run_summary(
        tracked_root: Path,
        scenes: list[str],
    ) -> dict[str, Any]:
    summary: dict[str, Any] = {
        'confidence_threshold_strict_gt': CONFIDENCE_THRESHOLD,
        'maximum_ankle_bbox_width_ratio_strict_lt': MAXIMUM_ANKLE_BBOX_WIDTH_RATIO,
        'selected_total': 0,
        'scenes': {},
    }
    selected_total = 0
    scene_summaries = cast(dict[str, Any], summary['scenes'])
    for scene in scenes:
        tracked_scene = load_tracked_scene(
            tracked_root / ('%s_tracked_scene.bin' % scene),
        )
        observations = collect_ground_observations(
            tracked_scene,
            TOP_JOINT_PAIR,
            BOTTOM_JOINT_PAIR,
            CONFIDENCE_THRESHOLD,
            maximum_bottom_pair_bbox_width_ratio=MAXIMUM_ANKLE_BBOX_WIDTH_RATIO,
        )
        if observations.count != EXPECTED_COUNTS[scene]:
            raise ValueError('reviewed observation count changed for %s' % scene)
        selected_total += observations.count
        scene_summaries[scene] = {'selected_count': observations.count}
    if selected_total != sum(EXPECTED_COUNTS.values()):
        raise ValueError('reviewed total observation count changed')
    summary['selected_total'] = selected_total
    return summary


def run_virtualcrowd_density_balanced_rcr_ground(
        *,
        path_dataset_root: Path,
        path_tracked_scene_root: Path,
        path_ground_effect_support_root: Path,
        path_output_root: Path,
        dry_run: bool = False,
    ) -> dict[str, Any]:
    '''Run the reviewed unweighted and exact-LOO KDE single arm.'''
    dataset_root = path_dataset_root.resolve(strict=True)
    tracked_root = path_tracked_scene_root.resolve(strict=True)
    support_root = path_ground_effect_support_root.resolve(strict=True)
    if not tracked_root.is_dir() or not support_root.is_dir():
        raise NotADirectoryError('tracked-scene and support roots must be directories')
    if path_output_root.exists():
        raise FileExistsError('output root already exists: %s' % path_output_root)
    dataset = VirtualCrowd_Std(str(dataset_root))
    scenes = dataset.get_list_scene_names()
    if scenes != sorted(EXPECTED_COUNTS) or len(scenes) != 8:
        raise ValueError('VirtualCrowd scenes differ from reviewed population')
    if dry_run:
        return dry_run_summary(tracked_root, scenes)

    summary, payloads = compute_complete_result(dataset, tracked_root, support_root, scenes)
    write_plain_result(
        path_output_root,
        scenes,
        VARIANTS,
        summary,
        payloads,
    )
    validate_written_results(
        path_output_root,
        dataset_root,
        tracked_root,
        support_root,
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
        dry_run: bool = typer.Option(False),
    ) -> None:
    '''Estimate and evaluate one KDE density-balanced VirtualCrowd RCR arm.'''
    summary = run_virtualcrowd_density_balanced_rcr_ground(
        path_dataset_root=path_dataset_root,
        path_tracked_scene_root=path_tracked_scene_root,
        path_ground_effect_support_root=path_ground_effect_support_root,
        path_output_root=path_output_root,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    typer.run(main)
