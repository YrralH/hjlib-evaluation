'''Prepare or execute the reviewed VirtualCrowd RCR Cartesian grid.'''

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import typer

from evaluate_virtualcrowd_density_balanced_rcr_ground import (
    BOTTOM_JOINT_PAIR,
    CARTESIAN_ANKLE_RATIO_THRESHOLDS,
    CARTESIAN_CONFIDENCE_THRESHOLDS,
    COMMON_NPZ_KEYS,
    EXPECTED_EFFECT_SUPPORT_TOTAL,
    H_PRIOR_M,
    KDE_NPZ_KEYS,
    TOP_JOINT_PAIR,
    VARIANT_UNWEIGHTED,
    VARIANTS,
    VirtualCrowd_RCR_Cartesian_Config,
    evaluate_scene,
    validate_plain_result,
    virtualcrowd_rcr_cartesian_configs,
    write_plain_result,
)
from evaluate_virtualcrowd_rcr_ground import (
    load_scene_ground_effect_support,
    scene_camera_K,
)
from hjlib_dataset_std import VirtualCrowd_Std
from hjlib_detection import Tracked_Scene, load_tracked_scene
from hjlib_evaluation import (
    Ground_Observation_Set,
    collect_ground_observations,
    summarize_ground_errors,
)


SCENES = (
    'scene1',
    'scene1_view2',
    'scene2',
    'scene2_view2',
    'scene3',
    'scene3_view2',
    'scene4',
    'scene4_view2',
)
EXPECTED_CARTESIAN_COUNTS = {
    'conf_gt_4p0__ankle_lt_0p15': (441, 836, 1929, 868, 911, 692, 1586, 2164),
    'conf_gt_4p0__ankle_lt_0p20': (885, 1397, 3370, 1486, 1638, 1220, 3046, 4950),
    'conf_gt_4p5__ankle_lt_0p15': (285, 724, 1229, 638, 787, 549, 759, 1799),
    'conf_gt_4p5__ankle_lt_0p20': (613, 1230, 2307, 1152, 1438, 986, 1678, 4120),
    'conf_gt_5p0__ankle_lt_0p15': (87, 390, 531, 391, 578, 350, 214, 1298),
    'conf_gt_5p0__ankle_lt_0p20': (256, 810, 1099, 761, 1104, 674, 560, 3062),
}


def config_dict(
        config: VirtualCrowd_RCR_Cartesian_Config,
    ) -> dict[str, Any]:
    return {
        'name': config.name,
        'population_name': config.population_name,
        'confidence_threshold_strict_gt': config.confidence_threshold_strict_gt,
        'maximum_ankle_bbox_width_ratio_strict_lt': (
            config.maximum_ankle_bbox_width_ratio_strict_lt
        ),
        'density_mode': config.density_mode,
    }


def population_configs(
        configs: tuple[VirtualCrowd_RCR_Cartesian_Config, ...],
    ) -> tuple[VirtualCrowd_RCR_Cartesian_Config, ...]:
    output: list[VirtualCrowd_RCR_Cartesian_Config] = []
    seen: set[str] = set()
    for config in configs:
        if config.population_name not in seen:
            output.append(config)
            seen.add(config.population_name)
    if len(output) != 6:
        raise ValueError('Cartesian configs must contain exactly six populations')
    return tuple(output)


def observation_identities(
        observations: Ground_Observation_Set,
    ) -> set[tuple[int, int]]:
    identities = set(zip(
        observations.frame_indices.tolist(),
        observations.person_ids.tolist(),
        strict=True,
    ))
    if len(identities) != observations.count:
        raise ValueError('observation population contains duplicate identities')
    return identities


def collect_scene_populations(
        tracked_scene: Tracked_Scene,
        configs: tuple[VirtualCrowd_RCR_Cartesian_Config, ...],
    ) -> dict[str, Ground_Observation_Set]:
    populations: dict[str, Ground_Observation_Set] = {}
    representative_configs = population_configs(configs)
    for config in representative_configs:
        populations[config.population_name] = collect_ground_observations(
            tracked_scene,
            TOP_JOINT_PAIR,
            BOTTOM_JOINT_PAIR,
            config.confidence_threshold_strict_gt,
            maximum_bottom_pair_bbox_width_ratio=(
                config.maximum_ankle_bbox_width_ratio_strict_lt
            ),
        )
    validate_population_nesting(populations, representative_configs)
    return populations


def validate_population_nesting(
        populations: dict[str, Ground_Observation_Set],
        representative_configs: tuple[VirtualCrowd_RCR_Cartesian_Config, ...],
    ) -> None:
    identities = {
        config.population_name: observation_identities(
            populations[config.population_name],
        )
        for config in representative_configs
    }
    by_axes = {
        (
            config.confidence_threshold_strict_gt,
            config.maximum_ankle_bbox_width_ratio_strict_lt,
        ): identities[config.population_name]
        for config in representative_configs
    }
    for ankle_ratio in CARTESIAN_ANKLE_RATIO_THRESHOLDS:
        if not (
                by_axes[(5.0, ankle_ratio)]
                <= by_axes[(4.5, ankle_ratio)]
                <= by_axes[(4.0, ankle_ratio)]
            ):
            raise ValueError('confidence-threshold populations are not nested')
    for confidence in CARTESIAN_CONFIDENCE_THRESHOLDS:
        if not by_axes[(confidence, 0.15)] <= by_axes[(confidence, 0.20)]:
            raise ValueError('ankle-ratio populations are not nested')


def validate_reviewed_population_counts(
        scene: str,
        populations: dict[str, Ground_Observation_Set],
    ) -> None:
    scene_index = SCENES.index(scene)
    if set(populations) != set(EXPECTED_CARTESIAN_COUNTS):
        raise ValueError('Cartesian population names differ from reviewed matrix')
    for population_name, observations in populations.items():
        expected = EXPECTED_CARTESIAN_COUNTS[population_name][scene_index]
        if observations.count != expected:
            raise ValueError(
                'reviewed Cartesian count changed for %s/%s'
                % (scene, population_name),
            )


def prepare_virtualcrowd_density_balanced_rcr_cartesian(
        *,
        path_dataset_root: Path,
        path_tracked_scene_root: Path,
    ) -> dict[str, Any]:
    '''Report the exact Cartesian matrix and six real selection populations.'''
    dataset_root = path_dataset_root.resolve(strict=True)
    tracked_root = path_tracked_scene_root.resolve(strict=True)
    if not tracked_root.is_dir():
        raise NotADirectoryError('tracked-scene root must be a directory')
    dataset = VirtualCrowd_Std(str(dataset_root))
    scenes = dataset.get_list_scene_names()
    if tuple(scenes) != SCENES:
        raise ValueError('VirtualCrowd scenes differ from reviewed population')
    configs = virtualcrowd_rcr_cartesian_configs()
    representatives = population_configs(configs)
    counts_by_population: dict[str, dict[str, Any]] = {
        config.population_name: {
            'population_name': config.population_name,
            'confidence_threshold_strict_gt': (
                config.confidence_threshold_strict_gt
            ),
            'maximum_ankle_bbox_width_ratio_strict_lt': (
                config.maximum_ankle_bbox_width_ratio_strict_lt
            ),
            'selected_total': 0,
            'scene_counts': {},
        }
        for config in representatives
    }
    for scene in scenes:
        tracked_scene = load_tracked_scene(
            tracked_root / ('%s_tracked_scene.bin' % scene),
        )
        populations = collect_scene_populations(tracked_scene, configs)
        validate_reviewed_population_counts(scene, populations)
        for config in representatives:
            count = populations[config.population_name].count
            record = counts_by_population[config.population_name]
            cast(dict[str, int], record['scene_counts'])[scene] = count
            record['selected_total'] = cast(int, record['selected_total']) + count
    return {
        'mode': 'cartesian_prepare_only',
        'configuration_count': len(configs),
        'population_count': len(representatives),
        'configs': [config_dict(config) for config in configs],
        'populations': [
            counts_by_population[config.population_name]
            for config in representatives
        ],
    }


def compute_cartesian_result(
        dataset: VirtualCrowd_Std,
        tracked_root: Path,
        support_root: Path,
        scenes: list[str],
    ) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, np.ndarray]]]:
    configs = virtualcrowd_rcr_cartesian_configs()
    representatives = population_configs(configs)
    config_by_population_mode = {
        (config.population_name, config.density_mode): config
        for config in configs
    }
    summary: dict[str, Any] = {
        'baseline': 'VirtualCrowd exact-LOO KDE Cartesian RCR',
        'configuration_count': len(configs),
        'population_count': len(representatives),
        'h_prior_m': H_PRIOR_M,
        'effect_support_total': 0,
        'configs': {
            config.name: {**config_dict(config), 'selected_total': 0, 'scenes': {}}
            for config in configs
        },
    }
    payloads: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    global_errors: dict[str, dict[str, list[NDArray[np.float64]]]] = {
        config.name: {
            'ground_effect': [],
            'normal_oracle': [],
            'distance_only': [],
        }
        for config in configs
    }
    support_total = 0
    config_summaries = cast(dict[str, Any], summary['configs'])
    for scene in scenes:
        tracked_scene = load_tracked_scene(
            tracked_root / ('%s_tracked_scene.bin' % scene),
        )
        populations = collect_scene_populations(tracked_scene, configs)
        validate_reviewed_population_counts(scene, populations)
        K = scene_camera_K(dataset, scene, tracked_scene.num_frame)
        support = load_scene_ground_effect_support(support_root, scene)
        gt_plane = dataset.get_ground_param_by_name_scene(scene).astype(np.float64)
        support_total += support.count
        for representative in representatives:
            observations = populations[representative.population_name]
            scene_payloads, scene_summaries = evaluate_scene(
                observations,
                K,
                support,
                gt_plane,
            )
            for density_mode in VARIANTS:
                config = config_by_population_mode[
                    (representative.population_name, density_mode)
                ]
                payload = scene_payloads[density_mode]
                payloads[(config.name, scene)] = payload
                config_summary = cast(dict[str, Any], config_summaries[config.name])
                cast(dict[str, Any], config_summary['scenes'])[scene] = (
                    scene_summaries[density_mode]
                )
                config_summary['selected_total'] = (
                    int(config_summary['selected_total']) + observations.count
                )
                global_errors[config.name]['ground_effect'].append(
                    cast(NDArray[np.float64], payload['ground_effect_error_m']),
                )
                global_errors[config.name]['normal_oracle'].append(
                    cast(NDArray[np.float64], payload['normal_oracle_error_m']),
                )
                global_errors[config.name]['distance_only'].append(
                    cast(NDArray[np.float64], payload['distance_only_error_m']),
                )
    if support_total != EXPECTED_EFFECT_SUPPORT_TOTAL:
        raise ValueError('frozen ground-effect support count changed')
    summary['effect_support_total'] = support_total
    for config in configs:
        config_summary = cast(dict[str, Any], config_summaries[config.name])
        for name, arrays in global_errors[config.name].items():
            config_summary[name] = summarize_ground_errors(np.concatenate(arrays))
    return summary, payloads


def validate_cartesian_result(
        output_root: Path,
        path_dataset_root: Path,
        path_tracked_scene_root: Path,
        path_ground_effect_support_root: Path,
    ) -> dict[str, Any]:
    dataset = VirtualCrowd_Std(str(path_dataset_root.resolve(strict=True)))
    tracked_root = path_tracked_scene_root.resolve(strict=True)
    support_root = path_ground_effect_support_root.resolve(strict=True)
    scenes = dataset.get_list_scene_names()
    if tuple(scenes) != SCENES:
        raise ValueError('VirtualCrowd scenes differ from reviewed population')
    configs = virtualcrowd_rcr_cartesian_configs()
    summary, payloads = compute_cartesian_result(
        dataset,
        tracked_root,
        support_root,
        scenes,
    )
    key_sets = {
        config.name: (
            COMMON_NPZ_KEYS
            if config.density_mode == VARIANT_UNWEIGHTED
            else COMMON_NPZ_KEYS | KDE_NPZ_KEYS
        )
        for config in configs
    }
    validate_plain_result(
        output_root,
        scenes,
        tuple(config.name for config in configs),
        key_sets,
        summary,
        payloads,
    )
    return summary


def run_virtualcrowd_density_balanced_rcr_cartesian(
        *,
        path_dataset_root: Path,
        path_tracked_scene_root: Path,
        path_ground_effect_support_root: Path,
        path_output_root: Path,
    ) -> dict[str, Any]:
    '''Execute all 12 reviewed Cartesian configs and independently rebuild.'''
    dataset_root = path_dataset_root.resolve(strict=True)
    tracked_root = path_tracked_scene_root.resolve(strict=True)
    support_root = path_ground_effect_support_root.resolve(strict=True)
    if not tracked_root.is_dir() or not support_root.is_dir():
        raise NotADirectoryError('tracked-scene and support roots must be directories')
    if path_output_root.exists():
        raise FileExistsError('output root already exists: %s' % path_output_root)
    dataset = VirtualCrowd_Std(str(dataset_root))
    scenes = dataset.get_list_scene_names()
    if tuple(scenes) != SCENES:
        raise ValueError('VirtualCrowd scenes differ from reviewed population')
    configs = virtualcrowd_rcr_cartesian_configs()
    summary, payloads = compute_cartesian_result(
        dataset,
        tracked_root,
        support_root,
        scenes,
    )
    write_plain_result(
        path_output_root,
        scenes,
        tuple(config.name for config in configs),
        summary,
        payloads,
    )
    validate_cartesian_result(
        path_output_root,
        dataset_root,
        tracked_root,
        support_root,
    )
    return summary


def main(
        path_dataset_root: Path = typer.Option(..., exists=True, file_okay=False),
        path_tracked_scene_root: Path = typer.Option(..., exists=True, file_okay=False),
        execute: bool = typer.Option(False),
        path_ground_effect_support_root: Path | None = typer.Option(None),
        path_output_root: Path | None = typer.Option(None),
    ) -> None:
    '''Prepare the 12-config matrix by default; execute only when explicit.'''
    if execute:
        if path_ground_effect_support_root is None or path_output_root is None:
            raise typer.BadParameter(
                '--execute requires support and output roots',
            )
        summary = run_virtualcrowd_density_balanced_rcr_cartesian(
            path_dataset_root=path_dataset_root,
            path_tracked_scene_root=path_tracked_scene_root,
            path_ground_effect_support_root=path_ground_effect_support_root,
            path_output_root=path_output_root,
        )
    else:
        summary = prepare_virtualcrowd_density_balanced_rcr_cartesian(
            path_dataset_root=path_dataset_root,
            path_tracked_scene_root=path_tracked_scene_root,
        )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    typer.run(main)
