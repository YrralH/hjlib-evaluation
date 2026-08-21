'''Portable coverage for the VirtualCrowd RCR Cartesian preparation.'''

import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import pytest
import typer
from typer.testing import CliRunner

from hjlib_detection import Tracked_Person, Tracked_Scene
from hjlib_evaluation import (
    Ground_Effect_Support,
    Ground_Estimation_Result,
    Ground_Observation_Set,
)


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / 'script'
sys.path.insert(0, str(SCRIPT_ROOT))
single_operation = importlib.import_module(
    'evaluate_virtualcrowd_density_balanced_rcr_ground',
)
cartesian = importlib.import_module(
    'evaluate_virtualcrowd_density_balanced_rcr_cartesian',
)

SCENES = [
    'scene1', 'scene1_view2', 'scene2', 'scene2_view2',
    'scene3', 'scene3_view2', 'scene4', 'scene4_view2',
]


class Fake_VirtualCrowd:
    def __init__(self, unused_root: str) -> None:
        del unused_root

    def get_list_scene_names(self) -> list[str]:
        return SCENES.copy()

    def get_ground_param_by_name_scene(
            self,
            unused_scene: str,
        ) -> NDArray[np.float64]:
        del unused_scene
        return np.array([0.0, 0.0, 1.0, -2.0], dtype=np.float64)


def make_uniform_tracked_scene() -> Tracked_Scene:
    count = 65
    keypoints = np.zeros((count, 133, 3), dtype=np.float32)
    keypoints[:, :, 2] = 6.0
    x = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    y = 1.0 + 0.2 * np.sin(np.arange(count, dtype=np.float32))
    keypoints[:, 5, :2] = np.column_stack([x - 0.1, y - 0.5])
    keypoints[:, 6, :2] = np.column_stack([x + 0.1, y - 0.5])
    keypoints[:, 15, :2] = np.column_stack([x - 0.05, y])
    keypoints[:, 16, :2] = np.column_stack([x + 0.05, y])
    bboxes = np.tile(
        np.array([[0.0, 100.0, -50.0, 50.0, 1.0]], dtype=np.float32),
        (count, 1),
    )
    person = Tracked_Person(
        person_id=0,
        list_ranges=((0, count),),
        bboxes=bboxes,
        keypoints=keypoints,
        keypoints_mask=np.ones(count, dtype=np.bool_),
        num_frame_scene=count,
    )
    return Tracked_Scene(count, 1, True, (133, 3), (person,))


def make_boundary_tracked_scene() -> Tracked_Scene:
    scores = np.array([4.0, 4.1, 4.5, 4.6, 5.0, 5.1], dtype=np.float32)
    separations = np.array([2.0, 3.0, 2.98, 4.0, 3.98, 2.8], dtype=np.float32)
    count = scores.size
    keypoints = np.zeros((count, 133, 3), dtype=np.float32)
    keypoints[:, :, 2] = scores[:, None]
    keypoints[:, 15:17, 1] = 10.0
    keypoints[:, 15, 0] = -separations / 2.0
    keypoints[:, 16, 0] = separations / 2.0
    bboxes = np.tile(
        np.array([[0.0, 100.0, -10.0, 10.0, 1.0]], dtype=np.float32),
        (count, 1),
    )
    reconstructed_ratios = (
        keypoints[:, 16, 0].astype(np.float64)
        - keypoints[:, 15, 0].astype(np.float64)
    ) / (
        bboxes[:, 3].astype(np.float64)
        - bboxes[:, 2].astype(np.float64)
    )
    assert reconstructed_ratios[1] == 0.15
    assert reconstructed_ratios[3] == 0.20
    person = Tracked_Person(
        person_id=7,
        list_ranges=((0, count),),
        bboxes=bboxes,
        keypoints=keypoints,
        keypoints_mask=np.ones(count, dtype=np.bool_),
        num_frame_scene=count,
    )
    return Tracked_Scene(count, 8, True, (133, 3), (person,))


def make_support() -> Ground_Effect_Support:
    return Ground_Effect_Support(
        np.array([0, 0], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]], dtype=np.float64),
    )


def fake_estimate(
        observations: Ground_Observation_Set,
        unused_K: NDArray[np.generic],
        unused_estimator: object = None,
    ) -> Ground_Estimation_Result:
    del unused_K, unused_estimator
    return Ground_Estimation_Result(
        observations,
        np.array([0.0, 0.0, 1.0, -2.0], dtype=np.float64),
        0.0,
    )


def uniform_expected_counts() -> dict[str, tuple[int, ...]]:
    return {
        config.population_name: (65,) * 8
        for config in cartesian.population_configs(
            cartesian.virtualcrowd_rcr_cartesian_configs(),
        )
    }


def test_cartesian_config_order_and_strict_population_boundaries() -> None:
    configs = cartesian.virtualcrowd_rcr_cartesian_configs()
    assert len(configs) == 12
    assert len({config.name for config in configs}) == 12
    assert configs[0].name == (
        'conf_gt_4p0__ankle_lt_0p15__filtered_unweighted'
    )
    assert configs[-1].name == (
        'conf_gt_5p0__ankle_lt_0p20__density_kde_scott_loo'
    )
    assert cartesian.EXPECTED_CARTESIAN_COUNTS[
        'conf_gt_4p0__ankle_lt_0p20'
    ] == tuple(single_operation.EXPECTED_COUNTS[scene] for scene in SCENES)
    populations = cartesian.collect_scene_populations(
        make_boundary_tracked_scene(),
        configs,
    )
    expected_frames = {
        'conf_gt_4p0__ankle_lt_0p15': {2, 5},
        'conf_gt_4p0__ankle_lt_0p20': {1, 2, 4, 5},
        'conf_gt_4p5__ankle_lt_0p15': {5},
        'conf_gt_4p5__ankle_lt_0p20': {4, 5},
        'conf_gt_5p0__ankle_lt_0p15': {5},
        'conf_gt_5p0__ankle_lt_0p20': {5},
    }
    for population_name, observations in populations.items():
        assert set(observations.frame_indices.tolist()) == expected_frames[
            population_name
        ]


def test_prepare_cartesian_reads_only_selection_inputs() -> None:
    tracked = make_uniform_tracked_scene()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset_root = root / 'dataset'
        tracked_root = root / 'tracked'
        dataset_root.mkdir()
        tracked_root.mkdir()
        with (
                patch.object(cartesian, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(
                    cartesian,
                    'EXPECTED_CARTESIAN_COUNTS',
                    uniform_expected_counts(),
                ),
                patch.object(cartesian, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    cartesian,
                    'evaluate_scene',
                    side_effect=AssertionError('prepare must not solve'),
                ),
                patch.object(
                    cartesian,
                    'load_scene_ground_effect_support',
                    side_effect=AssertionError('prepare must not load support'),
                ),
                patch.object(
                    cartesian,
                    'collect_ground_observations',
                    wraps=cartesian.collect_ground_observations,
                ) as collect_mock,
            ):
            summary = cartesian.prepare_virtualcrowd_density_balanced_rcr_cartesian(
                path_dataset_root=dataset_root,
                path_tracked_scene_root=tracked_root,
            )
        assert summary['mode'] == 'cartesian_prepare_only'
        assert summary['configuration_count'] == 12
        assert summary['population_count'] == 6
        populations = cast(list[dict[str, Any]], summary['populations'])
        assert all(record['selected_total'] == 520 for record in populations)
        assert collect_mock.call_count == 48
        assert not (root / 'output').exists()


def test_cartesian_cli_defaults_to_prepare_and_execute_requires_roots() -> None:
    app = typer.Typer()
    app.command()(cartesian.main)
    runner = CliRunner()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset_root = root / 'dataset'
        tracked_root = root / 'tracked'
        dataset_root.mkdir()
        tracked_root.mkdir()
        prepared: dict[str, object] = {
            'mode': 'cartesian_prepare_only',
            'configuration_count': 12,
            'population_count': 6,
            'configs': [],
            'populations': [],
        }
        with (
                patch.object(
                    cartesian,
                    'prepare_virtualcrowd_density_balanced_rcr_cartesian',
                    return_value=prepared,
                ) as prepare_mock,
                patch.object(
                    cartesian,
                    'run_virtualcrowd_density_balanced_rcr_cartesian',
                    side_effect=AssertionError('default CLI must not execute'),
                ),
            ):
            result = runner.invoke(app, [
                '--path-dataset-root', str(dataset_root),
                '--path-tracked-scene-root', str(tracked_root),
                '--path-ground-effect-support-root', str(root / 'unused-support'),
                '--path-output-root', str(root / 'unused-output'),
            ])
        assert result.exit_code == 0
        prepare_mock.assert_called_once_with(
            path_dataset_root=dataset_root,
            path_tracked_scene_root=tracked_root,
        )
        assert not (root / 'unused-output').exists()

        missing = runner.invoke(app, [
            '--path-dataset-root', str(dataset_root),
            '--path-tracked-scene-root', str(tracked_root),
            '--execute',
        ])
        assert missing.exit_code != 0
        assert '--execute requires support and output roots' in missing.output


def test_compact_cartesian_writer_and_source_reconstruction() -> None:
    tracked = make_uniform_tracked_scene()
    support = make_support()
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset_root = root / 'dataset'
        tracked_root = root / 'tracked'
        support_root = root / 'support'
        output_root = root / 'output'
        single_output_root = root / 'single-output'
        dataset_root.mkdir()
        tracked_root.mkdir()
        support_root.mkdir()
        with (
                patch.object(cartesian, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(
                    cartesian,
                    'EXPECTED_CARTESIAN_COUNTS',
                    uniform_expected_counts(),
                ),
                patch.object(cartesian, 'EXPECTED_EFFECT_SUPPORT_TOTAL', 16),
                patch.object(cartesian, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    cartesian,
                    'scene_camera_K',
                    return_value=np.eye(3, dtype=np.float64),
                ),
                patch.object(
                    cartesian,
                    'load_scene_ground_effect_support',
                    return_value=support,
                ),
                patch.object(
                    single_operation,
                    'estimate_ground_from_observations',
                    side_effect=fake_estimate,
                ),
                patch.object(
                    cartesian,
                    'evaluate_scene',
                    wraps=cartesian.evaluate_scene,
                ) as evaluate_mock,
            ):
            summary = cartesian.run_virtualcrowd_density_balanced_rcr_cartesian(
                path_dataset_root=dataset_root,
                path_tracked_scene_root=tracked_root,
                path_ground_effect_support_root=support_root,
                path_output_root=output_root,
            )
        assert evaluate_mock.call_count == 96
        assert summary['configuration_count'] == 12
        assert len(list(output_root.rglob('*.npz'))) == 96
        config_summaries = cast(dict[str, Any], summary['configs'])
        assert all(value['selected_total'] == 520 for value in config_summaries.values())
        with (
                patch.object(single_operation, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(
                    single_operation,
                    'EXPECTED_COUNTS',
                    {scene: 65 for scene in SCENES},
                ),
                patch.object(single_operation, 'EXPECTED_EFFECT_SUPPORT_TOTAL', 16),
                patch.object(single_operation, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    single_operation,
                    'scene_camera_K',
                    return_value=np.eye(3, dtype=np.float64),
                ),
                patch.object(
                    single_operation,
                    'load_scene_ground_effect_support',
                    return_value=support,
                ),
                patch.object(
                    single_operation,
                    'estimate_ground_from_observations',
                    side_effect=fake_estimate,
                ),
            ):
            single_summary = single_operation.run_virtualcrowd_density_balanced_rcr_ground(
                path_dataset_root=dataset_root,
                path_tracked_scene_root=tracked_root,
                path_ground_effect_support_root=support_root,
                path_output_root=single_output_root,
            )
        for density_mode in single_operation.VARIANTS:
            config_name = 'conf_gt_4p0__ankle_lt_0p20__%s' % density_mode
            cartesian_summary = cast(dict[str, Any], summary['configs'])[config_name]
            single_variant_summary = cast(
                dict[str, Any],
                single_summary['variants'][density_mode],
            )
            for key in ('ground_effect', 'normal_oracle', 'distance_only', 'scenes'):
                assert cartesian_summary[key] == single_variant_summary[key]
            for scene in SCENES:
                with (
                        np.load(
                            output_root / config_name / ('%s.npz' % scene),
                            allow_pickle=False,
                        ) as cartesian_loaded,
                        np.load(
                            single_output_root / density_mode / ('%s.npz' % scene),
                            allow_pickle=False,
                        ) as single_loaded,
                    ):
                    assert set(cartesian_loaded.files) == set(single_loaded.files)
                    for name in cartesian_loaded.files:
                        assert np.array_equal(cartesian_loaded[name], single_loaded[name])
        first_config = cartesian.virtualcrowd_rcr_cartesian_configs()[0]
        path = output_root / first_config.name / 'scene1.npz'
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
        arrays['effect_gt_track_id'][0] += 1
        np.savez(path, **cast(dict[str, Any], arrays))
        with (
                patch.object(cartesian, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(
                    cartesian,
                    'EXPECTED_CARTESIAN_COUNTS',
                    uniform_expected_counts(),
                ),
                patch.object(cartesian, 'EXPECTED_EFFECT_SUPPORT_TOTAL', 16),
                patch.object(cartesian, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    cartesian,
                    'scene_camera_K',
                    return_value=np.eye(3, dtype=np.float64),
                ),
                patch.object(
                    cartesian,
                    'load_scene_ground_effect_support',
                    return_value=support,
                ),
                patch.object(
                    single_operation,
                    'estimate_ground_from_observations',
                    side_effect=fake_estimate,
                ),
                pytest.raises(ValueError, match='source reconstruction'),
            ):
            cartesian.validate_cartesian_result(
                output_root,
                dataset_root,
                tracked_root,
                support_root,
            )


def smoke_test_density_balanced_rcr_cartesian() -> None:
    test_cartesian_config_order_and_strict_population_boundaries()
    test_prepare_cartesian_reads_only_selection_inputs()
    test_cartesian_cli_defaults_to_prepare_and_execute_requires_roots()
    test_compact_cartesian_writer_and_source_reconstruction()
