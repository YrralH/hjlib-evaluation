'''Compact operation coverage for density-balanced VirtualCrowd RCR.'''

from collections.abc import Callable
import importlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray
import pytest

from hjlib_detection import Tracked_Person, Tracked_Scene
from hjlib_evaluation import (
    Ground_Effect_Support,
    Ground_Estimation_Result,
    Ground_Observation_Set,
)


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / 'script'
sys.path.insert(0, str(SCRIPT_ROOT))


class Operation_Module(Protocol):
    VirtualCrowd_Std: object
    EXPECTED_COUNTS: dict[str, int]
    load_tracked_scene: object
    scene_camera_K: object
    load_scene_ground_effect_support: object
    estimate_ground_from_observations: object
    COMMON_NPZ_KEYS: set[str]
    KDE_NPZ_KEYS: set[str]
    EXPECTED_EFFECT_SUPPORT_TOTAL: int
    collect_ground_observations: object

    def run_virtualcrowd_density_balanced_rcr_ground(
            self,
            *,
            path_dataset_root: Path,
            path_tracked_scene_root: Path,
            path_ground_effect_support_root: Path,
            path_output_root: Path,
            dry_run: bool = False,
        ) -> dict[str, Any]: ...

    def validate_written_results(
            self,
            output_root: Path,
            path_dataset_root: Path,
            path_tracked_scene_root: Path,
            path_ground_effect_support_root: Path,
        ) -> dict[str, Any]: ...


operation = cast(
    Operation_Module,
    importlib.import_module('evaluate_virtualcrowd_density_balanced_rcr_ground'),
)


SCENES = [
    'scene1',
    'scene1_view2',
    'scene2',
    'scene2_view2',
    'scene3',
    'scene3_view2',
    'scene4',
    'scene4_view2',
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


def make_tracked_scene() -> Tracked_Scene:
    count = 65
    keypoints = np.zeros((count, 133, 3), dtype=np.float32)
    keypoints[:, :, 2] = 5.0
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


def make_observations(count: int) -> Ground_Observation_Set:
    frames = np.arange(count, dtype=np.int64)
    persons = np.zeros(count, dtype=np.int64)
    top = np.column_stack([
        np.linspace(-1.0, 1.0, count, dtype=np.float64),
        np.zeros(count, dtype=np.float64),
    ])
    bottom = top + np.array([0.0, 1.0], dtype=np.float64)
    return Ground_Observation_Set(
        frames,
        persons,
        top,
        bottom,
        np.full(count, 5.0, dtype=np.float64),
        np.full(count, 0.01, dtype=np.float64),
    )


def test_compact_density_balanced_operation() -> None:
    tracked = make_tracked_scene()
    support = make_support()
    expected_counts = {scene: 65 for scene in SCENES}
    passed_estimators: list[object] = []
    built_estimators: list[Callable[..., object]] = []
    captured_weights: list[NDArray[np.float64]] = []

    def fake_weighted_estimator(
            weights: NDArray[np.float64],
        ) -> Callable[..., object]:
        captured_weights.append(weights.copy())

        def sentinel(*unused_args: object) -> object:
            del unused_args
            raise AssertionError('sentinel estimator must be passed, not invoked by fake')

        built_estimators.append(sentinel)
        return sentinel

    def recording_fake_estimate(
            observations: Ground_Observation_Set,
            K: NDArray[np.generic],
            estimator: object = None,
        ) -> Ground_Estimation_Result:
        passed_estimators.append(estimator)
        return fake_estimate(observations, K, estimator)

    with TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        dataset_root = temporary_root / 'dataset'
        tracked_root = temporary_root / 'tracked'
        support_root = temporary_root / 'support'
        output_root = temporary_root / 'output'
        dataset_root.mkdir()
        tracked_root.mkdir()
        support_root.mkdir()
        with (
                patch.object(operation, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(operation, 'EXPECTED_COUNTS', expected_counts),
                patch.object(operation, 'EXPECTED_EFFECT_SUPPORT_TOTAL', 16),
                patch.object(operation, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    operation,
                    'scene_camera_K',
                    return_value=np.eye(3, dtype=np.float64),
                ),
                patch.object(
                    operation,
                    'load_scene_ground_effect_support',
                    return_value=support,
                ),
                patch.object(
                    operation,
                    'estimate_ground_from_observations',
                    side_effect=recording_fake_estimate,
                ),
                patch.object(
                    operation,
                    'weighted_estimator',
                    side_effect=fake_weighted_estimator,
                ),
            ):
            summary = operation.run_virtualcrowd_density_balanced_rcr_ground(
                path_dataset_root=dataset_root,
                path_tracked_scene_root=tracked_root,
                path_ground_effect_support_root=support_root,
                path_output_root=output_root,
            )
        assert summary['selected_total'] == 520
        assert len(list(output_root.rglob('*.npz'))) == 16
        assert len(passed_estimators) == 32
        assert sum(estimator is None for estimator in passed_estimators) == 16
        assert len(built_estimators) == len(captured_weights) == 16
        assert {
            id(estimator) for estimator in passed_estimators if estimator is not None
        } == {id(estimator) for estimator in built_estimators}
        assert all(weights.shape == (65,) for weights in captured_weights)
        assert all(np.isclose(np.mean(weights), 1.0) for weights in captured_weights)

        density_path = output_root / 'density_kde_scott_loo' / 'scene1.npz'
        with np.load(
                density_path,
                allow_pickle=False,
            ) as loaded:
            assert set(loaded.files) == operation.COMMON_NPZ_KEYS | operation.KDE_NPZ_KEYS
            assert float(loaded['scott_bandwidth_factor']) == 65 ** (-1.0 / 6.0)
            assert loaded['ground_effect_error_m'].shape == (2,)
            density_arrays = {name: loaded[name].copy() for name in loaded.files}

        objective_tamper = {
            name: value.copy() for name, value in density_arrays.items()
        }
        objective_tamper['rcr_objective'] = np.asarray(1.0, dtype=np.float64)
        np.savez(density_path, **cast(dict[str, Any], objective_tamper))
        with (
                patch.object(operation, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(operation, 'EXPECTED_COUNTS', expected_counts),
                patch.object(operation, 'EXPECTED_EFFECT_SUPPORT_TOTAL', 16),
                patch.object(operation, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    operation,
                    'scene_camera_K',
                    return_value=np.eye(3, dtype=np.float64),
                ),
                patch.object(
                    operation,
                    'load_scene_ground_effect_support',
                    return_value=support,
                ),
                patch.object(
                    operation,
                    'estimate_ground_from_observations',
                    side_effect=recording_fake_estimate,
                ),
                patch.object(
                    operation,
                    'weighted_estimator',
                    side_effect=fake_weighted_estimator,
                ),
                pytest.raises(ValueError, match='source reconstruction'),
            ):
            operation.validate_written_results(
                output_root,
                dataset_root,
                tracked_root,
                support_root,
            )
        np.savez(density_path, **cast(dict[str, Any], density_arrays))

        density_tamper = {name: value.copy() for name, value in density_arrays.items()}
        density_tamper['effective_sample_size'] += 1.0
        np.savez(density_path, **cast(dict[str, Any], density_tamper))
        with (
                patch.object(operation, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(operation, 'EXPECTED_COUNTS', expected_counts),
                patch.object(operation, 'EXPECTED_EFFECT_SUPPORT_TOTAL', 16),
                patch.object(operation, 'load_tracked_scene', return_value=tracked),
                patch.object(
                    operation,
                    'scene_camera_K',
                    return_value=np.eye(3, dtype=np.float64),
                ),
                patch.object(
                    operation,
                    'load_scene_ground_effect_support',
                    return_value=support,
                ),
                patch.object(
                    operation,
                    'estimate_ground_from_observations',
                    side_effect=recording_fake_estimate,
                ),
                patch.object(
                    operation,
                    'weighted_estimator',
                    side_effect=fake_weighted_estimator,
                ),
                pytest.raises(ValueError, match='source reconstruction'),
            ):
            operation.validate_written_results(
                output_root,
                dataset_root,
                tracked_root,
                support_root,
            )


def test_reviewed_count_dry_run_and_script_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_ROOT / 'evaluate_virtualcrowd_density_balanced_rcr_ground.py'),
            '--help',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert 'density-balanced' in completed.stdout

    def scene_from_path(path: Path) -> str:
        return path.stem.removesuffix('_tracked_scene')

    def observations_for_scene(
            scene: object,
            unused_top: object,
            unused_bottom: object,
            unused_confidence: object,
            maximum_bottom_pair_bbox_width_ratio: object = None,
        ) -> Ground_Observation_Set:
        del (
            unused_top,
            unused_bottom,
            unused_confidence,
            maximum_bottom_pair_bbox_width_ratio,
        )
        assert isinstance(scene, str)
        return make_observations(operation.EXPECTED_COUNTS[scene])

    with TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        dataset_root = temporary_root / 'dataset'
        tracked_root = temporary_root / 'tracked'
        support_root = temporary_root / 'support'
        dataset_root.mkdir()
        tracked_root.mkdir()
        support_root.mkdir()
        with (
                patch.object(operation, 'VirtualCrowd_Std', Fake_VirtualCrowd),
                patch.object(operation, 'load_tracked_scene', side_effect=scene_from_path),
                patch.object(
                    operation,
                    'collect_ground_observations',
                    side_effect=observations_for_scene,
                ),
            ):
            summary = operation.run_virtualcrowd_density_balanced_rcr_ground(
                path_dataset_root=dataset_root,
                path_tracked_scene_root=tracked_root,
                path_ground_effect_support_root=support_root,
                path_output_root=temporary_root / 'dry-output',
                dry_run=True,
            )
    assert summary['selected_total'] == 17_992


def smoke_test_density_balanced_rcr_operation() -> None:
    test_compact_density_balanced_operation()
    test_reviewed_count_dry_run_and_script_help()
