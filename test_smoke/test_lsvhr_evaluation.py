'''Smoke for method-neutral LSV-HR entry and matrix evaluation.'''
from dataclasses import replace

import numpy as np
import pytest

from hjlib_dataset_std import (
    VirtualCrowd_Accepted_Track_Span,
    VirtualCrowd_Eval_Population_Selection,
    VirtualCrowd_Occurrence_Population,
)
from hjlib_evaluation import (
    CORRECTED_CROWD_SCHEMA_VERSION,
    Corrected_Crowd_Sequence,
    LSVHR_Evaluation_Entry,
    LSVHR_Evaluation_Population,
    LSVHR_Evaluation_Profile,
    evaluate_lsvhr_virtualcrowd_entry,
    evaluate_lsvhr_virtualcrowd_matrix,
    selected_gt_mask_for_lsvhr_population_scene,
)


def make_sequence(
        scene_id: str,
        *,
        reverse_rows: bool = False,
    ) -> Corrected_Crowd_Sequence:
    '''Build ten direct-matched occurrences with nonzero root acceleration.'''
    frames = np.arange(10, dtype=np.int64)
    root = np.zeros((10, 3), dtype=np.float64)
    root[:, 1] = 0.01 * frames.astype(np.float64) ** 2
    root[:, 2] = 5.0
    joints = np.repeat(root[:, None, :], 24, axis=1)
    coco = np.zeros((10, 17, 2), dtype=np.float64)
    visibility = np.ones((10, 17), dtype=np.float64)
    order = np.arange(9, -1, -1) if reverse_rows else frames
    return Corrected_Crowd_Sequence(
        schema_version=CORRECTED_CROWD_SCHEMA_VERSION,
        scene_id=scene_id,
        frame_domain=frames,
        gt_frame_ids=frames[order],
        gt_track_ids=np.full(10, 7, dtype=np.int64),
        gt_joints_world_m=joints[order],
        gt_coco17_xy_px=coco[order],
        gt_visibility_native=visibility[order],
        gt_bbox_xyxy_px=np.tile(
            np.array([0.0, 0.0, 100.0, 200.0]),
            (10, 1),
        ),
        gt_pelvis_camera_depth_m=np.full(10, 5.0),
        prediction_frame_ids=frames[order],
        prediction_local_track_ids=np.full(10, 7, dtype=np.int64),
        prediction_joints_world_m=joints[order],
        prediction_coco17_xy_px=coco[order],
        prediction_coco17_camera_depth_m=np.full((10, 17), 5.0),
        prediction_pelvis_camera_depth_m=np.full(10, 5.0),
        prediction_identity_target_gt_rows=np.arange(10, dtype=np.int64),
        matched_gt_rows=np.arange(10, dtype=np.int64),
        matched_prediction_rows=np.arange(10, dtype=np.int64),
        common_gt_mask=np.ones(10, dtype=np.bool_),
    )


class Synthetic_Loader:
    '''Structural loader with observable one-load-per-scene behavior.'''

    def __init__(self, scenes: dict[str, Corrected_Crowd_Sequence]) -> None:
        self.scenes = scenes
        self.calls: list[str] = []

    def load_scene(self, scene_id: str) -> Corrected_Crowd_Sequence:
        self.calls.append(scene_id)
        return self.scenes[scene_id]


def population() -> LSVHR_Evaluation_Population:
    '''Build a two-scene split projection from a three-scene selection.'''
    return LSVHR_Evaluation_Population(
        filtering_id='vc.visible_common',
        split_id='vc.test6',
        rule_id='vc.visibility_continuity_v1',
        selection=VirtualCrowd_Eval_Population_Selection(
            population=(
                VirtualCrowd_Occurrence_Population
                .EVAL_PROTOCOL_VISIBLE_COMMON
            ),
            spans=(
                VirtualCrowd_Accepted_Track_Span('scene-a', 7, 1, 9),
                VirtualCrowd_Accepted_Track_Span('scene-b', 7, 1, 9),
                VirtualCrowd_Accepted_Track_Span('scene-train', 7, 1, 9),
            ),
        ),
        split_scene_ids=('scene-a', 'scene-b'),
    )


def test_exact_split_projection_and_entry_evaluation() -> None:
    selected = population()
    assert tuple(span.scene_name for span in selected.spans) == (
        'scene-a',
        'scene-b',
    )
    assert selected.count_span == 2
    assert selected.count_occurrence == 16

    loader = Synthetic_Loader({
        'scene-a': make_sequence('scene-a', reverse_rows=True),
        'scene-b': make_sequence('scene-b'),
    })
    evaluated = evaluate_lsvhr_virtualcrowd_entry(
        LSVHR_Evaluation_Profile.NAIVE,
        LSVHR_Evaluation_Entry('ours/010', loader),
        selected,
    )
    assert evaluated.profile is LSVHR_Evaluation_Profile.NAIVE
    assert evaluated.entry_id == 'ours/010'
    assert evaluated.result.scene_count == 2
    assert evaluated.result.selected_gt_count == 16
    assert evaluated.result.mpjpe_world_mm == pytest.approx(0.0)
    assert evaluated.result.t_mpjpe_mm == pytest.approx(0.0)
    assert evaluated.result.oks_vis == pytest.approx(1.0)
    assert evaluated.result.acc_root_ratio == pytest.approx(1.0)
    assert loader.calls == ['scene-a', 'scene-b']


def test_ordered_matrix_retains_official_entry_identity() -> None:
    scenes = {
        'scene-a': make_sequence('scene-a'),
        'scene-b': make_sequence('scene-b'),
    }
    second = Synthetic_Loader(scenes)
    first = Synthetic_Loader(scenes)
    results = evaluate_lsvhr_virtualcrowd_matrix(
        LSVHR_Evaluation_Profile.NAIVE,
        (
            LSVHR_Evaluation_Entry('dycrowd/001', second),
            LSVHR_Evaluation_Entry('crowd4d/001', first),
        ),
        population(),
    )
    assert tuple(item.entry_id for item in results) == (
        'dycrowd/001',
        'crowd4d/001',
    )
    assert second.calls == ['scene-a', 'scene-b']
    assert first.calls == ['scene-a', 'scene-b']

    with pytest.raises(ValueError, match='entry IDs must be unique'):
        evaluate_lsvhr_virtualcrowd_matrix(
            LSVHR_Evaluation_Profile.NAIVE,
            (
                LSVHR_Evaluation_Entry('ours/010', first),
                LSVHR_Evaluation_Entry('ours/010', second),
            ),
            population(),
        )


def test_population_and_loader_drift_failures() -> None:
    selected = population()
    sequence = make_sequence('scene-a', reverse_rows=True)
    mask = selected_gt_mask_for_lsvhr_population_scene(sequence, selected)
    assert sorted(sequence.gt_frame_ids[mask].tolist()) == list(range(1, 9))
    assert not mask.flags.writeable

    with pytest.raises(ValueError, match='select every exact split scene'):
        LSVHR_Evaluation_Population(
            filtering_id=selected.filtering_id,
            split_id=selected.split_id,
            rule_id=selected.rule_id,
            selection=selected.selection,
            split_scene_ids=('scene-a', 'scene-missing'),
        )

    missing_tracks = np.full(10, 7, dtype=np.int64)
    missing_tracks[np.flatnonzero(sequence.gt_frame_ids == 8)[0]] = 8
    missing_key_sequence = replace(sequence, gt_track_ids=missing_tracks)
    with pytest.raises(ValueError, match='missing .* selected GT keys'):
        selected_gt_mask_for_lsvhr_population_scene(
            missing_key_sequence,
            selected,
        )

    wrong_scene_loader = Synthetic_Loader({
        'scene-a': make_sequence('wrong-scene'),
        'scene-b': make_sequence('scene-b'),
    })
    with pytest.raises(ValueError, match='wrong scene identity'):
        evaluate_lsvhr_virtualcrowd_entry(
            LSVHR_Evaluation_Profile.NAIVE,
            LSVHR_Evaluation_Entry('ours/010', wrong_scene_loader),
            selected,
        )


def smoke_test_lsvhr_evaluation() -> None:
    '''Exercise exact population projection and registered-entry composition.'''
    test_exact_split_projection_and_entry_evaluation()
    test_ordered_matrix_retains_official_entry_identity()
    test_population_and_loader_drift_failures()
