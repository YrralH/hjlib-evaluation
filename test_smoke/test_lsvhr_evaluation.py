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
    VirtualCrowd_Naive_Matched_Result,
    evaluate_lsvhr_virtualcrowd_entry,
    evaluate_lsvhr_virtualcrowd_matrix,
    evaluate_lsvhr_virtualcrowd_naive_matched_entry,
    evaluate_virtualcrowd_naive_matched,
    reduce_virtualcrowd_naive_matched_summaries,
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
    joint_index = np.arange(24, dtype=np.float64)
    joint_template = np.stack((
        0.001 * joint_index,
        0.0001 * joint_index ** 2,
        0.00001 * joint_index ** 3,
    ), axis=1)
    joints = root[:, None, :] + joint_template[None, :, :]
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


def full_population() -> LSVHR_Evaluation_Population:
    '''Select every synthetic GT occurrence in two scenes.'''
    return LSVHR_Evaluation_Population(
        filtering_id='vc.visible',
        split_id='vc.test2',
        rule_id='vc.visibility_continuity_v1',
        selection=VirtualCrowd_Eval_Population_Selection(
            population=(
                VirtualCrowd_Occurrence_Population.EVAL_PROTOCOL_VISIBLE
            ),
            spans=(
                VirtualCrowd_Accepted_Track_Span('scene-a', 7, 0, 10),
                VirtualCrowd_Accepted_Track_Span('scene-b', 7, 0, 10),
            ),
        ),
        split_scene_ids=('scene-a', 'scene-b'),
    )


def with_association(
        sequence: Corrected_Crowd_Sequence,
        matched_gt_rows: tuple[int, ...],
        false_positive_source_rows: tuple[int, ...],
    ) -> Corrected_Crowd_Sequence:
    '''Select prediction rows and install one exact sparse partition.'''
    matched = np.asarray(matched_gt_rows, dtype=np.int64)
    false_positive = np.asarray(false_positive_source_rows, dtype=np.int64)
    source_rows = np.concatenate((matched, false_positive))
    prediction_count = len(source_rows)
    targets = np.concatenate((
        matched,
        np.full(len(false_positive), -1, dtype=np.int64),
    ))
    return replace(
        sequence,
        prediction_frame_ids=sequence.prediction_frame_ids[source_rows],
        prediction_local_track_ids=np.arange(
            prediction_count, dtype=np.int64,
        ),
        prediction_joints_world_m=sequence.prediction_joints_world_m[
            source_rows
        ],
        prediction_coco17_xy_px=sequence.prediction_coco17_xy_px[source_rows],
        prediction_coco17_camera_depth_m=(
            sequence.prediction_coco17_camera_depth_m[source_rows]
        ),
        prediction_pelvis_camera_depth_m=(
            sequence.prediction_pelvis_camera_depth_m[source_rows]
        ),
        prediction_identity_target_gt_rows=targets,
        matched_gt_rows=matched,
        matched_prediction_rows=np.arange(len(matched), dtype=np.int64),
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


def test_naive_matched_profile_micro_reduces_counts_and_metrics() -> None:
    first = with_association(
        make_sequence('scene-a'), tuple(range(8)), (9,),
    )
    second = with_association(
        make_sequence('scene-b'), tuple(range(10)), (),
    )
    loader = Synthetic_Loader({'scene-a': first, 'scene-b': second})
    evaluated = evaluate_lsvhr_virtualcrowd_naive_matched_entry(
        LSVHR_Evaluation_Entry('crowd3d/released', loader),
        full_population(),
    )
    assert evaluated.profile is LSVHR_Evaluation_Profile.NAIVE_MATCHED
    assert type(evaluated.result) is VirtualCrowd_Naive_Matched_Result
    result = evaluated.result
    assert (result.tp, result.fn, result.fp) == (18, 2, 1)
    assert result.precision == pytest.approx(18.0 / 19.0)
    assert result.recall == pytest.approx(0.9)
    assert result.f1 == pytest.approx(
        2.0 * result.precision * result.recall
        / (result.precision + result.recall)
    )
    assert result.mpjpe_world_mm == pytest.approx(0.0)
    assert result.t_mpjpe_mm == pytest.approx(0.0)
    assert result.pa_mpjpe_mm == pytest.approx(0.0)
    assert result.oks_vis == pytest.approx(1.0)
    assert result.acc_root_ratio == pytest.approx(1.0)


def test_naive_matched_nullable_support_and_temporal_gap() -> None:
    sequence = make_sequence('scene-gap')
    gap = with_association(sequence, (0, 1, 2, 3, 5, 6, 7, 8, 9), ())
    gap_summary = evaluate_virtualcrowd_naive_matched(
        gap, 'vc.visible', 'vc.test1', np.ones(10, dtype=np.bool_),
    )
    gap_result = reduce_virtualcrowd_naive_matched_summaries((gap_summary,))
    assert gap_result.acc_root_sample_count == 0
    assert gap_result.acc_root_ratio is None

    no_predictions = with_association(sequence, (), ())
    empty_summary = evaluate_virtualcrowd_naive_matched(
        no_predictions,
        'vc.visible',
        'vc.test1',
        np.ones(10, dtype=np.bool_),
    )
    empty_result = reduce_virtualcrowd_naive_matched_summaries((empty_summary,))
    assert (empty_result.tp, empty_result.fn, empty_result.fp) == (0, 10, 0)
    assert (empty_result.precision, empty_result.recall, empty_result.f1) \
        == (0.0, 0.0, 0.0)
    assert empty_result.mpjpe_world_mm is None
    assert empty_result.t_mpjpe_mm is None
    assert empty_result.pa_mpjpe_mm is None
    assert empty_result.oks_vis is None
    assert empty_result.acc_root_ratio is None

    no_selected_gt = with_association(sequence, (), (0,))
    no_gt_summary = evaluate_virtualcrowd_naive_matched(
        no_selected_gt,
        'vc.visible',
        'vc.test1',
        np.zeros(10, dtype=np.bool_),
    )
    no_gt_result = reduce_virtualcrowd_naive_matched_summaries((
        no_gt_summary,
    ))
    assert (no_gt_result.gt_count, no_gt_result.prediction_count) == (0, 1)
    assert (no_gt_result.precision, no_gt_result.recall, no_gt_result.f1) \
        == (0.0, 0.0, 0.0)

    invisible = replace(
        sequence,
        prediction_coco17_camera_depth_m=np.zeros(
            (10, 17), dtype=np.float64,
        ),
    )
    invisible_summary = evaluate_virtualcrowd_naive_matched(
        invisible,
        'vc.visible',
        'vc.test1',
        np.ones(10, dtype=np.bool_),
    )
    invisible_result = reduce_virtualcrowd_naive_matched_summaries((
        invisible_summary,
    ))
    assert invisible_result.tp == 10
    assert invisible_result.oks_vis_count == 0
    assert invisible_result.oks_vis is None

    frame_motion = 0.01 * np.arange(10, dtype=np.float64) ** 2
    stationary_reference = replace(
        sequence,
        gt_joints_world_m=(
            sequence.gt_joints_world_m
            - np.stack((
                np.zeros(10), frame_motion, np.zeros(10),
            ), axis=1)[:, None, :]
        ),
    )
    stationary_summary = evaluate_virtualcrowd_naive_matched(
        stationary_reference,
        'vc.visible',
        'vc.test1',
        np.ones(10, dtype=np.bool_),
    )
    stationary_result = reduce_virtualcrowd_naive_matched_summaries((
        stationary_summary,
    ))
    assert stationary_result.acc_root_sample_count > 0
    assert stationary_result.acc_root_ratio is None

    with pytest.raises(ValueError, match='matching metrics differ'):
        replace(stationary_result, precision=0.25)


def test_naive_matched_rejects_unclosed_mapping_and_sorts_scenes() -> None:
    first = with_association(make_sequence('scene-b'), (0, 1), (2,))
    second = with_association(make_sequence('scene-a'), (0, 1, 2), ())
    summaries = tuple(
        evaluate_virtualcrowd_naive_matched(
            sequence,
            'vc.visible',
            'vc.test2',
            np.ones(10, dtype=np.bool_),
        )
        for sequence in (first, second)
    )
    result = reduce_virtualcrowd_naive_matched_summaries(summaries)
    assert tuple(
        value.matched.scene_id for value in result.scene_summaries
    ) == ('scene-a', 'scene-b')

    unclosed = replace(
        first,
        prediction_identity_target_gt_rows=np.array([0, 1, 2]),
    )
    with pytest.raises(ValueError, match='mapped predictions must equal'):
        evaluate_virtualcrowd_naive_matched(
            unclosed,
            'vc.visible',
            'vc.test2',
            np.ones(10, dtype=np.bool_),
        )


def smoke_test_lsvhr_evaluation() -> None:
    '''Exercise exact population projection and registered-entry composition.'''
    test_exact_split_projection_and_entry_evaluation()
    test_ordered_matrix_retains_official_entry_identity()
    test_population_and_loader_drift_failures()
    test_naive_matched_profile_micro_reduces_counts_and_metrics()
    test_naive_matched_nullable_support_and_temporal_gap()
    test_naive_matched_rejects_unclosed_mapping_and_sorts_scenes()
