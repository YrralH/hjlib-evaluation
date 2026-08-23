'''Data-free gates for unordered JTA person-detection evaluation.'''
from time import perf_counter

import numpy as np
import pytest

from hjlib_evaluation import (
    JTA_CAMERA_K,
    JTA_ENDPOINT_INDICES,
    JTA_Person_Detection_GT_Frame,
    JTA_Person_Detection_Prediction_Frame,
    JTA_Person_Detection_Reducer,
    associate_jta_people,
    evaluate_jta_person_detection_frame,
    jta_person_detection_result_from_json,
    jta_person_detection_result_to_json,
    make_jta_person_detection_gt_frame,
)


SOURCE_SHA = 'a' * 64
PROFILE_SHA = 'b' * 64


def raw_people(count: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.arange(count, dtype=np.int64)
    xyz = np.empty((count, 22, 3), dtype=np.float64)
    for index_person in range(count):
        index_joint = np.arange(22, dtype=np.float64)
        xyz[index_person, :, 0] = (index_joint % 5 - 2.0) * 0.12 + index_person
        xyz[index_person, :, 1] = (index_joint // 5 - 2.0) * 0.10
        xyz[index_person, :, 2] = 8.0 + index_joint * 0.01
    projected = xyz @ JTA_CAMERA_K.T
    xy = projected[:, :, :2] / projected[:, :, 2:3]
    zeros = np.zeros((count, 22), dtype=np.int64)
    return ids, xy, xyz, zeros.copy(), zeros.copy()


def make_gt(count: int = 1) -> JTA_Person_Detection_GT_Frame:
    return make_jta_person_detection_gt_frame(
        'seq_1', 0, *raw_people(count),
    )


def make_prediction(
        gt: JTA_Person_Detection_GT_Frame,
        xyz: np.ndarray | None = None,
) -> JTA_Person_Detection_Prediction_Frame:
    values = gt.gt_xyz_camera if xyz is None else xyz
    return JTA_Person_Detection_Prediction_Frame(
        scene_id=gt.scene_id,
        frame_id=gt.frame_id,
        prediction_source_sha256=SOURCE_SHA,
        prediction_profile_sha256=PROFILE_SHA,
        prediction_row_ids=np.arange(len(values), dtype=np.int64),
        pred_xyz_camera=values,
    )


def test_raw_jta_constructor_sorts_filters_and_owns_visibility() -> None:
    ids, xy, xyz, occ, self_occ = raw_people(3)
    ids[:] = [8, 2, 5]
    xy[2, 0, 0] = np.nan
    occ[0, JTA_ENDPOINT_INDICES[0]] = 1
    gt = make_jta_person_detection_gt_frame(
        'seq_7', 15, ids, xy, xyz, occ, self_occ,
    )
    np.testing.assert_array_equal(gt.gt_source_ids, [2, 8])
    assert gt.gt_visible.shape == (2, 12)
    assert not gt.gt_visible[1, 0]
    assert np.all(gt.gt_bbox_xyxy[:, 2:] > gt.gt_bbox_xyxy[:, :2])
    assert not gt.gt_xy.flags.writeable
    assert len(gt.semantic_sha256) == 64


def test_threshold_graph_maximizes_cardinality_before_oks() -> None:
    oks = np.asarray([[0.90, 0.51], [0.50, 0.49]], dtype=np.float64)
    association = associate_jta_people(oks, oks >= 0.50)
    np.testing.assert_array_equal(association.match_gt_indices, [0, 1])
    np.testing.assert_array_equal(association.match_prediction_indices, [1, 0])
    np.testing.assert_allclose(association.matched_oks, [0.51, 0.50])


def test_perfect_frame_reducer_and_canonical_round_trip() -> None:
    gt = make_gt(2)
    prediction = make_prediction(gt)
    frame = evaluate_jta_person_detection_frame(gt, prediction)
    assert len(frame.association.match_gt_indices) == 2
    assert frame.absolute_mpjpe_person_sum_mm == pytest.approx(0.0)
    assert frame.pelvis_mpjpe_person_sum_mm == pytest.approx(0.0)
    assert frame.pa_mpjpe_person_sum_mm == pytest.approx(0.0, abs=1e-10)
    reducer = JTA_Person_Detection_Reducer(
        (('seq_1', 0),), SOURCE_SHA, PROFILE_SHA,
    )
    reducer.add_frame(gt, prediction)
    result = reducer.finalize()
    assert result.matched_mean_oks == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.absolute_mpjpe_mm == pytest.approx(0.0)
    encoded = jta_person_detection_result_to_json(result)
    assert b'NaN' not in encoded
    assert jta_person_detection_result_from_json(encoded) == result
    with pytest.raises(RuntimeError, match='sealed'):
        reducer.finalize()


def test_projection_invalid_prediction_remains_unmatched() -> None:
    gt = make_gt()
    xyz = np.array(gt.gt_xyz_camera, copy=True)
    xyz[0, 0, 2] = 0.0
    prediction = make_prediction(gt, xyz)
    metrics = evaluate_jta_person_detection_frame(gt, prediction)
    assert len(metrics.association.match_gt_indices) == 0
    assert metrics.projection_invalid_prediction_count == 1


def test_empty_populations_and_degenerate_pa_have_explicit_denominators() -> None:
    gt_empty = make_gt(0)
    prediction_empty = make_prediction(gt_empty)
    reducer_empty = JTA_Person_Detection_Reducer(
        (('seq_1', 0),), SOURCE_SHA, PROFILE_SHA,
    )
    reducer_empty.add_frame(gt_empty, prediction_empty)
    empty = reducer_empty.finalize()
    assert empty.matched_mean_oks is None
    assert empty.all_gt_mean_oks is None
    assert empty.recall is None

    gt = make_gt()
    repeated = np.repeat(gt.gt_xyz_camera[:, :1], 12, axis=1)
    prediction = make_prediction(gt, repeated)
    metrics = evaluate_jta_person_detection_frame(gt, prediction)
    if len(metrics.association.match_gt_indices):
        assert metrics.pa_valid_person_count == 0
        assert metrics.pa_degenerate_person_count == 1


def test_reducer_validation_failure_does_not_advance_population() -> None:
    gt = make_gt()
    prediction = make_prediction(gt)
    wrong = JTA_Person_Detection_Prediction_Frame(
        scene_id='seq_2',
        frame_id=0,
        prediction_source_sha256=SOURCE_SHA,
        prediction_profile_sha256=PROFILE_SHA,
        prediction_row_ids=prediction.prediction_row_ids,
        pred_xyz_camera=prediction.pred_xyz_camera,
    )
    reducer = JTA_Person_Detection_Reducer(
        (('seq_1', 0),), SOURCE_SHA, PROFILE_SHA,
    )
    with pytest.raises(ValueError, match='out of order'):
        reducer.add_frame(gt, wrong)
    reducer.add_frame(gt, prediction)
    assert reducer.finalize().frame_count == 1


def test_large_all_tie_assignment_is_lexical_and_bounded() -> None:
    size = 128
    oks = np.ones((size, size), dtype=np.float64)
    started = perf_counter()
    association = associate_jta_people(oks, np.ones_like(oks, dtype=np.bool_))
    elapsed = perf_counter() - started
    np.testing.assert_array_equal(
        association.match_prediction_indices,
        np.arange(size, dtype=np.int64),
    )
    assert association.solver_call_count <= 129
    assert elapsed < 2.0
