'''Smoke tests for sparse SMPL occurrence MPJPE/T-MPJPE reduction.'''
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from hjlib_evaluation import (
    Metric_Spec_3D,
    SMPL_Joint_Occurrence_Result,
    compute_smpl_joint_occurrence_metric,
    reduce_smpl_joint_occurrences,
)


METRIC = Metric_Spec_3D(
    name='known',
    joint_indices_smpl_54=(1, 2),
    root_indices_smpl_54_for_alignment=(1, 2),
)


def joints(count: int = 2, joint_count: int = 24) -> np.ndarray:
    return np.zeros((count, joint_count, 3), dtype=np.float32)


def ids(count: int = 2) -> np.ndarray:
    return np.arange(count, dtype=np.int64)


def reduce_known(
    pred: np.ndarray,
    gt: np.ndarray,
) -> SMPL_Joint_Occurrence_Result:
    occurrence_ids = ids(pred.shape[0])
    return compute_smpl_joint_occurrence_metric(
        pred,
        gt,
        occurrence_ids,
        occurrence_ids,
        METRIC,
        joint_layout='smpl_24',
        unit_world='m',
        pred_coordinate_frame='camera',
        gt_coordinate_frame='camera',
    )


def test_known_translation_and_root_alignment() -> None:
    pred = joints()
    gt = joints()
    pred[:, 1:3, 0] = 0.01
    result = reduce_known(pred, gt)
    assert result.metric_name == 'known'
    assert result.num_occurrence == 2
    assert result.num_joint == 2
    assert result.mpjpe_mm == pytest.approx(10.0)
    assert result.t_mpjpe_mm == pytest.approx(0.0)


def test_occurrence_weighting_and_common_permutation() -> None:
    pred = joints(3)
    gt = joints(3)
    pred[2, 1:3, 0] = 0.03
    occurrence_ids = ids(3)
    expected = compute_smpl_joint_occurrence_metric(
        pred,
        gt,
        occurrence_ids,
        occurrence_ids,
        METRIC,
        joint_layout='smpl_24',
        unit_world='m',
        pred_coordinate_frame='camera',
        gt_coordinate_frame='camera',
    )
    order = np.array([2, 0, 1])
    actual = compute_smpl_joint_occurrence_metric(
        pred[order],
        gt[order],
        occurrence_ids[order],
        occurrence_ids[order],
        METRIC,
        joint_layout='smpl_24',
        unit_world='m',
        pred_coordinate_frame='camera',
        gt_coordinate_frame='camera',
    )
    assert expected.mpjpe_mm == pytest.approx(10.0)
    assert actual == expected


def test_all_54_and_multiple_specs() -> None:
    pred = joints(joint_count=54)
    gt = joints(joint_count=54)
    results = reduce_smpl_joint_occurrences(
        pred,
        gt,
        ids(),
        ids(),
        (
            METRIC,
            Metric_Spec_3D('second', (3,), (1, 2)),
        ),
        joint_layout='smpl_all_54',
        unit_world='mm',
        pred_coordinate_frame='world',
        gt_coordinate_frame='world',
    )
    assert tuple(item.metric_name for item in results) == ('known', 'second')
    assert all(item.mpjpe_mm == 0.0 for item in results)


@pytest.mark.parametrize(
    'call',
    [
        lambda: reduce_known(joints(0), joints(0)),
        lambda: reduce_known(joints().astype(np.float16), joints()),
        lambda: reduce_known(joints(), joints(3)),
        lambda: compute_smpl_joint_occurrence_metric(
            joints(), joints(), ids(), ids()[::-1], METRIC,
            joint_layout='smpl_24', unit_world='m',
            pred_coordinate_frame='camera', gt_coordinate_frame='camera',
        ),
        lambda: compute_smpl_joint_occurrence_metric(
            joints(), joints(), np.array([0, 0], dtype=np.int64),
            np.array([0, 0], dtype=np.int64), METRIC,
            joint_layout='smpl_24', unit_world='m',
            pred_coordinate_frame='camera', gt_coordinate_frame='camera',
        ),
        lambda: compute_smpl_joint_occurrence_metric(
            joints(), joints(), ids(), ids(), METRIC,
            joint_layout='smpl_24', unit_world='m',
            pred_coordinate_frame='camera', gt_coordinate_frame='world',
        ),
        lambda: reduce_smpl_joint_occurrences(
            joints(), joints(), ids(), ids(), (),
            joint_layout='smpl_24', unit_world='m',
            pred_coordinate_frame='camera', gt_coordinate_frame='camera',
        ),
    ],
)
def test_invalid_population_inputs_fail(call: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()


@pytest.mark.parametrize(
    'metric',
    [
        Metric_Spec_3D('empty_subject', (), (1,)),
        Metric_Spec_3D('empty_root', (1,), ()),
        Metric_Spec_3D('negative', (-1,), (1,)),
        Metric_Spec_3D('outside', (24,), (1,)),
        Metric_Spec_3D('duplicate_subject', (1, 1), (2,)),
        Metric_Spec_3D('duplicate_root', (1,), (2, 2)),
        Metric_Spec_3D('bool_index', (True,), (2,)),
    ],
)
def test_invalid_metric_indices_fail(metric: Metric_Spec_3D) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_smpl_joint_occurrence_metric(
            joints(),
            joints(),
            ids(),
            ids(),
            metric,
            joint_layout='smpl_24',
            unit_world='m',
            pred_coordinate_frame='camera',
            gt_coordinate_frame='camera',
        )


def test_nonfinite_selected_and_invalid_declarations_fail() -> None:
    bad = joints()
    bad[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        reduce_known(bad, joints())
    with pytest.raises(ValueError, match='unit_world'):
        compute_smpl_joint_occurrence_metric(
            joints(), joints(), ids(), ids(), METRIC,
            joint_layout='smpl_24', unit_world='cm',  # type: ignore[arg-type]
            pred_coordinate_frame='camera', gt_coordinate_frame='camera',
        )
    with pytest.raises(ValueError, match='layout'):
        compute_smpl_joint_occurrence_metric(
            joints(), joints(), ids(), ids(), METRIC,
            joint_layout='bad', unit_world='m',  # type: ignore[arg-type]
            pred_coordinate_frame='camera', gt_coordinate_frame='camera',
        )
