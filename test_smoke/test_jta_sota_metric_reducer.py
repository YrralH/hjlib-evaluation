'''Smoke tests for the canonical JTA SOTA six-metric reducer.'''
from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from hjlib_evaluation import (
    JTA_ENDPOINT_OKS_SIGMAS,
    JTA_SOTA_METRIC_PROFILE,
    SMPL54_ENDPOINT_INDICES,
    JTA_SOTA_Metric_Sums,
    compute_jta_sota_metric_sums,
    compute_paired_keypoint_oks,
    finalize_jta_sota_metric_sums,
    validate_jta_sota_occurrence_partition,
)
from hjlib_evaluation.jta_sota_metric_reducer import JTA_SOTA_JOINT_COUNT


Float_Array = NDArray[np.float64]


def base_joints(count: int = 2) -> Float_Array:
    one = np.empty((24, 3), dtype=np.float64)
    for index in range(24):
        one[index] = (
            0.03 * (index % 4),
            0.02 * (index // 4),
            1.0 + 0.01 * (index % 3),
        )
    return np.repeat(one[None, :, :], count, axis=0)


def image_inputs(
        joints: Float_Array,
    ) -> tuple[Float_Array, Float_Array]:
    full = joints[:, :, :2] * 1000.0 + np.asarray((500.0, 300.0))
    endpoint = full[:, SMPL54_ENDPOINT_INDICES, :]
    return endpoint, full


def metric_sums(
        pred: Float_Array,
        gt: Float_Array,
        *,
        pred_xy: Float_Array | None = None,
        gt_xy: Float_Array | None = None,
        gt_full_xy: Float_Array | None = None,
        unit_world: Literal['m', 'mm'] = 'm',
    ) -> JTA_SOTA_Metric_Sums:
    default_gt_xy, default_gt_full_xy = image_inputs(gt)
    if gt_xy is None:
        gt_xy = default_gt_xy
    if gt_full_xy is None:
        gt_full_xy = default_gt_full_xy
    if pred_xy is None:
        pred_xy = gt_xy
    ids = np.arange(pred.shape[0], dtype=np.int64)
    return compute_jta_sota_metric_sums(
        pred,
        gt,
        ids,
        ids,
        pred_xy,
        gt_xy,
        gt_full_xy,
        joint_layout='smpl_24',
        unit_world=unit_world,
        pred_coordinate_frame='camera',
        gt_coordinate_frame='camera',
        image_coordinate_frame='full_image_pixel',
    )


def test_perfect_and_translation_metrics() -> None:
    gt = base_joints()
    perfect = finalize_jta_sota_metric_sums(metric_sums(gt.copy(), gt))
    assert perfect.profile == JTA_SOTA_METRIC_PROFILE
    assert perfect.num_occurrence == 2
    assert perfect.num_joint == JTA_SOTA_JOINT_COUNT
    assert perfect.root_error_mm == pytest.approx(0.0, abs=1.0e-10)
    assert perfect.mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert perfect.t_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert perfect.rt_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert perfect.pa_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert perfect.oks == pytest.approx(1.0)

    offset = np.asarray((0.01, -0.02, 0.03), dtype=np.float64)
    translated = finalize_jta_sota_metric_sums(metric_sums(gt + offset, gt))
    expected = float(np.linalg.norm(offset) * 1000.0)
    assert translated.root_error_mm == pytest.approx(expected)
    assert translated.mpjpe_mm == pytest.approx(expected)
    assert translated.t_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert translated.rt_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert translated.pa_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)


def test_rigid_similarity_and_reflection_behavior() -> None:
    gt = base_joints(1)
    angle = np.deg2rad(37.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rigid_pred = gt @ rotation.T + np.asarray((0.4, -0.2, 0.1))
    rigid = finalize_jta_sota_metric_sums(metric_sums(rigid_pred, gt))
    assert rigid.rt_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)
    assert rigid.pa_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)

    scaled_pred = rigid_pred * 1.7
    scaled = finalize_jta_sota_metric_sums(metric_sums(scaled_pred, gt))
    assert scaled.rt_mpjpe_mm > 0.0
    assert scaled.pa_mpjpe_mm == pytest.approx(0.0, abs=1.0e-10)

    reflected_pred = gt.copy()
    reflected_pred[:, :, 0] *= -1.0
    reflected = finalize_jta_sota_metric_sums(metric_sums(reflected_pred, gt))
    assert reflected.rt_mpjpe_mm > 0.0
    assert reflected.pa_mpjpe_mm > 0.0


def test_addition_units_metadata_and_denominators() -> None:
    gt = base_joints(3)
    pred = gt.copy()
    pred[1] += np.asarray((0.01, 0.0, 0.0))
    pred[2] += np.asarray((0.0, 0.02, 0.0))
    full = metric_sums(pred, gt)
    first = metric_sums(pred[:1], gt[:1])
    second = metric_sums(pred[1:], gt[1:])
    combined = first.plus(second)
    assert combined.num_occurrence == 3
    assert combined.root_denominator == 3
    assert combined.joint_denominator == 36
    assert combined.oks_denominator == 3
    for name in (
            'root_error_sum_mm', 'mpjpe_sum_mm', 't_mpjpe_sum_mm',
            'rt_mpjpe_sum_mm', 'pa_mpjpe_sum_mm', 'oks_sum',
        ):
        assert getattr(combined, name) == pytest.approx(getattr(full, name))

    millimetre = metric_sums(pred * 1000.0, gt * 1000.0, unit_world='mm')
    assert millimetre.mpjpe_sum_mm == pytest.approx(full.mpjpe_sum_mm)
    assert millimetre.root_error_sum_mm == pytest.approx(full.root_error_sum_mm)
    with pytest.raises(ValueError, match='metadata'):
        first.plus(replace(second, image_coordinate_frame='crop_pixel'))


def test_paired_oks_invariance_and_envelope_rotation() -> None:
    gt = base_joints(1)
    gt_xy, gt_full_xy = image_inputs(gt)
    pred_xy = gt_xy + np.asarray((2.0, -3.0))
    area = np.prod(gt_full_xy.max(axis=1) - gt_full_xy.min(axis=1), axis=1)
    baseline = compute_paired_keypoint_oks(
        gt_xy,
        pred_xy,
        area,
        JTA_ENDPOINT_OKS_SIGMAS,
    )
    scale = 3.5
    translation = np.asarray((91.0, -44.0))
    transformed_gt = gt_xy * scale + translation
    transformed_pred = pred_xy * scale + translation
    transformed_full = gt_full_xy * scale + translation
    transformed_area = np.prod(
        transformed_full.max(axis=1) - transformed_full.min(axis=1),
        axis=1,
    )
    transformed = compute_paired_keypoint_oks(
        transformed_gt,
        transformed_pred,
        transformed_area,
        JTA_ENDPOINT_OKS_SIGMAS,
    )
    np.testing.assert_allclose(transformed, baseline)

    angle = np.deg2rad(31.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    rotated_gt = gt_xy @ rotation.T
    rotated_pred = pred_xy @ rotation.T
    rotated_full = gt_full_xy @ rotation.T
    rotated_area = np.prod(
        rotated_full.max(axis=1) - rotated_full.min(axis=1),
        axis=1,
    )
    rotated = compute_paired_keypoint_oks(
        rotated_gt,
        rotated_pred,
        rotated_area,
        JTA_ENDPOINT_OKS_SIGMAS,
    )
    assert not np.allclose(rotated, baseline)


def test_invalid_spread_area_shapes_and_partition_fail() -> None:
    gt = base_joints(1)
    gt_xy, gt_full_xy = image_inputs(gt)
    degenerate = np.zeros_like(gt)
    with pytest.raises(ValueError, match='spread'):
        metric_sums(degenerate, gt)
    zero_area = gt_full_xy.copy()
    zero_area[:, :, 0] = 1.0
    with pytest.raises(ValueError, match='positive'):
        metric_sums(gt, gt, gt_full_xy=zero_area)
    with pytest.raises(ValueError, match='shape'):
        metric_sums(gt, gt, pred_xy=gt_xy[:, :-1])

    expected = np.asarray((2, 0, 1), dtype=np.int64)
    validate_jta_sota_occurrence_partition(
        expected,
        (
            np.asarray((2, 0), dtype=np.int64),
            np.asarray((1,), dtype=np.int64),
        ),
    )
    with pytest.raises(ValueError, match='differs'):
        validate_jta_sota_occurrence_partition(
            expected,
            (
                np.asarray((0, 2), dtype=np.int64),
                np.asarray((1,), dtype=np.int64),
            ),
        )
    with pytest.raises(ValueError, match='unique'):
        validate_jta_sota_occurrence_partition(
            np.asarray((1, 1), dtype=np.int64),
            (np.asarray((1, 1), dtype=np.int64),),
        )


def smoke_test_jta_sota_metric_reducer() -> None:
    test_perfect_and_translation_metrics()
    test_rigid_similarity_and_reflection_behavior()
    test_addition_units_metadata_and_denominators()
    test_paired_oks_invariance_and_envelope_rotation()
    test_invalid_spread_area_shapes_and_partition_fail()


if __name__ == '__main__':
    smoke_test_jta_sota_metric_reducer()
