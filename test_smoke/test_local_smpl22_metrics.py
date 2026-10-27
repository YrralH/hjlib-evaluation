'''Data-free local SMPL22 alignment, support and population controls.'''

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

from hjlib_evaluation import (
    compute_local_smpl22_rt_mpjpe_values,
    compute_local_smpl22_t_mpjpe_values,
)


def local_smpl22_points() -> NDArray[np.float64]:
    '''Build deterministic noncoplanar geometry with a nonzero world pelvis.'''
    return np.random.default_rng(174).normal(size=(2, 22, 3)) + [2.0, -1.0, 3.0]


def local_smpl22_zero_mean_basis() -> NDArray[np.float64]:
    '''Provide orthogonal point channels to isolate rank and covariance gates.'''
    raw = np.random.default_rng(981).normal(size=(22, 5))
    raw -= raw.mean(axis=0)
    basis, _ = np.linalg.qr(raw)
    return basis


def test_local_smpl22_translation_rotation_and_framewise_alignment() -> None:
    reference = local_smpl22_points()
    translated = reference + np.array([6.0, -4.0, 8.0])
    t_values = compute_local_smpl22_t_mpjpe_values(translated, reference)
    assert np.allclose(t_values, 0.0, atol=1e-14)
    assert t_values.shape == (2, 22) and t_values.dtype == np.float64

    rotations = np.array([
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    ])
    rotated = np.einsum('nij,nkj->nki', rotations, reference)
    rotated += np.array([[3.0, -7.0, 2.0], [-4.0, 5.0, 8.0]])[:, None]
    assert float(compute_local_smpl22_t_mpjpe_values(rotated, reference).mean()) > 0.1
    rt_values = compute_local_smpl22_rt_mpjpe_values(rotated, reference)
    assert np.allclose(rt_values, 0.0, atol=1e-13)
    assert rt_values.shape == (2, 22) and rt_values.dtype == np.float64

    at_origin = reference - reference[:, :1]
    assert np.allclose(
        compute_local_smpl22_t_mpjpe_values(at_origin, reference),
        0.0,
        atol=1e-14,
    )
    assert np.allclose(
        compute_local_smpl22_rt_mpjpe_values(at_origin, reference),
        0.0,
        atol=1e-13,
    )


def test_local_smpl22_rt_does_not_fit_scale_or_reflection() -> None:
    reference = local_smpl22_points()
    scaled = 1.8 * reference + np.array([5.0, 2.0, -3.0])
    assert float(compute_local_smpl22_rt_mpjpe_values(scaled, reference).mean()) > 0.1
    mirrored = reference.copy()
    mirrored[..., 0] *= -1.0
    assert float(compute_local_smpl22_rt_mpjpe_values(mirrored, reference).mean()) > 0.1


def test_local_smpl22_selection_precedes_fit_and_finite_checks() -> None:
    reference = local_smpl22_points()
    predicted = reference.copy()
    predicted[0, 20, 0] += 0.4
    predicted[1, 21, 1] -= 0.6
    predicted_extended = np.concatenate((predicted, np.full((2, 2, 3), np.nan)), axis=1)
    reference_extended = np.concatenate((reference, np.full((2, 32, 3), np.inf)), axis=1)
    predicted_extended[:, 22, 0] = 1e300
    for metric in (
            compute_local_smpl22_t_mpjpe_values,
            compute_local_smpl22_rt_mpjpe_values,
        ):
        baseline = metric(predicted, reference)
        with_extra = metric(predicted_extended, reference_extended)
        assert np.array_equal(with_extra, baseline)
        assert with_extra.shape == (2, 22)
        assert not with_extra.flags.writeable
        with pytest.raises(ValueError):
            with_extra[0, 0] = 2.0
        with pytest.raises(ValueError):
            with_extra.setflags(write=True)
    assert np.isnan(predicted_extended[0, 23]).all()
    assert np.isinf(reference_extended[:, 22:]).all()


def test_local_smpl22_wrists_and_pelvis_stay_in_micro_denominator() -> None:
    reference = local_smpl22_points()
    predicted = reference.copy()
    predicted[0, 20, 0] += 1.0
    predicted[1, 21, 0] += 3.0
    values = compute_local_smpl22_t_mpjpe_values(predicted, reference)
    assert np.array_equal(values[:, 0], np.zeros(2))
    assert values[0, 20] == pytest.approx(1.0)
    assert values[1, 21] == pytest.approx(3.0)
    assert values.size == 44
    assert float(values.sum()) == pytest.approx(4.0)
    assert 1000.0 * float(values.sum()) / values.size == pytest.approx(4000.0 / 44.0)
    assert values.size * 2550 == 5100 * 22 == 112200


def test_local_smpl22_rejects_invalid_shape_type_support_and_selected_nonfinite() -> None:
    points = local_smpl22_points()
    invalid: tuple[tuple[NDArray[np.generic], NDArray[np.generic]], ...] = (
        (points[:, :21], points),
        (points[0], points[0]),
        (points[..., :2], points[..., :2]),
        (points[:0], points[:0]),
        (points[:1], points),
        (points.astype(np.complex128), points),
        (points.astype(np.str_), points),
        (points.astype(np.bool_), points),
    )
    for metric in (
            compute_local_smpl22_t_mpjpe_values,
            compute_local_smpl22_rt_mpjpe_values,
        ):
        for predicted, reference in invalid:
            with pytest.raises((ValueError, TypeError)):
                metric(predicted, reference)
        for bad_value in (np.nan, np.inf, -np.inf):
            bad = points.copy()
            bad[0, 21, 2] = bad_value
            with pytest.raises(ValueError, match='finite'):
                metric(bad, points)
            with pytest.raises(ValueError, match='finite'):
                metric(points, bad)


def test_local_smpl22_rt_planar_support_and_degenerate_controls() -> None:
    planar = local_smpl22_points()
    planar[..., 2] = 0.0
    assert np.allclose(
        compute_local_smpl22_rt_mpjpe_values(planar + [2.0, -1.0, 4.0], planar),
        0.0,
        atol=1e-13,
    )
    coincident = np.zeros((2, 22, 3))
    collinear = np.zeros_like(coincident)
    collinear[..., 0] = np.arange(22)
    for bad in (coincident, collinear):
        with pytest.raises(ValueError, match='rank >= 2'):
            compute_local_smpl22_rt_mpjpe_values(bad, planar)
        with pytest.raises(ValueError, match='rank >= 2'):
            compute_local_smpl22_rt_mpjpe_values(planar, bad)
        assert np.isfinite(compute_local_smpl22_t_mpjpe_values(bad, planar)).all()


def test_local_smpl22_rt_rejects_rank1_cross_covariance() -> None:
    basis = local_smpl22_zero_mean_basis()
    predicted = basis[:, :3][None]
    reference = basis[:, [0, 3, 4]][None]
    assert np.linalg.matrix_rank(predicted[0]) == 3
    assert np.linalg.matrix_rank(reference[0]) == 3
    with pytest.raises(ValueError, match='cross-covariance'):
        compute_local_smpl22_rt_mpjpe_values(predicted, reference)


def test_local_smpl22_rt_point_and_covariance_tolerances() -> None:
    basis = local_smpl22_zero_mean_basis()
    predicted = np.zeros((1, 22, 3))
    reference = np.zeros_like(predicted)
    predicted[0, :, 0] = basis[:, 0]
    predicted[0, :, 1] = basis[:, 1] * 1e-11
    reference[0, :, :2] = basis[:, :2]
    with pytest.raises(ValueError, match='predicted points'):
        compute_local_smpl22_rt_mpjpe_values(predicted, reference)
    predicted[0, :, :2] = basis[:, :2] * 1e-13
    with pytest.raises(ValueError, match='predicted points'):
        compute_local_smpl22_rt_mpjpe_values(predicted, reference)

    predicted[0, :, 0] = basis[:, 0]
    predicted[0, :, 1] = basis[:, 1] * 5e-6
    with pytest.raises(ValueError, match='cross-covariance'):
        compute_local_smpl22_rt_mpjpe_values(predicted, predicted)
    predicted[0, :, 1] = basis[:, 1] * 1e-4
    assert np.allclose(
        compute_local_smpl22_rt_mpjpe_values(predicted, predicted), 0.0, atol=1e-14)

    predicted[0, :, 0] = basis[:, 0] * 1e-10
    predicted[0, :, 1] = basis[:, 1] * 2e-12
    reference[0, :, 0] = basis[:, 0] * 1e-10
    reference[0, :, 1] = (basis[:, 1] * 0.1 + basis[:, 2] * np.sqrt(0.99)) * 2e-12
    with pytest.raises(ValueError, match='cross-covariance'):
        compute_local_smpl22_rt_mpjpe_values(predicted, reference)
    tiny_valid = np.zeros((1, 22, 3))
    tiny_valid[0, :, :2] = basis[:, :2] * 2e-12
    assert np.allclose(
        compute_local_smpl22_rt_mpjpe_values(tiny_valid, tiny_valid), 0.0, atol=1e-25)


def smoke_test_local_smpl22_metrics() -> None:
    tests: tuple[Callable[[], None], ...] = (
        test_local_smpl22_translation_rotation_and_framewise_alignment,
        test_local_smpl22_rt_does_not_fit_scale_or_reflection,
        test_local_smpl22_selection_precedes_fit_and_finite_checks,
        test_local_smpl22_wrists_and_pelvis_stay_in_micro_denominator,
        test_local_smpl22_rejects_invalid_shape_type_support_and_selected_nonfinite,
        test_local_smpl22_rt_planar_support_and_degenerate_controls,
        test_local_smpl22_rt_rejects_rank1_cross_covariance,
        test_local_smpl22_rt_point_and_covariance_tolerances,
    )
    for test in tests:
        test()
    print('test_local_smpl22_metrics: %s checks passed' % len(tests))


if __name__ == '__main__':
    smoke_test_local_smpl22_metrics()
