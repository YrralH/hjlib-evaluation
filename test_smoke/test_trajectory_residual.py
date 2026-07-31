'''Smoke tests for scalar trajectory residual summaries.'''
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from hjlib_evaluation import (
    Trajectory_Residual_Reduction,
    Trajectory_Residual_Summary,
    reduce_trajectory_residual_summaries,
    summarize_trajectory_residuals,
)


def test_known_summary_and_linear_quantiles() -> None:
    residual = np.array([0.0, 1.0, 2.0, 10.0])
    summary = summarize_trajectory_residuals(
        residual,
        np.array([True, True, True, True]),
    )

    assert summary.valid_frame_count == 4
    assert summary.residual_sum == 13.0
    assert summary.residual_squared_sum == 105.0
    assert summary.mean == 3.25
    assert summary.median == float(
        np.quantile(residual, 0.5, method='linear')
    )
    assert summary.p95 == float(
        np.quantile(residual, 0.95, method='linear')
    )
    assert summary.mse == 26.25
    assert summary.rmse == np.sqrt(26.25)


def test_mask_is_authoritative() -> None:
    summary = summarize_trajectory_residuals(
        np.array([1.0, np.nan, np.inf]),
        np.array([True, False, False]),
    )
    assert summary.valid_frame_count == 1
    assert summary.mean == 1.0


@pytest.mark.parametrize(
    ('call', 'error_type'),
    [
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0]),
                np.array([1]),
            ),
            TypeError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([[1.0]]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0, 2.0]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([-1.0]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([np.nan]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0]),
                np.array([False]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0 + 2.0j]),
                np.array([True]),
            ),
            TypeError,
        ),
        (
            lambda: reduce_trajectory_residual_summaries([]),
            ValueError,
        ),
    ],
)
def test_invalid_summary_inputs_fail(
    call: Callable[[], object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        call()


def test_squared_residual_overflow_fails() -> None:
    with pytest.raises(ValueError, match='overflowed'):
        summarize_trajectory_residuals(
            np.array([np.finfo(np.float64).max]),
            np.array([True]),
        )
    with pytest.raises(ValueError, match='residual sum overflowed'):
        summarize_trajectory_residuals(
            np.array([1.0e308, 1.0e308]),
            np.array([True, True]),
        )
    with pytest.raises(ValueError, match='squared residual sum overflowed'):
        summarize_trajectory_residuals(
            np.array([1.0e154, 1.0e154]),
            np.array([True, True]),
        )


def test_direct_summary_rejects_inconsistent_fields() -> None:
    with pytest.raises(TypeError, match='exact int'):
        Trajectory_Residual_Summary(
            valid_frame_count=True,
            residual_sum=1.0,
            residual_squared_sum=1.0,
            mean=1.0,
            median=1.0,
            p95=1.0,
            mse=1.0,
            rmse=1.0,
        )
    with pytest.raises(ValueError, match='positive'):
        Trajectory_Residual_Summary(
            valid_frame_count=0,
            residual_sum=0.0,
            residual_squared_sum=0.0,
            mean=0.0,
            median=0.0,
            p95=0.0,
            mse=0.0,
            rmse=0.0,
        )
    with pytest.raises(ValueError, match='finite and non-negative'):
        Trajectory_Residual_Summary(
            valid_frame_count=1,
            residual_sum=np.inf,
            residual_squared_sum=np.inf,
            mean=np.inf,
            median=np.inf,
            p95=np.inf,
            mse=np.inf,
            rmse=np.inf,
        )
    with pytest.raises(ValueError, match='lower bound'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=10.0,
            residual_squared_sum=1.0,
            mean=5.0,
            median=1.0,
            p95=5.0,
            mse=0.5,
            rmse=np.sqrt(0.5),
        )
    with pytest.raises(ValueError, match='upper bound'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=1.0,
            residual_squared_sum=2.0,
            mean=0.5,
            median=0.0,
            p95=1.0,
            mse=1.0,
            rmse=1.0,
        )
    with pytest.raises(ValueError, match='lower bound'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=1.0e-100,
            residual_squared_sum=1.0e-214,
            mean=5.0e-101,
            median=0.0,
            p95=1.0e-100,
            mse=5.0e-215,
            rmse=np.sqrt(5.0e-215),
        )
    with pytest.raises(ValueError, match='mean is inconsistent'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=2.0,
            residual_squared_sum=2.0,
            mean=0.5,
            median=1.0,
            p95=1.0,
            mse=1.0,
            rmse=1.0,
        )
    with pytest.raises(ValueError, match='mse is inconsistent'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=2.0,
            residual_squared_sum=2.0,
            mean=1.0,
            median=1.0,
            p95=1.0,
            mse=0.5,
            rmse=np.sqrt(0.5),
        )
    with pytest.raises(ValueError, match='rmse is inconsistent'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=2.0,
            residual_squared_sum=2.0,
            mean=1.0,
            median=1.0,
            p95=1.0,
            mse=1.0,
            rmse=0.5,
        )
    with pytest.raises(ValueError, match='single-frame'):
        Trajectory_Residual_Summary(
            valid_frame_count=1,
            residual_sum=2.0,
            residual_squared_sum=4.0,
            mean=2.0,
            median=1.0,
            p95=2.0,
            mse=4.0,
            rmse=2.0,
        )
    with pytest.raises(ValueError, match='median'):
        Trajectory_Residual_Summary(
            valid_frame_count=2,
            residual_sum=2.0,
            residual_squared_sum=2.0,
            mean=1.0,
            median=2.0,
            p95=1.0,
            mse=1.0,
            rmse=1.0,
        )


def test_macro_and_micro_use_different_weights() -> None:
    short = summarize_trajectory_residuals(
        np.array([10.0]),
        np.array([True]),
    )
    long = summarize_trajectory_residuals(
        np.array([0.0, 0.0, 0.0]),
        np.array([True, True, True]),
    )
    reduction = reduce_trajectory_residual_summaries([short, long])

    assert reduction.trajectory_count == 2
    assert reduction.valid_frame_count == 4
    assert reduction.macro_mean == 5.0
    assert reduction.micro_mean == 2.5
    assert reduction.macro_mse == 50.0
    assert reduction.micro_mse == 25.0
    assert not hasattr(reduction, 'median')
    assert not hasattr(reduction, 'p95')


def test_micro_matches_direct_pooled_sufficient_statistics() -> None:
    first = summarize_trajectory_residuals(
        np.array([1.0, 2.0]),
        np.array([True, True]),
    )
    second = summarize_trajectory_residuals(
        np.array([4.0, 8.0, 16.0]),
        np.array([True, False, True]),
    )
    reduction = reduce_trajectory_residual_summaries([first, second])
    pooled = np.array([1.0, 2.0, 4.0, 16.0])

    assert reduction.micro_mean == float(np.sum(pooled)) / pooled.size
    assert reduction.micro_mse == float(np.sum(pooled * pooled)) / pooled.size


def test_reduction_type_validates_public_fields() -> None:
    with pytest.raises(ValueError):
        Trajectory_Residual_Reduction(
            trajectory_count=0,
            valid_frame_count=1,
            macro_mean=0.0,
            micro_mean=0.0,
            macro_mse=0.0,
            micro_mse=0.0,
        )
    with pytest.raises(ValueError, match='at least trajectory_count'):
        Trajectory_Residual_Reduction(
            trajectory_count=2,
            valid_frame_count=1,
            macro_mean=0.0,
            micro_mean=0.0,
            macro_mse=0.0,
            micro_mse=0.0,
        )
    with pytest.raises(ValueError, match='macro_mse'):
        Trajectory_Residual_Reduction(
            trajectory_count=1,
            valid_frame_count=1,
            macro_mean=2.0,
            micro_mean=2.0,
            macro_mse=1.0,
            micro_mse=4.0,
        )
    with pytest.raises(ValueError, match='micro_mse'):
        Trajectory_Residual_Reduction(
            trajectory_count=1,
            valid_frame_count=1,
            macro_mean=2.0,
            micro_mean=2.0,
            macro_mse=4.0,
            micro_mse=1.0,
        )
    with pytest.raises(ValueError, match='macro_mean and micro_mean'):
        Trajectory_Residual_Reduction(
            trajectory_count=1,
            valid_frame_count=3,
            macro_mean=1.0,
            micro_mean=2.0,
            macro_mse=1.0,
            micro_mse=4.0,
        )
    with pytest.raises(ValueError, match='macro_mean and micro_mean'):
        Trajectory_Residual_Reduction(
            trajectory_count=2,
            valid_frame_count=2,
            macro_mean=1.0,
            micro_mean=2.0,
            macro_mse=1.0,
            micro_mse=4.0,
        )


def smoke_test_trajectory_residual() -> None:
    test_known_summary_and_linear_quantiles()
    test_mask_is_authoritative()
    invalid_cases: tuple[tuple[Callable[[], object], type[Exception]], ...] = (
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0]),
                np.array([1]),
            ),
            TypeError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([[1.0]]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0, 2.0]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([-1.0]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([np.nan]),
                np.array([True]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0]),
                np.array([False]),
            ),
            ValueError,
        ),
        (
            lambda: summarize_trajectory_residuals(
                np.array([1.0 + 2.0j]),
                np.array([True]),
            ),
            TypeError,
        ),
        (
            lambda: reduce_trajectory_residual_summaries([]),
            ValueError,
        ),
    )
    for call, error_type in invalid_cases:
        test_invalid_summary_inputs_fail(call, error_type)
    test_squared_residual_overflow_fails()
    test_direct_summary_rejects_inconsistent_fields()
    test_macro_and_micro_use_different_weights()
    test_micro_matches_direct_pooled_sufficient_statistics()
    test_reduction_type_validates_public_fields()


if __name__ == '__main__':
    smoke_test_trajectory_residual()
    print('trajectory residual smoke tests passed')
