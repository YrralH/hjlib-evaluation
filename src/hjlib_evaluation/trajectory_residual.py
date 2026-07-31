'''Summarize and reduce scalar trajectory residual populations.'''
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray


_FLOAT_TOLERANCE = 64.0 * np.finfo(np.float64).eps


def _close(left: float, right: float) -> bool:
    if left == right:
        return True
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return False
    return abs(left - right) / scale <= _FLOAT_TOLERANCE


def _require_finite_nonnegative(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError('%s must be finite and non-negative' % name)


@dataclass(frozen=True, slots=True)
class Trajectory_Residual_Summary:
    '''Summary of one caller-defined scalar residual population.'''

    valid_frame_count: int
    residual_sum: float
    residual_squared_sum: float
    mean: float
    median: float
    p95: float
    mse: float
    rmse: float

    def __post_init__(self) -> None:
        if type(self.valid_frame_count) is not int:
            raise TypeError('valid_frame_count must be an exact int')
        if self.valid_frame_count <= 0:
            raise ValueError('valid_frame_count must be positive')
        for name in (
            'residual_sum',
            'residual_squared_sum',
            'mean',
            'median',
            'p95',
            'mse',
            'rmse',
        ):
            _require_finite_nonnegative(
                float(getattr(self, name)),
                name=name,
            )
        if self.median > self.p95:
            raise ValueError('median must not exceed p95')

        expected_mean = self.residual_sum / self.valid_frame_count
        expected_mse = self.residual_squared_sum / self.valid_frame_count
        expected_rmse = math.sqrt(expected_mse)
        if not _close(self.mean, expected_mean):
            raise ValueError('mean is inconsistent with sufficient statistics')
        if not _close(self.mse, expected_mse):
            raise ValueError('mse is inconsistent with sufficient statistics')
        if not _close(self.rmse, expected_rmse):
            raise ValueError('rmse is inconsistent with sufficient statistics')

        if self.residual_sum == 0.0:
            if self.residual_squared_sum != 0.0:
                raise ValueError(
                    'zero residual_sum requires zero residual_squared_sum'
                )
        else:
            squared_to_sum = self.residual_squared_sum / self.residual_sum
            lower = self.residual_sum / self.valid_frame_count
            upper = self.residual_sum
            if squared_to_sum < lower and not _close(squared_to_sum, lower):
                raise ValueError(
                    'sufficient statistics violate the population lower bound'
                )
            if squared_to_sum > upper and not _close(squared_to_sum, upper):
                raise ValueError(
                    'sufficient statistics violate the population upper bound'
                )

        if self.valid_frame_count == 1:
            single_value = self.residual_sum
            single_fields = {
                'mean': self.mean,
                'median': self.median,
                'p95': self.p95,
                'rmse': self.rmse,
            }
            for name, value in single_fields.items():
                if not _close(value, single_value):
                    raise ValueError(
                        '%s is inconsistent with a single-frame summary' % name
                    )


@dataclass(frozen=True, slots=True)
class Trajectory_Residual_Reduction:
    '''Macro and micro reduction across trajectory summaries.'''

    trajectory_count: int
    valid_frame_count: int
    macro_mean: float
    micro_mean: float
    macro_mse: float
    micro_mse: float

    def __post_init__(self) -> None:
        if type(self.trajectory_count) is not int:
            raise TypeError('trajectory_count must be an exact int')
        if type(self.valid_frame_count) is not int:
            raise TypeError('valid_frame_count must be an exact int')
        if self.trajectory_count <= 0:
            raise ValueError('trajectory_count must be positive')
        if self.valid_frame_count <= 0:
            raise ValueError('valid_frame_count must be positive')
        for name in ('macro_mean', 'micro_mean', 'macro_mse', 'micro_mse'):
            _require_finite_nonnegative(float(getattr(self, name)), name=name)
        if self.valid_frame_count < self.trajectory_count:
            raise ValueError(
                'valid_frame_count must be at least trajectory_count'
            )
        if self.macro_mean == 0.0:
            if self.macro_mse != 0.0:
                raise ValueError('zero macro_mean requires zero macro_mse')
        elif self.macro_mse / self.macro_mean < self.macro_mean and not _close(
            self.macro_mse / self.macro_mean,
            self.macro_mean,
        ):
            raise ValueError('macro_mse must be at least macro_mean squared')
        if self.micro_mean == 0.0:
            if self.micro_mse != 0.0:
                raise ValueError('zero micro_mean requires zero micro_mse')
        elif self.micro_mse / self.micro_mean < self.micro_mean and not _close(
            self.micro_mse / self.micro_mean,
            self.micro_mean,
        ):
            raise ValueError('micro_mse must be at least micro_mean squared')
        if (
            self.trajectory_count == 1
            or self.valid_frame_count == self.trajectory_count
        ):
            if not _close(self.macro_mean, self.micro_mean):
                raise ValueError(
                    'macro_mean and micro_mean must match for equal weights'
                )
            if not _close(self.macro_mse, self.micro_mse):
                raise ValueError(
                    'macro_mse and micro_mse must match for equal weights'
                )


def summarize_trajectory_residuals(
    residual: NDArray[np.generic],
    valid_frame_mask: NDArray[np.generic],
) -> Trajectory_Residual_Summary:
    '''Summarize one scalar residual vector on the exact supplied mask.'''
    residual_array = np.asarray(residual)
    mask = np.asarray(valid_frame_mask)
    if not np.issubdtype(residual_array.dtype, np.number):
        raise TypeError('residual must have numeric dtype')
    if np.issubdtype(residual_array.dtype, np.complexfloating):
        raise TypeError('residual must have real numeric dtype')
    if residual_array.ndim != 1:
        raise ValueError(
            'residual must have shape (T,), got %s' % (residual_array.shape,)
        )
    if mask.dtype != np.bool_:
        raise TypeError('valid_frame_mask must have boolean dtype')
    if mask.shape != residual_array.shape:
        raise ValueError(
            'valid_frame_mask shape %s does not match residual shape %s'
            % (mask.shape, residual_array.shape)
        )

    residual_float64 = np.asarray(residual_array, dtype=np.float64)
    bool_mask = cast(NDArray[np.bool_], mask)
    valid = residual_float64[bool_mask]
    if valid.size == 0:
        raise ValueError('trajectory residual population is empty')
    if not np.isfinite(valid).all():
        raise ValueError('masked-in residual values must be finite')
    if np.any(valid < 0.0):
        raise ValueError('masked-in residual values must be non-negative')

    try:
        residual_sum = math.fsum(float(value) for value in valid)
    except OverflowError as err:
        raise ValueError('residual sum overflowed') from err
    try:
        with np.errstate(over='raise', invalid='raise'):
            squared = np.multiply(valid, valid)
    except FloatingPointError as err:
        raise ValueError('squared residual values overflowed') from err
    try:
        residual_squared_sum = math.fsum(float(value) for value in squared)
    except OverflowError as err:
        raise ValueError('squared residual sum overflowed') from err
    if not math.isfinite(residual_sum):
        raise ValueError('residual sum overflowed')
    if not math.isfinite(residual_squared_sum):
        raise ValueError('squared residual sum overflowed')

    valid_frame_count = int(valid.size)
    mean = residual_sum / valid_frame_count
    mse = residual_squared_sum / valid_frame_count
    rmse = math.sqrt(mse)
    median = float(np.quantile(valid, 0.5, method='linear'))
    p95 = float(np.quantile(valid, 0.95, method='linear'))
    for name, value in (
        ('mean', mean),
        ('median', median),
        ('p95', p95),
        ('mse', mse),
        ('rmse', rmse),
    ):
        if not math.isfinite(value):
            raise ValueError('%s is not finite' % name)

    return Trajectory_Residual_Summary(
        valid_frame_count=valid_frame_count,
        residual_sum=residual_sum,
        residual_squared_sum=residual_squared_sum,
        mean=mean,
        median=median,
        p95=p95,
        mse=mse,
        rmse=rmse,
    )


def reduce_trajectory_residual_summaries(
    summaries: Sequence[Trajectory_Residual_Summary],
) -> Trajectory_Residual_Reduction:
    '''Reduce trajectory summaries with trajectory- and frame-weighted views.'''
    summary_tuple = tuple(summaries)
    if not summary_tuple:
        raise ValueError('trajectory residual summary collection is empty')
    trajectory_count = len(summary_tuple)
    valid_frame_count = sum(
        summary.valid_frame_count for summary in summary_tuple
    )
    try:
        residual_sum = math.fsum(
            summary.residual_sum for summary in summary_tuple
        )
        residual_squared_sum = math.fsum(
            summary.residual_squared_sum for summary in summary_tuple
        )
        macro_mean = math.fsum(
            summary.residual_sum / summary.valid_frame_count
            for summary in summary_tuple
        ) / trajectory_count
        macro_mse = math.fsum(
            summary.residual_squared_sum / summary.valid_frame_count
            for summary in summary_tuple
        ) / trajectory_count
    except OverflowError as err:
        raise ValueError('trajectory residual reduction overflowed') from err
    micro_mean = residual_sum / valid_frame_count
    micro_mse = residual_squared_sum / valid_frame_count
    for name, value in (
        ('residual_sum', residual_sum),
        ('residual_squared_sum', residual_squared_sum),
        ('macro_mean', macro_mean),
        ('micro_mean', micro_mean),
        ('macro_mse', macro_mse),
        ('micro_mse', micro_mse),
    ):
        if not math.isfinite(value):
            raise ValueError('%s is not finite' % name)

    return Trajectory_Residual_Reduction(
        trajectory_count=trajectory_count,
        valid_frame_count=valid_frame_count,
        macro_mean=macro_mean,
        micro_mean=micro_mean,
        macro_mse=macro_mse,
        micro_mse=micro_mse,
    )
