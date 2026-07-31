# Trajectory residual summaries

## Requirements

`hjlib-evaluation` owns reusable numerical summaries of a caller-supplied final
trajectory residual population. The primitive consumes one scalar Euclidean
residual per frame and an explicit boolean validity mask.

It must provide:

- a typed per-trajectory summary with exact sufficient statistics;
- exact macro trajectory-weighted and micro valid-frame-weighted reductions;
- explicit quantile semantics;
- no alignment fitting, Dataset traversal, coordinate-system policy, units,
  serialization, or plotting.

This module is independent of the dense protocol-specific `eval_reducer.py` and
does not change that API.

## Mathematical Architecture

For residual vector `r` with shape `(T,)` and caller-supplied final mask `M`, the
valid population is exactly:

```text
R = {r[t] | M[t]}.
```

The supplied mask is authoritative. Every masked-in value must be finite and
non-negative. Masked-out placeholders may be non-finite. The evaluation helper
must not silently shrink the mask by applying another finite filter.

For `N = |R| > 0`:

```text
sum_r = fsum(R)
sum_r2 = fsum(x * x for x in R)
mean = sum_r / N
mse = sum_r2 / N
rmse = sqrt(mse)
median = quantile(R, 0.5, method='linear')
p95 = quantile(R, 0.95, method='linear').
```

A finite residual may overflow when squared. Squared terms, both sufficient
statistics, and every derived scalar must be finite; otherwise summarization
fails instead of publishing a non-finite metric.

For trajectory summaries `S_i`, macro values weight each trajectory equally:

```text
macro_mean = fsum(S_i.residual_sum / S_i.valid_frame_count)
    / trajectory_count
macro_mse = fsum(S_i.residual_squared_sum / S_i.valid_frame_count)
    / trajectory_count.
```

Micro values reconstruct the exact pooled mean and MSE from sufficient
statistics:

```text
micro_mean = fsum(S_i.residual_sum) / fsum(S_i.valid_frame_count)
micro_mse = fsum(S_i.residual_squared_sum) / fsum(S_i.valid_frame_count).
```

The reducer derives every reconstructible metric from sufficient statistics; it
does not trust caller-supplied `mean`, `mse`, or `rmse` fields independently.
Pooled median and p95 are not exposed because they cannot be reconstructed from
per-trajectory summaries.

## Code Architecture

Residence:

```text
src/hjlib_evaluation/trajectory_residual.py
```

Public contracts:

```python
@dataclass(frozen=True, slots=True)
class Trajectory_Residual_Summary:
    valid_frame_count: int
    residual_sum: float
    residual_squared_sum: float
    mean: float
    median: float
    p95: float
    mse: float
    rmse: float


@dataclass(frozen=True, slots=True)
class Trajectory_Residual_Reduction:
    trajectory_count: int
    valid_frame_count: int
    macro_mean: float
    micro_mean: float
    macro_mse: float
    micro_mse: float


def summarize_trajectory_residuals(
    residual: NDArray[np.generic],
    valid_frame_mask: NDArray[np.generic],
) -> Trajectory_Residual_Summary: ...


def reduce_trajectory_residual_summaries(
    summaries: Sequence[Trajectory_Residual_Summary],
) -> Trajectory_Residual_Reduction: ...
```

The implementation normalizes residual values to NumPy `float64`, requires the
mask array itself to have boolean dtype, validates shape and masked-in values,
and uses `math.fsum` for all additive sufficient statistics and reductions.
Squaring runs under strict floating-point error handling, and all published
statistics must be finite.

`Trajectory_Residual_Summary.__post_init__` enforces a positive exact-integer
count, finite non-negative scalar fields, `median <= p95`, and consistency of
`mean`, `mse`, and `rmse` with the sufficient statistics using the same Python
arithmetic used by the factory. It also enforces the feasibility bounds for a
non-negative population:

```text
residual_sum^2 / valid_frame_count
    <= residual_squared_sum
    <= residual_sum^2.
```

The implementation checks equivalent ratio/bounded forms so validation itself
does not overflow, with a scale-aware floating-point allowance. A zero residual
sum requires a zero squared sum. For one valid frame, residual sum, the square
root of squared sum, mean, median, p95, and RMSE must all describe that same
single value. The reducer additionally derives macro and micro values from
`valid_frame_count`, `residual_sum`, and `residual_squared_sum`. A publicly
constructed summary that violates the reconstructible population invariants
therefore fails at its construction boundary and cannot corrupt reduction
semantics.

Top-level exports from `hjlib_evaluation` provide IDE and runtime navigation. The
dataclasses intentionally use unit-neutral field names. A Campaign consumer may
serialize them under metre-specific schema names when its input semantic is
metres.

No paired before/after type or `improved` flag is exposed. Those concepts are
application policy, particularly because a least-squares alignment may improve
MSE while worsening mean Euclidean distance.

## Smoke-Test Standard

Data-free smoke coverage must verify:

- exact summary fields on a known residual vector;
- masked-out non-finite placeholders are allowed;
- masked-in negative or non-finite values fail;
- numeric masks fail rather than being silently cast;
- shape mismatch and empty valid population fail;
- squared-term or additive-statistic overflow fails;
- median and p95 use NumPy linear quantiles;
- direct construction rejects invalid counts, non-finite/negative fields,
  infeasible sufficient statistics, inconsistent derived fields,
  inconsistent single-frame summaries, and `median > p95`;
- unequal-length trajectories distinguish macro from micro results;
- reducer macro and micro values are reconstructed from sufficient statistics;
- micro values equal direct pooled sufficient-statistic calculations;
- empty reduction fails;
- the reduction type exposes no pooled median or p95;
- top-level public imports.

## Migration Boundary

Campaign 03 will construct before/after Euclidean residual vectors from its
absolute-WORLD GT, aligned GT, canonical S0 reference, and final fit mask. It will
call this module separately for each vector and retain Campaign-owned units,
identities, grouping, schemas, and plots.

The task-local reference-preserving Dataset/collate adapter does not move into
this module. It remains Campaign-local until another independent consumer
justifies a separate promotion decision.

## Modification History

- 2026-07-20: Initial durable design recorded for generic trajectory residual
  summary and exact macro/micro reduction.
- 2026-07-20: Mathematical and code architecture review found two important
  issues. The design now rejects squared/statistic overflow, validates public
  summary construction, and reconstructs reducer metrics from sufficient
  statistics rather than independently trusting derived fields.
- 2026-07-20: Focused re-review accepted the overflow disposition but found that
  public summaries also need non-negative-population feasibility bounds. The
  design now specifies overflow-safe feasibility validation and the exact
  single-frame invariant.
- 2026-07-20: Second focused re-review accepted the remaining disposition with
  no new blocker or important finding. Implementation may proceed.
- 2026-07-20: Initial implementation review found strict typing at boolean
  indexing, near-zero feasibility tolerance, complex-input truncation, reduction
  construction invariants, and missing overflow/invariant smoke cases. The
  implementation now uses typed masks, scale-relative comparisons without a
  fixed absolute floor, rejects complex residuals, validates reduction
  feasibility, normalizes additive overflow to `ValueError`, and expands both
  pytest and master smoke coverage.
- 2026-07-20: Focused implementation re-review accepted those fixes but found
  two exact equal-weight identities missing from public reduction validation.
  Single-trajectory and one-frame-per-trajectory reductions now require matching
  macro/micro mean and MSE, with smoke coverage.
- 2026-07-20: Final focused implementation re-review accepted all prior finding
  dispositions. Focused pytest, complete smoke pytest/master runner, public
  import, targeted strict pyright, and diff-check gates passed. Repository-wide
  pyright remains blocked by pre-existing missing gitignored test settings;
  source and changed smoke files are strict-clean.
