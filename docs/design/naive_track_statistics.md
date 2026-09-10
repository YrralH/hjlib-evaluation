# GT-MOT sequence statistics

The additive track API partitions one globally validated direct-target join by
native GT identity and frame order. It sorts once and does not reconstruct or
revalidate the entire scene per person. Per-track temporary arrays scale with one
track's occurrences and joints; no dense identity-pair OKS matrix is allocated.

Joint error math reuses `compute_joint_position_errors`. The existing paired OKS
primitive now accepts an optional visibility mask, leaving its unmasked callers
unchanged. Unsupported rows are excluded before calling that primitive. Temporal
math reuses the existing central difference twice and trims three frames at either
end through the public `root_acceleration_magnitudes` primitive. Prediction-local
IDs never define temporal segmentation; native selected frame gaps do.

Every persisted track stores additive statistics, selected frame ranges and GT ID.
Null finalization allows unsupported tracks to remain present. Merging retains
micro-weighting and the ratio of acceleration sums. The legacy scene evaluator and
strict reducer retain their contracts; synthetic partition parity is the migration
guard. Disk IO, method loaders, catalogs and export formats remain consumer-owned.

Task design: `hjlib-experiments-results/docs/design/tasks/evaluator_store_cli/README.md`.

## Extension boundary

Extend this module when a new additive statistic belongs to the same fixed NAIVE
profile and can be computed after the existing direct-target join. A different
population, association rule or metric profile needs a sibling protocol module.
Any sibling must define its partition identity, temporal-gap rule, additive
sufficient statistics and nullable finalization before implementation, then add
partition-parity and merge-compatibility smoke coverage.
