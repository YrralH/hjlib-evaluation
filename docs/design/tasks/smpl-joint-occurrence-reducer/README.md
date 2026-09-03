# SMPL Joint Occurrence Reducer Task Design

## Requirements

Add one evaluation-owned, stateless public reducer for sparse person-frame
occurrences. It serves methods whose protocol population is not representable
as the existing dense `TestSet` segment dump without inventing missing rows.
It owns MPJPE/T-MPJPE reduction only; dataset/manifest identity, checkpoint
selection, model adaptation, file IO, and training remain with callers.

The initial consumer is CoMHR/GroupRec JTA fitted-SMPL training. The API must
reuse `Metric_Spec_3D` and `JTA_EVAL_META`, accept canonical SMPL-24 output from
the pinned method as well as full SMPL-54 output, and return structured values
rather than print a table.

## Mathematical Architecture

The public function is:

```text
reduce_smpl_joint_occurrences(
    pred_joints, gt_joints,
    pred_occurrence_ids, gt_occurrence_ids,
    metric_specs,
    unit_world,
    joint_layout,
    pred_coordinate_frame, gt_coordinate_frame,
) -> tuple[SMPL_Joint_Occurrence_Result, ...]
```

`pred_joints` and `gt_joints` have equal shape `[N,J,3]`, where every row is one
person-frame occurrence. Separate `int64[N]` prediction and GT occurrence IDs
must be unique and exactly equal row-by-row; the reducer rejects reordered,
duplicated, or mismatched pairs. Applying one common permutation to joints and
IDs on both sides preserves the result, while permuting only one side fails.
`pred_coordinate_frame` and `gt_coordinate_frame` are each explicitly
`'camera'` or `'world'` and must match; the reducer cannot infer axes from
numbers, so the caller remains responsible for the declared Euclidean frame and
axis convention. `joint_layout='smpl_24'` requires
`J=24`; `joint_layout='smpl_all_54'` requires `J=54`. SMPL-24 is the exact
prefix of `SMPL_ALL_54`, so the existing `Metric_Spec_3D` indices are valid when
every subject/root index is a non-bool integer in `[0,J)`; otherwise the reducer
rejects the call. Subject and root index tuples must each be nonempty and
unique, so accidental negative indexing or duplicate weighting cannot pass.
`N` must be positive. The reducer accepts float32 or float64 and requires finite
values at all selected subject/root indices.

For each metric, MPJPE is the arithmetic mean of Euclidean error over all
`N * len(joint_indices)` occurrence-joint cells. T-MPJPE independently subtracts
the mean of the declared root joints from prediction and GT for each occurrence,
then takes the same arithmetic mean. Inputs use `unit_world` (`'m'` or `'mm'`);
reported values are millimetres. No scene, frame, person, or group receives an
extra reduction weight.

Each immutable result contains `metric_name`, `num_occurrence`, `num_joint`,
`mpjpe_mm`, and `t_mpjpe_mm`. A common input permutation cannot affect the
result except for ordinary floating-point summation tolerance. Jitter is
intentionally absent: sparse occurrence rows do not assert temporal continuity.

## Code Architecture

The implementation lives in
`src/hjlib_evaluation/smpl_joint_occurrence_reducer.py` and is re-exported from
`hjlib_evaluation.__init__`. It depends only on NumPy plus existing
`Metric_Spec_3D`; it does not depend on a TPA, dataset assembly, a GT provider,
or filesystem schema.

`SMPL_Joint_Layout` is a literal public type, and
`SMPL_Joint_Occurrence_Result` is a frozen dataclass. Validation and reduction
are ordinary functions with no cache or global state. A public stateless
`compute_smpl_joint_occurrence_metric` kernel owns one metric's MPJPE/T-MPJPE
math; the tuple reducer calls it for every spec, and the existing dense
`eval_reducer.py` calls the same kernel while retaining its segment IO,
frame-weighting, jitter, and table behavior. This direction avoids a private
cross-module test dependency, keeps one math owner, lets both TPAs call
evaluation, and prevents evaluation from importing either method.

## Smoke-Test Standard

Pure tests cover zero error, a known absolute translation error, independent
root translation removal, both JTA metric variants, metre/millimetre scaling,
occurrence-weighted rather than scene/group-weighted means, input-order
permutation tolerance, both 24/54 layouts, and rejection of empty, mismatched,
nonfinite, wrong-dtype, wrong-layout, and out-of-range metric-index inputs.
Separate gates reject prediction/GT occurrence-ID mismatch, one-sided
permutation, duplicate IDs, coordinate-frame mismatch/invalid values, and
invalid `unit_world`. Empty, negative, bool, and duplicate subject/root indices
each have rejection cases.

The shared public kernel has fixed-value regression cases matching the existing
dense formula. Existing dense reducer smoke tests must remain green after it is
routed through that kernel; no test imports an underscore-prefixed function.
Public import and strict pyright are mandatory.

## Migration Plan

This is an additive public surface. No dense dump or existing result format is
migrated. Stable design/usage documentation is updated separately after code
completion.

## Modification History

- 2026-08-29: residence established and Requirements, Mathematical
  Architecture, Code Architecture, and Smoke-Test Standard frozen before code.
- 2026-08-29: closed first Code Architecture review by requiring strict index
  validation and extracting a public stateless metric kernel reused by both
  sparse and dense reducers instead of testing a private function.
- 2026-08-29: closed first Mathematical Architecture review by adding exact
  occurrence-ID pairing, explicit matching coordinate-frame declarations,
  nonempty index validation, and an invalid-unit rejection gate.
