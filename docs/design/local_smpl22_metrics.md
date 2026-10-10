# Local SMPL22 T/RT metrics

`local_smpl22_metrics.py` owns pure local joint-error leaves used by the
`local-smpl22-v1` evaluation profile. The caller supplies SMPL-ordered world
joint arrays in metres with equal positive frame counts. Each array may have
a different joint count greater than or equal to22. The leaves select joints
0..21 before finite checks or fitting, retaining pelvis0 and wrists20/21 and
excluding hands22/23. Every selected point must be finite; invalid points are
not dropped or repaired.

T-MPJPE subtracts each array's pelvis independently per frame, then calls the
existing Euclidean joint-error primitive. RT-MPJPE fits one proper rotation and
translation per frame through `hjlib_geometry` rigid-registration primitives.
It fits neither scale nor reflection. Fitting and scoring both use all22
selected points.

RT rejects rank<2 centered predicted points, centered reference points or their
cross-covariance. Rank counts singular values strictly greater than the maximum
of a relative threshold (`largest_singular_value * 1e-10`) and the relevant
absolute floor: `1e-12` metres for point sets, `1e-24` square metres for
cross-covariance. Planar support is valid; coincident or collinear support is
not. Explicit covariance checking also rejects pairings whose individual point
sets are nondegenerate but whose cross-covariance has rank1. T imposes no rank
condition.

Both leaves return immutable float64 `(N,22)` metre errors. They do not reduce,
convert units, filter frames, load GT, schedule inference or declare Output
compliance. The task consumer owns exact frame support and reports
`1000 * sum(values) / values.size` millimetres. Pelvis remains in the denominator.
The four complete VRv1 main views therefore contain `5100 * 22 = 112200`
joint occurrences per metric.

Existing SMPL24 standard metrics and SMPL22 T/PA body-validation contracts keep
their own definitions. This addition shares generic geometry/error primitives
without changing those contracts. The attended cross-repository design record
is [HMD-based VRv1 protocol](../../../hjlib-experiments-results/docs/design/tasks/hmd_vrv1_protocol/README.md).

Data-free smoke controls cover translation, independent per-frame rotation,
scale, a noncoplanar reflection, extra hand outliers/NaNs, unequal input joint
counts, selected nonfinite values, retained wrists/pelvis, immutable outputs,
planar support, rank1 covariance and relative/absolute rank tolerances.
