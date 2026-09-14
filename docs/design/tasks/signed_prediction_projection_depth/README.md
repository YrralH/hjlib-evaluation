# Signed prediction projection depth

## Requirements

- `Corrected_Crowd_Sequence` records normalized prediction facts. Every float
  remains finite, but prediction camera depth may be zero or negative when a
  method trajectory reaches or crosses its camera plane.
- This result-data contract must not classify that finite method output as a
  corrupt scene.
- A subsequent user decision permits removing the very small non-positive-depth
  subset from OKS support while preserving the accepted publication, all 3D
  populations, and every other valid OKS joint/person-frame.
- `hjlib-experiments-results` records the observed Genmo WorldPose case and the
  adopted support-exclusion treatment in a short note.

## Mathematical Architecture

For one identity-paired person-frame with native GT visibility `v[j]` and
prediction camera depth `z[j]`, define:

```text
oks_joint_valid[j] = (v[j] > 0) and (z[j] > 0)
```

OKS is the existing mean over `oks_joint_valid`. A row with no valid joint adds
neither an OKS value nor one unit to `oks_vis_count`. This removes only invalid
2D projection support: completeness, selected/matched people, and every non-OKS
metric sum/count remain unchanged, including all frame/sequence alignment,
layout, depth-order and dynamics metrics. Cross-scene reduction continues to
use the retained OKS sum and its independent support count.

Existing empty-support behavior remains path-specific: corrected-crowd reduction
returns `None`; strict NAIVE cross-scene reduction raises when total OKS support
is zero; track finalization returns `None`.

## Code Architecture

- Remove only the positive-depth relation check from
  `Corrected_Crowd_Sequence.validate_values_and_keys()`. Existing normalization
  already rejects non-finite floats and continues to validate shapes, keys,
  associations, visibility and bboxes.
- Add `make_positive_depth_joint_mask(reference_joint_valid,
  target_camera_depth)` in `keypoint_oks.py`. It accepts exact same-shape bool
  and real numeric `(N,J)` arrays, forbids broadcasting/non-finite depth, and
  returns bool `(N,J)` without knowing native visibility encoding.
- Apply it only in `add_frame_layout_and_oks_metrics()`,
  `compute_virtualcrowd_oks_vis_statistics()` and `track_oks_statistics()`.
  Each derives `row_supported = mask.any(axis=1)`, then synchronously filters
  GT rows, prediction rows, areas and the joint mask with that same `(N,)`
  selector before calling `compute_paired_keypoint_oks()`. Direct identities
  therefore remain aligned and no pairwise matrix/diagonal is involved.
  A frame with no supported row is skipped; a track call with no supported row
  returns `(0.0, 0)`, so the nonempty-only paired primitive is never called with
  `N == 0`.
- Add a regression test that constructs a sequence containing zero and an
  observed-class negative included prediction depth and verifies that both
  values survive immutable normalization.
- Update the stable evaluation design/usage boundary and the Genmo result note;
  do not change metric formulas, summary schemas or public names.

## Smoke-Test Standard

- Finite zero and negative included prediction depths are accepted by
  `Corrected_Crowd_Sequence` and retained exactly at the container boundary.
- All three direct-identity OKS paths exclude the same non-positive-depth joints;
  rows with partial support remain and rows with zero support disappear only
  from OKS sum/count.
- Caller tests place a zero-support row between supported identities and include
  a partial-support row whose excluded joint has a large XY error, catching
  diagonal shifts and accidental whole-row removal.
- Tests compare every non-OKS sum/count plus completeness, selected/matched and
  acceleration support before/after this OKS-only exclusion. They freeze the
  corrected/strict-NAIVE/track empty-global-support outcomes separately.
- Targeted corrected-crowd, NAIVE comparison and track-statistics smoke tests
  pass; strict pyright remains clean.

## Modification History

- 2026-09-11: Requirements and Code Architecture drafted from the observed
  Genmo GVHMR WorldPose camera-plane crossing case.
- 2026-09-11: Code Architecture review found the legacy corrected-crowd OKS
  consumer had relied on the record-level assertion. Accepted: preserve its
  policy by moving the assertion into that OKS path. Accepted: use a reachable
  negative-depth regression rather than claiming an end-to-end zero-depth case.
- 2026-09-11: Whole-repo contract review found one historical generic-gate
  statement, the NAIVE track usage caveat, and an untested record-level zero
  boundary. Accepted all three; zero is tested only as a container boundary.
- 2026-09-11: Whole-repo behavior/owner review found the observed incident was
  recorded in the GVHMR document although the evaluated adapter is Genmo.
  Accepted: moved the short non-normative note to `genmo_results.md`.
- 2026-09-11: The user permitted removing the very small damaged projection
  subset while retaining the rest, selecting positive-depth OKS support
  filtering and GPU-6 continuation.
- 2026-09-11: Mathematical architecture/Code Architecture reviews accepted the joint mask but
  required freezing three distinct empty-support outcomes, all non-OKS
  invariants, a generic exact-shape helper, synchronous identity-row filtering,
  and paired rather than matrix OKS. All findings accepted into the design.
- 2026-09-11: Code Architecture re-review required explicit zero-supported
  short-circuits before the nonempty-only paired primitive. Accepted; no other
  findings remain.
- 2026-09-11: User authorized continuing on physical GPU 6 while removing only
  the very small damaged subset and retaining the majority. Requirements and
  Mathematical/Code Architecture updated to exclude non-positive-depth joints
  only from OKS support; new layer reviews pending.
