# testing_protocol_from_monolith

驻地目标:重新解析 monolith `plans_and_notes/test_with_detected_kp/` 里的测试
filter 设计,给出当前可复用的筛选逻辑和筛选比例。本文只解释 test-set
population,不重新定义 metric reducer。

## Sources

Primary monolith notes:

- `dynamic_hvip/plans_and_notes/test_with_detected_kp/general_structure/test_filter_requirements_20260412.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/general_structure/good_visual_subset.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/general_structure/good_visual_subset_implementation_20260421.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/worldpose/wp_filter_plan.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/worldpose/wp_filter_final_decision_plan_20260420.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/jta_and_jta_ext/filter_plan.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/jta_and_jta_ext/status_20260501.md`
- `dynamic_hvip/plans_and_notes/test_with_detected_kp/jta_and_jta_ext/operations_log_20260503_20260504.md`

Implementation anchors:

- `dynamic_hvip/lib_dynamic_hvip/dataset/prod/multi_dynamics/divider/seq_modification.py`
- `dynamic_hvip/script/dataset/make_unifiy_dataset/filter_for_testing/filter_wp/debug043_wp_produce_seq_modifications.py`
- `dynamic_hvip/script/dataset/make_unifiy_dataset/filter_for_testing/filter_jta/debug070_jta_load_separate_filtering_results.py`
- `dynamic_hvip/script/dataset/make_unifiy_dataset/filter_for_testing/filter_jta/debug071_jta_produce_seq_modifications.py`

Local comparison anchors:

- `dynamic_hvip/plans_and_notes/test_our_method/RESULTS.md`
- `dynamic_hvip/plans_and_notes/eval_other_methods/COMPARISON.md`
- `dynamic_hvip/plans_and_notes/eval_unified/PAPER_CKPT_MAP.md`

Statistics below were recomputed on 2026-07-03 from
`/home/hj/Data_Process/*_filter_stats/seq_modifications_jsonbin/v1` and the
per-signal JTA/JTA_Ext npz files with these scratch scripts:

- `/tmp/codex-scratch-1001/2026-07-03/3c6ecc5b_testing-protocol-filter-stats.py`
- `/tmp/codex-scratch-1001/2026-07-03/c625d842_rebuttal-filter-table.py`
- `/tmp/codex-scratch-1001/2026-07-03/a40607dd_rebuttal-reproducible-criteria-stats.py`
- `/tmp/codex-scratch-1001/2026-07-03/b6880da5_jta-single-ground-union.py`
- `/tmp/codex-scratch-1001/2026-07-03/97192b67_jta-h2p5-rebuttal-table.py`

## Practical Conclusion

The original protocol is fair because it refuses to filter test frames by model
output or detector sparsity. It only removes data whose GT/geometry is
intrinsically unreliable before inference: persistent GT bias, severe
out-of-image / visibility / geometry invalidity, and segments too short for a
temporal protocol.

It is complicated because the fair decision is not a single threshold. WorldPose
and JTA/JTA_Ext have different GT failure modes, and JTA also has a separate
paper-report scene filter (`h2p5_p75`) that must not be confused with the
`seq_modifications/full` testset population currently consumed by
`hjlib-evaluation`.

For rebuttal framing, the important limitation to admit is the single-ground
polygon / ground-validity branch. It is not a cherry-pick by measured error,
but it is still a limitation of the method/protocol assumption: the method
expects a valid single-person crop with usable single-ground / camera geometry,
and some JTA frames violate that assumption. The honest claim is "we removed
protocol-invalid geometry cases and report the counts," not "the model handles
these cases."

## Rebuttal Logic Chain

1. **Predefined population, not performance-based selection.**
   The filter is defined over GT annotations, geometry, visibility, and
   high-confidence 2D references. It is applied before running a 3D method and
   does not look at our prediction error.
2. **GT-offset annotation reliability.**
   This is the only branch that uses a learned detector. The detector is only a
   high-confidence 2D reference to identify persistent annotation bias, not a
   criterion for "easy detections." Bias means a long, directionally consistent,
   significant reprojection offset from high-confidence 2D keypoints.
3. **Visibility / crop / subsegment validity.**
   If enough joints are out of image, raw-invisible, or too small in the crop,
   the frame does not provide a valid visual observation for the temporal crop.
   The protocol splits sequences on such bad frames and drops remaining
   subsegments shorter than 120 frames.
4. **Extreme-camera validity.**
   JTA near-camera projection blow-up is detected from GT geometry
   (`min_depth_m < 2.5`). This is a per-person-frame GT camera geometry
   condition, not a method score and not the scene-level `h2p5_p75` cut.
5. **Single-ground polygon / ground validity.**
   JTA frames with ankles in exclusion polygons, or outside the accepted
   single-ground band, violate the method's geometric operating assumption.
   The ground-band labels were manually inspected to decide whether a step /
   elevated surface should really be treated as a higher ground level. This is
   a limitation and should be stated as such.

Recommended rebuttal order:

1. GT-offset / annotation bias.
2. Raw visibility.
3. Out-of-image.
4. Small bbox.
5. Extreme-camera validity.
6. Single-ground polygon / ground validity.
7. Short subsegment.

Short subsegment must be last: it is not a native signal, but the consequence
of applying the BAD-frame cuts and then dropping kept runs shorter than 120
frames. Extreme-camera and single-ground polygon / ground validity are JTA-only;
the single-ground branch is the explicit method / protocol limitation. Report
per-step numbers as order-dependent "additional removed at this step" counts,
not as independent marginal totals.

Occlusion status, checked on 2026-07-03:

- JTA dim #3 ray-cast occlusion looks wired into the current monolith `full`
  producer: `debug070_jta_load_separate_filtering_results.py` loads
  `occlusion/<scene>.npz` and ORs its 10 limb-joint mask into JTA-22 slots;
  the `jta` and `jta_ext` `seq_modifications/full/_meta.json` files also list
  `occlusion ray-cast` under `merged_signals`.
- WP occlusion also looks wired into the current monolith `full` producer:
  `debug043_wp_produce_seq_modifications.py` feeds
  `occ_smpl24_seq | ooi_smpl24_seq` to `compute_seq_modification`.
- Because the current question is rebuttal reporting rather than a fresh
  contribution audit, do **not** count ray-cast occlusion or WP occlusion as
  separate rebuttal dimensions yet. Treat them as unresolved/internal
  visibility signals until we compute their standalone and sequential
  contributions.

Conservative high-level rows for rebuttal:

| dataset | active rows to report now | not reported as separate dim |
|---|---|---|
| WorldPose | GT-offset; out-of-image; short subsegment | occlusion unresolved/internal; extreme-camera N/A; single-ground polygon N/A; raw visibility N/A; small bbox N/A |
| JTA | GT-offset; raw visibility; out-of-image; small bbox; extreme-camera; single-ground polygon/ground validity; short subsegment | dim #3 ray-cast occlusion unresolved/internal |

## Reproducible Criteria List

This is the direct list to use in rebuttal text. It intentionally keeps WP and
the two relevant JTA population choices: without the paper scene filter
(`61` test scenes) and with `h2p5_p75` (`22` test scenes). JTA_Ext is omitted
from this compact table. A `0` means the criterion does not exist for that
dataset. Direct person-frame ratios use annotated `(person, frame)` cells as
denominator. Short is listed last because it is derived after all frame-level
cuts.

| criterion | exact reproducible rule | WP test ratio | JTA test ratio, no `h2p5_p75` | JTA test ratio, with `h2p5_p75` |
|---|---|---:|---:|---:|
| GT-offset / annotation bias | High-confidence RTMLib keypoints (`conf >= 5.0`) vs GT 2D, selected limb joints, offset normalized by bbox diagonal, significant magnitude `> 0.0262`, directionally consistent WP segment detector, suspicious segment length `>= 120` frames. Whole raw seq is dropped when a suspicious range overlaps it. | 116 / 360 raw seq (32.22%) | 354 / 2950 raw seq (12.00%) | 170 / 951 raw seq (17.88%) |
| Raw visibility | Raw dataset visibility flag only: `bundle.kp2d[..., 2] == 0.0` on native JTA-22 joints. For direct person-frame reporting, collapse by `mean_22(raw_vis_bad) > 0.50`; in the final full producer it is OR-ed per joint with other BAD signals. | 0 | 317073 / 1433757 person-frames (22.11%) | 73412 / 496747 person-frames (14.78%) |
| Out-of-image | A joint is OOI when projected/labelled `(x, y)` is outside image bounds. Person-frame BAD uses a strict `mean_j(is_ooi) > 0.50` threshold, so for 22-joint JTA this means at least 12 joints; for SMPL-24 WP this means at least 13 joints. | OOI-only policy: 52 / 360 no-kept seq (14.44%); kept frames 249734 / 253808 (98.39%) | Direct OOI: 52405 / 1433757 person-frames (3.66%); OOI-only policy: 8 / 2950 no-kept seq (0.27%) | Direct OOI: 41502 / 496747 person-frames (8.35%); OOI-only policy: 6 / 951 no-kept seq (0.63%) |
| Small bbox | Online dim #6 from raw 22-joint kp2d: `bbox_h_padded = (nanmax(y) - nanmin(y)) * (1 + 2 * 0.15)`; invalid when finite and `< 50 px`; broadcast to all 22 joints. | 0 | 263646 / 1433757 person-frames (18.39%) | 72891 / 496747 person-frames (14.67%) |
| Extreme camera | Dim #1 invalid mask: closest-joint camera-space depth `min_depth_m < 2.5`. This is per `(person, frame)`, not scene-level; the invalid bit is broadcast to all 22 joints. | 0 | 5905 / 1433757 person-frames (0.41%) | 385 / 496747 person-frames (0.08%) |
| Single-ground polygon / ground validity | Dim #5a: any on-screen ankle inside the manual exclude polygon. Dim #5b: manually inspected step / ground-height consistency; the lowest joint height relative to the given ground plane must stay in `[-0.1, 0.3]` m, implemented as category neither keep (`0`) nor NaN sentinel (`255`). Report the union `dim5a OR dim5b`; both are one single-ground protocol branch. | 0 | Union: 238300 / 1433757 person-frames (16.62%); components polygon 14.16%, ground 8.69%, overlap 6.23% | Union: 95574 / 496747 person-frames (19.24%); components polygon 13.54%, ground 12.37%, overlap 6.67% |
| Short subsegment | **Last step.** After all BAD-frame cuts, contiguous kept runs shorter than `N_FRAME_MIN_SEQ = 120` frames are dropped. This is `< 120`, not `<= 120`; it is not an independent raw signal. | Full-policy no-kept after non-bias cuts: 52 / 360 raw seq (14.44%) | Full-policy no-kept after non-bias cuts: 682 / 2950 raw seq (23.12%) | Full-policy no-kept after non-bias cuts: 101 / 951 raw seq (10.62%) |

The final `full` policy then applies the unified temporal rule to the OR-ed
per-joint BAD mask: BAD frame when `mean_j(bad_seq[t, j]) > 0.50`; kill BAD
runs of length `>= 20`; kill any 120-frame window with more than 60 BAD frames;
drop kept runs shorter than 120 frames.

Concrete GT-offset definition used by the final WP/JTA-port bias detector:

- Compare selected limb joints against RTMLib keypoints with high confidence
  (`conf >= 5.0`).
- Normalize offsets by bbox diagonal.
- Significant per-frame magnitude threshold: `0.0262`, calibrated from manual
  sampled labels (`A_P90` in the original WP calibration).
- Directional consistency: WP `Compare_Aggregated_V1` with a 45 degree cone
  and log-symmetric relative magnitude threshold `0.5`.
- Segment persistence: suspicious segment length `>= 120` frames. This is
  about 2.4 s for WorldPose at 50 FPS and 4.0 s for JTA at 30 FPS.

## Population Vocabulary

- **raw seq**: one raw `(scene, name_seq/person)` sequence before test filtering.
- **bias drop**: a seq-level whole drop. The whole raw seq is removed before
  frame-level segmentation.
- **short/no-kept drop**: the seq was not bias-dropped, but the BAD-frame logic
  leaves no kept run of length at least `N_FRAME_MIN_SEQ`.
- **kept segment**: one contiguous kept sub-range after BAD-frame cuts.
- **kept frames ratio**: `sum(kept segment lengths) / sum(raw seq lengths)`.

## Shared Final Filter

The current monolith implementation is the 2026-05-01 unified rewrite in
`seq_modification.py`. The earlier `visible` cleaner design was folded away from
the main evaluation policy.

For each raw seq:

1. If the policy applies bias drop and the per-seq bias detector marks the seq,
   set `dropped_whole=True`.
2. Otherwise aggregate dataset-specific per-frame BAD signals into
   `bad_seq` with shape `(L, K)`.
3. A frame is BAD when `mean_j(bad_seq[t, j]) > P_BAD_FRAME`.
4. Drop all frames satisfying any of:
   - D1: inside a consecutive BAD run of length `>= M_BAD_RUN`.
   - D2: covered by a `W_WINDOW` window with BAD count `> M_BAD_IN_WINDOW`.
   - D3: inside a remaining kept-mask run shorter than `N_FRAME_MIN_SEQ`.
5. Kept subsegments are contiguous runs not killed by D1/D2 and long enough for
   D3.

Locked `full` hyperparameters:

| parameter | value | meaning |
|---|---:|---|
| `P_BAD_FRAME` | 0.50 | more than half the joints are bad |
| `M_BAD_RUN` | 20 | a long consecutive BAD run kills itself |
| `W_WINDOW` | 120 | sliding-window length |
| `M_BAD_IN_WINDOW` | 60 | more than half a 120-frame window is too bad |
| `N_FRAME_MIN_SEQ` | 120 | minimum temporal segment length |

Policy variants:

| policy | purpose | bias drop | BAD signal | key knobs |
|---|---|---|---|---|
| `full` | primary quantitative evaluation | yes | dataset-specific full BAD mask | table above |
| `visualize` | qualitative superset | no | OOI-only | `M_BAD_RUN=1`, D2 disabled, min len 120 |
| `visualize_v3` | no-filter-ish qualitative policy | no | WP: OOI-only; JTA: raw visibility OR OOI | `P_BAD_FRAME=0.9999`, D2 disabled, min len 32 |

## WorldPose Logic

WorldPose uses fitted SMPL + multiview data, so its main data-quality concern is
systematic GT bias.

Bias whole-drop:

- Compare projected fitted SMPL joints to high-confidence RTMLib keypoints used
  as reference, not as an easiness filter.
- Joint set: COCO-17 without head, hips, and shoulders, leaving elbows, wrists,
  knees, and ankles.
- Error unit: bbox diagonal ratio.
- Locked config: `default_10pct_w50_bad16-8`.
  - single-frame magnitude threshold: `0.0262`;
  - window size: 50;
  - max BAD in window: 16;
  - max consecutive BAD: 8;
  - min suspicious segment length: 120.

Frame BAD mask:

- `full`: `occlusion OR out_of_image` over SMPL-24 body joints.
- `visualize`: out-of-image only.
- `visualize_v3`: out-of-image only, with no-filter-ish thresholds.

For rebuttal counts, keep WP occlusion as an internal/unresolved full-policy
signal unless a later contribution audit separates it from OOI/subsegment
effects.

## JTA / JTA_Ext Logic

JTA/JTA_Ext use raw 22-joint labels as the test-filter reference. Their failure
modes are different from WorldPose, so the full BAD mask is a native JTA-22
joint mask.

Bias whole-drop, dim #2:

- Same WP segment detector and same `default_10pct_w50_bad16-8` config.
- Compare raw kp2d against high-confidence RTMLib keypoints produced from raw
  22-joint bboxes.
- Whole-drop a raw seq when the bias-suspicious range overlaps it.

Full BAD mask:

| signal | granularity | current locked rule |
|---|---|---|
| dim #1 near-camera | per `(person, frame)` broadcast to 22 joints | `min_depth_m < 2.5` |
| raw visibility | per native joint | `bundle.kp2d[..., 2] == 0` |
| dim #3 ray-cast occlusion | 10 limb joints | OR into matching JTA-22 slots; unresolved/internal for rebuttal counts |
| dim #4 out-of-image | per native joint | joint outside `[0, W) x [0, H)` |
| dim #5a exclude mask | per `(person, frame)` broadcast | single-ground branch: any on-screen ankle inside exclude polygon |
| dim #5b ground band | per `(person, frame)` broadcast | single-ground branch: manually inspected step / ground-height consistency; category != keep and category != NaN; keep band is `[-0.1, 0.3]` m |
| dim #6 small bbox | per `(person, frame)` broadcast | padded bbox height `< 50 px` |

For `visualize`, JTA uses OOI-only. For `visualize_v3`, JTA uses
`raw_vis OR OOI` with no-filter-ish thresholds, so a frame is BAD only when all
22 joints are bad.

## Current Full-Policy Ratios

These are the current washed `seq_modifications_jsonbin/v1/full` ratios.

| dataset | split | scenes | raw seq | bias drop | short/no-kept drop | kept seq | kept segments | kept frames | raw frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WorldPose | all-store | 89 | 3397 | 1172 (34.50%) | 497 (14.63%) | 1728 (50.87%) | 1748 | 1181219 (49.07%) | 2407375 |
| WorldPose | test | 11 | 360 | 116 (32.22%) | 52 (14.44%) | 192 (53.33%) | 195 | 145754 (57.43%) | 253808 |
| JTA | all-store | 216 | 10178 | 1139 (11.19%) | 2339 (22.98%) | 6700 (65.83%) | 6914 | 1982463 (57.03%) | 3476450 |
| JTA | test | 61 | 2950 | 354 (12.00%) | 682 (23.12%) | 1914 (64.88%) | 1967 | 549593 (55.93%) | 982630 |
| JTA `h2p5_p75` | test subset | 22 | 951 | 170 (17.88%) | 101 (10.62%) | 680 (71.50%) | 695 | 226313 (59.38%) | 381104 |
| JTA_Ext | all-store | 33 | 773 | 85 (11.00%) | 184 (23.80%) | 504 (65.20%) | 529 | 279966 (67.41%) | 415338 |
| JTA_Ext | test | 6 | 180 | 23 (12.78%) | 48 (26.67%) | 109 (60.56%) | 122 | 78939 (65.26%) | 120960 |

For rebuttal, the shortest final-survival statement is:

| dataset | test raw seq | final kept seq | final kept segments | final kept frames | final remaining frame ratio |
|---|---:|---:|---:|---:|---:|
| WorldPose | 360 | 192 | 195 | 145754 / 253808 | 57.43% |
| JTA | 2950 | 1914 | 1967 | 549593 / 982630 | 55.93% |
| JTA `h2p5_p75` | 951 | 680 | 695 | 226313 / 381104 | 59.38% |
| JTA_Ext | 180 | 109 | 122 | 78939 / 120960 | 65.26% |

## Per-Criterion Rebuttal Counts

The final full policy OR-merges several frame-level signals before cutting
subsegments, so per-criterion counts do not add up to the final removal count.
Use these numbers as transparent evidence for each criterion, then report the
final survival table above.

### Out-of-image / subsegment counts

The cleanest count for the OOI/subsegment branch is the OOI-only `visualize`
policy: it bypasses bias and all other full-policy geometry signals, splits on
OOI-only bad frames, and applies the same minimum subsegment length.

| dataset | raw seq | OOI-only no-kept seq | OOI-only kept seq | OOI-only kept segments | OOI-only kept frames |
|---|---:|---:|---:|---:|---:|
| WorldPose test | 360 | 52 (14.44%) | 308 (85.56%) | 311 | 249734 / 253808 (98.39%) |
| JTA test | 2950 | 8 (0.27%) | 2942 (99.73%) | 2947 | 977075 / 982630 (99.43%) |
| JTA test `h2p5_p75` | 951 | 6 (0.63%) | 945 (99.37%) | 950 | 376506 / 381104 (98.79%) |
| JTA_Ext test | 180 | 1 (0.56%) | 179 (99.44%) | 179 | 120004 / 120960 (99.21%) |

For JTA direct per-frame signal rates, using annotated `(person, frame)` cells as
the denominator and the current `> 50%` joint-ratio collapse:

| dataset | OOI bad `(person, frame)` cells |
|---|---:|
| JTA test | 52405 / 1433757 (3.66%) |
| JTA test `h2p5_p75` | 41502 / 496747 (8.35%) |
| JTA_Ext test | 6636 / 132743 (5.00%) |

### Raw visibility counts

This branch is JTA-specific and uses the raw dataset visibility channel, not a
detector. A joint is bad when `bundle.kp2d[..., 2] == 0.0`; the person-frame
summary below uses the same strict `> 50%` joint-ratio collapse.

| dataset | raw-visibility bad `(person, frame)` cells | raw-visibility bad joint labels |
|---|---:|---:|
| JTA test | 317073 / 1433757 (22.11%) | 7502194 / 31542654 (23.78%) |
| JTA test `h2p5_p75` | 73412 / 496747 (14.78%) | 1889001 / 10928434 (17.29%) |
| JTA_Ext test | 33372 / 132743 (25.14%) | 804652 / 2920346 (27.55%) |

### Small-bbox counts

This branch is JTA-specific. It is computed online from raw 22-joint kp2d with
the locked dim #6 rule:
`bbox_h_padded = (nanmax(y) - nanmin(y)) * 1.30 < 50 px`.

| dataset | small-bbox invalid `(person, frame)` cells |
|---|---:|
| JTA test | 263646 / 1433757 (18.39%) |
| JTA test `h2p5_p75` | 72891 / 496747 (14.67%) |
| JTA_Ext test | 11979 / 132743 (9.02%) |

### Extreme-camera counts

This branch is JTA-specific. It removes near-camera / projection-unstable
person-frame cells before subsegment construction. The 0.41% JTA test ratio is
small because dim #1 is a per-person-frame near-camera condition
(`min_depth_m < 2.5`), not the scene-level paper filter. A scene can survive
or fail the paper camera filter independently of whether individual
person-frames trip dim #1.

| dataset | extreme-camera invalid `(person, frame)` cells |
|---|---:|
| JTA test | 5905 / 1433757 (0.41%) |
| JTA test `h2p5_p75` | 385 / 496747 (0.08%) |
| JTA_Ext test | 37 / 132743 (0.03%) |

### Single-ground polygon / ground validity counts

This is the limitation branch. It is geometry/protocol validity, not evidence
that the method solves those cases. For transparent reporting, keep the
component signals visible but report their union: the explicit exclusion-polygon
mask and the manually inspected ground-band / step-height consistency test are
both proxies for whether the frame satisfies the single-ground protocol
assumption.

| dataset | polygon/exclusion invalid cells | ground-band invalid cells | overlap | union to report |
|---|---:|---:|---:|---:|
| JTA test | 203061 / 1433757 (14.16%) | 124615 / 1433757 (8.69%) | 89376 / 1433757 (6.23%) | 238300 / 1433757 (16.62%) |
| JTA test `h2p5_p75` | 67266 / 496747 (13.54%) | 61424 / 496747 (12.37%) | 33116 / 496747 (6.67%) | 95574 / 496747 (19.24%) |
| JTA_Ext test | 3996 / 132743 (3.01%) | 4022 / 132743 (3.03%) | 3825 / 132743 (2.88%) | 4193 / 132743 (3.16%) |

### GT-offset counts

This is the only detector-assisted criterion. It removes whole raw seqs with
persistent significant GT reprojection bias.

| dataset | GT-offset whole-seq drops |
|---|---:|
| WorldPose test | 116 / 360 (32.22%) |
| JTA test | 354 / 2950 (12.00%) |
| JTA test `h2p5_p75` | 170 / 951 (17.88%) |
| JTA_Ext test | 23 / 180 (12.78%) |

## Qualitative-Policy Ratios On Test

These are not the primary quantitative testset. They explain why qualitative
views keep many more frames.

| dataset | policy | scenes | raw seq | bias drop | short/no-kept drop | kept seq | kept segments | kept frames |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WorldPose | `visualize` | 11 | 360 | 0 (0.00%) | 52 (14.44%) | 308 (85.56%) | 311 | 249734 (98.39%) |
| WorldPose | `visualize_v3` | 11 | 360 | 0 (0.00%) | 11 (3.06%) | 349 (96.94%) | 351 | 253563 (99.90%) |
| JTA | `visualize` | 61 | 2950 | 0 (0.00%) | 8 (0.27%) | 2942 (99.73%) | 2947 | 977075 (99.43%) |
| JTA `h2p5_p75` | `visualize` | 22 | 951 | 0 (0.00%) | 6 (0.63%) | 945 (99.37%) | 950 | 376506 (98.79%) |
| JTA | `visualize_v3` | 61 | 2950 | 0 (0.00%) | 0 (0.00%) | 2950 (100.00%) | 2950 | 982630 (100.00%) |
| JTA_Ext | `visualize` | 6 | 180 | 0 (0.00%) | 1 (0.56%) | 179 (99.44%) | 179 | 120004 (99.21%) |

## JTA `h2p5_p75` Paper Filter

This is a separate scene-level filter used by the monolith quantitative master
JSON for the paper/SOTA JTA table. It is not the same as the `full`
`seq_modifications` policy above.

Rule:

```text
keep scene iff height_m >= 2.5 AND pitch_deg <= 75
```

The source file
`dynamic_hvip/data/jta_camera_height_and_pitch/jta_kept_h2p5_p75.txt` says
`kept=107 / 216` over all JTA scenes. In the test split, monolith master JSON
records `n_scene=22` for `h2p5_p75`, versus `n_scene=61` for nofilter.

This is the source of the two JTA "big table" populations: the 22-scene
`h2p5_p75` table is a subset of the 61-scene nofilter table, but the split
basis is scene-level camera height/pitch, not the dim #1 extreme-camera mask.
Dim #1 is per `(person, frame)` and only accounts for 5905 / 1433757 JTA test
person-frames (0.41%).

For the same monolith v06 dump:

| JTA regime | scenes | metric `n_valid` | relative to nofilter `n_valid` | MPJPE | T-MPJPE |
|---|---:|---:|---:|---:|---:|
| `h2p5_p75` | 22 | 218174 | 42.35% | 420.9647 | 88.2709 |
| nofilter master | 61 | 515149 | 100.00% | 820.3503 | 85.7709 |
| current `hjlib-evaluation` full testset | 61 | 549593 kept frames | not the same metric mask | see `jta_protocol_parity_and_standup.md` | see same |

Implication: when reproducing the original paper number, apply the
`h2p5_p75` scene cut in addition to understanding the `full` filter population.
When testing current `hjlib-evaluation` `policy='full'`, do not call it the
paper filtered population.

## Required Comparison Metrics

Local comparison source: `dynamic_hvip/plans_and_notes/test_our_method/RESULTS.md`
and its referenced `eval_other_methods/COMPARISON.md`; exact paper-row ckpt
mapping is in `dynamic_hvip/plans_and_notes/eval_unified/PAPER_CKPT_MAP.md`.
Units below are meters, matching the paper table.

JTA with/without the paper scene filter:

| method key | no `h2p5_p75` T-MPJPE | no `h2p5_p75` MPJPE | with `h2p5_p75` T-MPJPE | with `h2p5_p75` MPJPE |
|---|---:|---:|---:|---:|
| `tram_pretrain` | 0.089 | 1.840 | 0.102 | 2.347 |
| `tram_rich_bd_h36m` | 0.096 | 1.847 | 0.109 | 2.249 |
| `gvhmr_pretrain` | 0.097 | 4.012 | 0.111 | 5.459 |
| `gvhmr_rich_bd_h36m` | 0.122 | 5.748 | 0.132 | 7.747 |
| `genmo` | 0.104 | 6.056 | 0.115 | 7.234 |
| `ours__ief_global_v18agg_strict_mask_lowcam_K1_lam9e-3_RF31_ep0004__strictoff` | 0.077 | 0.809 | 0.081 | 0.426 |

Paper Table 1 style (`h2p5_p75` JTA + WorldPose test):

| method | JTA T-MPJPE | JTA MPJPE | JTA T-MPJPE-SEQ | JTA RT-MPJPE-SEQ | WP T-MPJPE | WP MPJPE | WP T-MPJPE-SEQ | WP RT-MPJPE-SEQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAM | 0.102 | 2.25 | 0.974 | 0.958 | 0.064 | 4.25 | 1.61 | 1.57 |
| GENMO | 0.115 | 7.23 | 2.22 | 1.99 | 0.092 | 18.2 | 2.70 | 2.47 |
| V1 (HVIP baseline) | 0.086 | 2.33 | 2.51 | 2.40 | 0.058 | 0.656 | 0.701 | 0.717 |
| Ours | 0.081 | 0.426 | 0.339 | 0.331 | 0.061 | 0.374 | 0.157 | 0.152 |

JTA no-`h2p5_p75` sequence metrics for the comparable rows with local master
entries:

| method key | T-MPJPE | MPJPE | T-MPJPE-SEQ | RT-MPJPE-SEQ |
|---|---:|---:|---:|---:|
| `tram_pretrain` | 0.089 | 1.840 | 0.773 | 0.760 |
| `tram_rich_bd_h36m` | 0.096 | 1.847 | 0.709 | 0.697 |
| `gvhmr_pretrain` | 0.097 | 4.012 | 0.931 | 0.881 |
| `gvhmr_rich_bd_h36m` | 0.122 | 5.748 | 1.343 | 1.232 |
| `genmo` | 0.104 | 6.056 | 1.514 | 1.386 |
| `ours__ief_global_v18agg_strict_mask_lowcam_K1_lam9e-3_RF31_ep0004__strictoff` | 0.077 | 0.809 | 0.485 | 0.464 |

Note: the current manuscript TRAM Table 1 JTA T-MPJPE cell is `0.102`.
`PAPER_CKPT_MAP.md` records that if Table 1 is forced to one coherent
`tram_rich_bd_h36m` row, only that cell becomes `0.109`; the ranking and all
Ours values above are unchanged.

## Historical Notes And Drift

- The 2026-04-21 WP implementation note records an older `full` result:
  `520 / 3397 = 15.3%` bias-dropped and 1813074 kept frames. That was before
  the 2026-04-23 WP bias config change to `default_10pct_w50_bad16-8`; the
  current washed store is `1172 / 3397 = 34.5%` bias-dropped.
- The old "visible subset" cleaner (head/tail iterative trim + global bad-ratio
  drop) is useful for understanding the design intent, but it is not the final
  primary evaluation path after the 2026-05-01 unified rewrite.
- `hjlib-evaluation` ports the consumer side. It reads the washed
  `seq_modifications_jsonbin/v1` store; it does not port the monolith filter
  production scripts.
