# Crowd3D / WorldPose Spotcheck

Temporary campaign home for a small qualitative comparison between Crowd3D outputs and
selected monolith/HJ baselines on original multi-person WorldPose frames.

This is a rebuttal-only frozen-output spot check. It is not the default AAAI
experiment path and should not be treated as a reusable monolith compatibility
layer.

## Scope

- Current monolith anchor: `ours__ablation_a00_hvip_hipmid_K1_ep0004`.
- Current comparison pair: `abla_v1` and `v18agg`.
- First step: select a few original WorldPose frames that contain multiple people and are
  covered by the existing a00 dump.
- Keep throwaway scripts under `scripts/`.
- Keep generated selections and images under `artifacts/`.

This directory is intentionally campaign-local. If the experiment becomes reusable, move
stable readers/visualizers to their owning hjlib package later.

## Local Environment Contract

The commands below are local reproduction commands for this frozen campaign, not
the public `hjlib-evaluation` package contract. The scripts intentionally stay
outside `src/`, `test_smoke/`, and `test/`; they are not covered by the repo
`pyrightconfig.json`, and their extra imports are not added to
`pyproject.toml`.

Assumptions:

- run in the `hjlib_py312` environment on the Code_as_Libs workstation;
- sibling hjlib repos exist at the local `Code_as_Libs/` layout injected by the
  scripts;
- `opencv-python`, `scipy`, and the relevant hjlib sibling packages are already
  installed in that environment;
- data roots listed below exist on this machine.

If this campaign becomes reusable, first move stable code into the owning hjlib
package, declare its dependencies/pins, and add it to the normal type/test
surface.

## Known Inputs

- a00 dump:
  `/home/hj/Data_Process/protocol_dynamic/inference_dumps/worldpose/full/kp_rtmlib/ablation_a00_hvip_hipmid_K1_ep0004`
- v18agg dump:
  `/home/hj/Data_Process/protocol_dynamic/inference_dumps/worldpose/full/kp_rtmlib/ief_global_v18agg_strict_mask_lowcam_K1_lam9e-3_RF31_ep0004__strictoff`
- a00 existing per-person vis:
  `/home/hj/Data_Process/protocol_dynamic/vis/worldpose/full/kp_rtmlib/ablation_a00_hvip_hipmid_K1_ep0004/per_seq`
- Crowd3D external downstream package:
  `/home/hj/Data_Process/protocol_dynamic/external_results/worldpose/crowd3d/NET_ARG_231908_downstream`
- Crowd3D remote source:
  `G1M-hj:/mnt/ssd_2T_1100_0/zl/crowd3d_NET_ARG_231908_downstream`

## First Command

Run from `hjlib-evaluation` repo root:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/select_worldpose_frames.py
```

Outputs:

- `artifacts/selected_worldpose_frames.json`
- `artifacts/selected_worldpose_frames.md`

Add `--render --render-mode pred2d` to render a00 `joints_54_world` through
`hjlib-camera`. This writes the same three selected frames into:

- `artifacts/overlays/abla_v1/`
- `artifacts/overlays/v18agg/`

Projection ranges are written to `artifacts/projection_stats_abla_v1.json` and
`artifacts/projection_stats_v18agg.json`.

Use `--render-mode gt` for the heavier GT keypoint/bbox overlay path. Mesh overlay
via `hjlib-pyrender` is a later step; the a00 dump contains SMPL rotmats/transl, so
that path is available once we wire the params into `hjlib-smpl`.

## Metric Command

Run from `hjlib-evaluation` repo root:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/compute_spotcheck_metrics.py
```

This computes WorldPose `SMPL_24_full` MPJPE / T-MPJPE for the selected frames and
both methods. Outputs:

- `artifacts/spotcheck_metrics.json`
- `artifacts/spotcheck_metrics.md`

Use `--max-frames 2` to reproduce the first-two-frame subset:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/compute_spotcheck_metrics.py \
    --max-frames 2 \
    --out-json campaigns/worldpose_crowd3d_spotcheck/artifacts/spotcheck_metrics_first2.json \
    --out-md campaigns/worldpose_crowd3d_spotcheck/artifacts/spotcheck_metrics_first2.md
```

## Crowd3D Visual Quickview

Run from `hjlib-evaluation` repo root:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/make_crowd3d_visual_quickview.py
```

This reads the pulled Crowd3D package manifest and creates input / Crowd3D render /
Crowd3D depth side-by-side panels without matching people to GT or monolith
predictions. Outputs:

- `artifacts/crowd3d_visual/contact_sheet.jpg`
- `artifacts/crowd3d_visual/side_by_side/`
- `artifacts/crowd3d_visual/summary.json`

Crowd3D result counts can differ from GT: the current package reports 33-43
merged persons per frame over 11 frames, so matching should be a separate step.

## Crowd3D 2D Matching

Run from `hjlib-evaluation` repo root:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/match_crowd3d_to_gt_2d.py
```

This matches Crowd3D merged scene-level people to WorldPose GT people for
`NET_ARG_231908_frame0` through `NET_ARG_231908_frame9`. The assignment step is
`scipy.optimize.linear_sum_assignment`, i.e. the linear-sum assignment problem /
Hungarian-style bipartite matching. The cost is a robust 2D-joint distance:
bidirectional nearest-joint distance between Crowd3D `pj2d_71_scene_px` and GT
projected 2D joints, plus a small center-distance term. Pairs above
`--max-cost-px` are rejected after assignment so extra Crowd3D people remain
unmatched.

Outputs:

- `artifacts/crowd3d_matching/matches.json`
- `artifacts/crowd3d_matching/matches.csv`
- `artifacts/crowd3d_matching/matches.md`
- `artifacts/crowd3d_matching/overlays/`
- `artifacts/crowd3d_matching/contact_sheet.jpg`

## Three-Model Frame0/Frame1 Visual

Run from `hjlib-evaluation` repo root:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/render_three_model_frame0_frame1.py
```

This clears `artifacts/overlays/` and regenerates exactly three method folders for
`NET_ARG_231908_frame0` and `NET_ARG_231908_frame1`:

- `artifacts/overlays/crowd3d/`
- `artifacts/overlays/abla_v1/`
- `artifacts/overlays/v18agg/`

Side-by-side review panels are written outside `overlays/`:

- `artifacts/three_model_visual/side_by_side/`
- `artifacts/three_model_visual/contact_sheet.jpg`

## Frame0/Frame1 Quantitative Reference

Run from `hjlib-evaluation` repo root:

```bash
python campaigns/worldpose_crowd3d_spotcheck/scripts/quant_frame0_frame1_three_models.py
```

This reports:

- `abla_v1` / `v18agg` metrics on the currently rendered overlay person set.
- Crowd3D metrics on the 2D-Hungarian matched GT person set.
- All three methods restricted to the common GT person/frame subset.

Outputs:

- `artifacts/frame01_quant/metrics.json`
- `artifacts/frame01_quant/metrics.md`
- `artifacts/frame01_quant/crowd3d_matching/`
