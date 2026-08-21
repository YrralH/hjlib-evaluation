# VirtualCrowd Ground Status and GeoCalib Handoff

## Review State

- Report date: 2026-08-19.
- Current task: complete; `baseline001` remains frozen and unchanged.
- Independent GeoCalib reproduction: complete in `hj-tpa-geocalib`.
- Next candidate: `GeoCalib normal + baseline001 D-search`; not activated.

## Frozen Current Result

`baseline001` is the immutable alias for
`conf_gt_5p0__ankle_lt_0p20__density_kde_scott_loo`:

- released VirtualCrowd, eight canonical scenes;
- existing GT-MOT RTMLib detections;
- shoulder midpoint to ankle midpoint observations;
- four-joint confidence strictly greater than `5.0`;
- ankle-distance / bbox-width strictly less than `0.20`;
- exact leave-one-out Gaussian KDE with Scott bandwidth and clipped,
  mean-normalized inverse-density weights;
- `H_prior=1.35 m`;
- 8,326 solver observations and the frozen 167,243-row evaluation support;
- person-frame-micro same-ray mean error `15.727719674759953 m`.

The frozen result resides at:

```text
/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-18/task/virtualcrowd-density-balanced-rcr-ground/result_kde_cartesian/conf_gt_5p0__ankle_lt_0p20__density_kde_scott_loo
```

Its limitations remain part of the result: the smallest scene has 256 solver
observations, while the combined same-ray improvement coexists with worse
support-weighted normal-oracle and distance-only means. The result therefore
contains systematic error cancellation and is not evidence that both plane
components improved.

The first-frame 4K normal-field visualization has two flat images per scene:
blue is `baseline001`, green is GT. The arrows point toward each normal's
camera-space vertical vanishing point. It is synchronized at:

```text
\\192.168.31.100\16Thj\Data_Process\sample_vis_sync_space\4090dv1\Code_as_Libs\virtualcrowd\baseline001-normal-field-4k
```

## Verified Crowd4D and GeoCalib Facts

GeoCalib is an independent ECCV 2024 project, not a Crowd4D-owned module. Its
official inference package accepts a single RGB image and returns camera and
gravity estimates. The official code is Apache-2.0 and the released weights
are CC BY 4.0:

- <https://github.com/cvg/GeoCalib>
- <https://arxiv.org/abs/2409.06704>

Crowd4D states that it uses GeoCalib to estimate gravity, then transforms its
camera trajectory and scene into a first-frame-referenced, gravity-aligned
world. Crowd4D additionally uses multi-keyframe scene geometry, metric SMPL
scale/contact, SIPC/SIS, HSIP, and temporal constraints. It does not use the
same `vertical-line normal -> fixed-height D search` decomposition as
`baseline001`.

- <https://arxiv.org/abs/2607.19517>
- <https://github.com/KHB1698/Crowd4D>

No direct GeoCalib intermediate is serialized in the supplied Crowd4D result.
The reviewed top-level fields are:

```text
betas, calculated_scale, cam_int, det_j2ds, extrinsic, ground_inlier_mask,
ground_plane, hsip_3ds, hsip_error_range, hsip_valid_mask, idxs,
scene_scale_factor, sipc, sipc_conf, sipc_filled_mask, thetas, track_flag,
trans, xscale_factor
```

In particular, there is no explicit `geocalib`, `gravity`, gravity confidence,
or uncertainty field. Crowd4D's ground producer code is not released, so its
exact GeoCalib revision, weights, image preprocessing, camera model, and use of
optional focal/gravity priors remain unknown.

## Recoverable Gravity Proxy

Because Crowd4D's scene frame is gravity aligned and its saved `extrinsic`
maps scene coordinates to camera coordinates, a post-alignment camera-frame
gravity direction can be recovered as

```text
n_gravity_proxy_camera = normalize(R_scene_to_camera @ [0, 1, 0]).
```

This is a **GeoCalib-aligned gravity proxy**, not raw GeoCalib output. It
contains no network confidence or uncertainty and may include Crowd4D's
downstream coordinate construction. The separately transformed
`ground_plane.normal` is Crowd4D's final reference-plane normal; it is not
identical to the gravity proxy.

First-frame sign-aligned angular errors against the GT camera ground normal
are:

| Scene | `baseline001` | Gravity proxy | Crowd4D plane | Proxy to Crowd4D plane |
| --- | ---: | ---: | ---: | ---: |
| `scene1` | 0.363 deg | 1.235 deg | 0.554 deg | 0.768 deg |
| `scene1_view2` | 2.223 deg | 0.483 deg | 0.449 deg | 0.118 deg |
| `scene2` | 1.245 deg | 1.024 deg | 0.681 deg | 0.343 deg |
| `scene2_view2` | 4.132 deg | 0.484 deg | 0.486 deg | 0.101 deg |
| `scene3` | 0.395 deg | 0.391 deg | 0.629 deg | 0.319 deg |
| `scene3_view2` | 1.918 deg | 0.620 deg | 0.644 deg | 0.034 deg |
| `scene4` | 2.938 deg | 2.012 deg | 2.038 deg | 0.558 deg |
| `scene4_view2` | 0.952 deg | 0.743 deg | 0.575 deg | 0.169 deg |
| **Mean** | **1.771 deg** | **0.874 deg** | **0.757 deg** | **0.301 deg** |

The gravity proxy is closer to GT than `baseline001` in seven of eight scenes.
It is not constrained to exact zero roll: its camera-frame x component reaches
approximately `0.008`. Across the 200 saved frames its maximum angular change
is below approximately `5e-6 deg`, consistent with VirtualCrowd's static
camera. The final Crowd4D plane differs from the proxy by `0.301 deg` on
average, supporting a small residual support-plane fit rather than a hard
gravity-normal assignment.

## Paper-Safe Interpretation

After an independent official GeoCalib run, it is valid to say that Crowd4D
and the proposed method both use the external off-the-shelf GeoCalib module for
gravity estimation. That claim requires our method to consume its own
GeoCalib result from original RGB, not the proxy or any Crowd4D artifact.

The following stronger claims are not currently supported:

- exact reproduction of Crowd4D's GeoCalib inference configuration;
- recovery of Crowd4D's raw GeoCalib output or confidence;
- proof that Crowd4D hard-codes VirtualCrowd roll to zero;
- equivalence between the gravity proxy and Crowd4D's final reference plane;
- evaluation of Crowd4D's full non-planar SIS/HSIP terrain through the current
  single-plane metric.

## Completed Independent GeoCalib Task

The independent residence is `hj-tpa-geocalib`, not `hj-tpa-crowd4d`. It:

1. froze one official GeoCalib revision, official pinhole weights, dependency
   environment, and original-image preprocessing;
2. ran official single-image inference on the first native frame of all eight
   VirtualCrowd scenes with no GT or Crowd4D input and no optional prior;
3. preserved raw gravity, camera/intrinsics, and official uncertainties
   that the official API actually returns;
4. retained the established camera-frame gravity convention and
   normal;
5. compared independent GeoCalib, recovered Crowd4D gravity proxy,
   `baseline001`, Crowd4D final plane, and GT;
6. left the later `GeoCalib normal + baseline001 D-search` arm unactivated.

The run did not use released VirtualCrowd intrinsics as a GeoCalib prior. A
separately named known-intrinsics ablation may be considered later.
The proxy is a read-only comparison target only and must never be an inference
input.

Two seed-0 runs were exactly equal array by array. Independent GeoCalib reached
`1.581703 deg` scene-macro normal error against GT, versus `0.873936 deg` for
the Crowd4D gravity proxy and `1.770651 deg` for `baseline001`. The full table,
source identities, runtime evidence, and limitations are recorded in
`hj-tpa-geocalib/docs/design/tasks/geocalib-image-inference/README.md`.

This completion does not mean Crowd4D inference parity or activation of a new
ground-effect headline result.

## Decision Before the Next Arm

The next action, if authorized, is to freeze a non-GT orientation rule for the
independent GeoCalib normal, keep the existing `H=1.35 m` D search unchanged,
and evaluate on the same 167,243 rows. It must be a separately named result.
