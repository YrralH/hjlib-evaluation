# Frame0/Frame1 Quantitative Reference

Frames: `NET_ARG_231908_frame0`, `NET_ARG_231908_frame1`.
Metrics are SMPL-24 MPJPE / T-MPJPE in millimeters.

## Monolith Overlay Scope

| method | person-frames | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: |
| `abla_v1` | 26 | 316.86 | 59.98 |
| `v18agg` | 26 | 361.01 | 63.65 |

## Crowd3D Matched Scope

| method | matched person-frames | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: |
| `crowd3d` | 40 | 572.75 | 99.24 |

Matching: accepted `40` of GT `42`; Crowd3D detections `76`.

## Common All-Three Subset

| method | person-frames | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: |
| `crowd3d` | 26 | 751.91 | 98.47 |
| `abla_v1` | 26 | 316.86 | 59.98 |
| `v18agg` | 26 | 361.01 | 63.65 |

Common GT person/frame keys:

`NET_ARG_231908_frame0:0, NET_ARG_231908_frame1:0, NET_ARG_231908_frame0:2, NET_ARG_231908_frame1:2, NET_ARG_231908_frame0:7, NET_ARG_231908_frame1:7, NET_ARG_231908_frame0:9, NET_ARG_231908_frame1:9, NET_ARG_231908_frame0:11, NET_ARG_231908_frame1:11, NET_ARG_231908_frame0:12, NET_ARG_231908_frame1:12, NET_ARG_231908_frame0:13, NET_ARG_231908_frame1:13, NET_ARG_231908_frame0:14, NET_ARG_231908_frame1:14, NET_ARG_231908_frame0:15, NET_ARG_231908_frame1:15, NET_ARG_231908_frame0:16, NET_ARG_231908_frame1:16, NET_ARG_231908_frame0:18, NET_ARG_231908_frame1:18, NET_ARG_231908_frame0:20, NET_ARG_231908_frame1:20, NET_ARG_231908_frame0:21, NET_ARG_231908_frame1:21`
