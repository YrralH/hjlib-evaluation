# Crowd3D 2D Matching

Algorithm: `scipy.optimize.linear_sum_assignment`.
Cost: robust bidirectional nearest-joint 2D distance plus center distance.

| frames | GT | Crowd3D | accepted | mean cost px | median cost px |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 210 | 374 | 195 | 4.36 | 2.90 |

| frame | GT | Crowd3D | accepted | rejected | unmatched GT | unmatched Crowd |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `NET_ARG_231908_frame0` | 21 | 39 | 20 | 1 | `1` | 19 |
| `NET_ARG_231908_frame1` | 21 | 37 | 20 | 1 | `3` | 17 |
| `NET_ARG_231908_frame2` | 21 | 33 | 20 | 1 | `18` | 13 |
| `NET_ARG_231908_frame3` | 21 | 38 | 19 | 2 | `1,3` | 19 |
| `NET_ARG_231908_frame4` | 21 | 38 | 19 | 2 | `1,3` | 19 |
| `NET_ARG_231908_frame5` | 21 | 36 | 20 | 1 | `3` | 16 |
| `NET_ARG_231908_frame6` | 21 | 40 | 19 | 2 | `1,18` | 21 |
| `NET_ARG_231908_frame7` | 21 | 38 | 20 | 1 | `1` | 18 |
| `NET_ARG_231908_frame8` | 21 | 38 | 19 | 2 | `1,3` | 19 |
| `NET_ARG_231908_frame9` | 21 | 37 | 19 | 2 | `1,3` | 18 |

Review overlays in `overlays/` and `contact_sheet.jpg` before using these matches for metrics.
