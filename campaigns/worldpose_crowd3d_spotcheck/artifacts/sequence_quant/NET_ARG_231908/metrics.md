# Sequence Monolith Quant

Scene: `NET_ARG_231908`
Metric: `SMPL_24_full`, unit: `mm`
Aggregation: frame-weighted person-frame mean over segment dumps.

## Own Dump Scope

| method | segments | person-frames | unique persons | frame span | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `abla_v1` | 14 | 17161 | 13 | `[0,1531), 1531 frames` | 384.65 | 53.10 |
| `v18agg` | 14 | 17161 | 13 | `[0,1531), 1531 frames` | 351.19 | 57.47 |

## Common Segment Scope

| method | segments | person-frames | unique persons | frame span | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `abla_v1` | 14 | 17161 | 13 | `[0,1531), 1531 frames` | 384.65 | 53.10 |
| `v18agg` | 14 | 17161 | 13 | `[0,1531), 1531 frames` | 351.19 | 57.47 |

Common segment filenames: `14`.

## Largest MPJPE Segments

### `abla_v1`

| segment | frames | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: |
| `NET_ARG_231908__0021_0000__p21__0000000_0001531.pkl` | 1531 | 664.92 | 48.54 |
| `NET_ARG_231908__0018_0000__p18__0000000_0001531.pkl` | 1531 | 648.36 | 53.15 |
| `NET_ARG_231908__0007_0000__p07__0000000_0000535.pkl` | 535 | 520.92 | 53.75 |
| `NET_ARG_231908__0009_0000__p09__0000000_0001230.pkl` | 1230 | 469.87 | 46.35 |
| `NET_ARG_231908__0007_0001__p07__0000925_0001531.pkl` | 606 | 454.21 | 59.25 |

### `v18agg`

| segment | frames | MPJPE mm | T-MPJPE mm |
| --- | ---: | ---: | ---: |
| `NET_ARG_231908__0011_0000__p11__0000000_0001153.pkl` | 1153 | 632.89 | 59.87 |
| `NET_ARG_231908__0016_0000__p16__0000000_0001531.pkl` | 1531 | 533.34 | 56.46 |
| `NET_ARG_231908__0021_0000__p21__0000000_0001531.pkl` | 1531 | 525.12 | 51.45 |
| `NET_ARG_231908__0002_0000__p02__0000000_0001531.pkl` | 1531 | 504.62 | 48.87 |
| `NET_ARG_231908__0012_0000__p12__0000000_0001235.pkl` | 1235 | 486.24 | 56.12 |

Full per-segment rows are in `metrics.json`.
