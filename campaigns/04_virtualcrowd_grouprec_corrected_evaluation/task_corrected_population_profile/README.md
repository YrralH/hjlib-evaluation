# Task: Corrected Population Profile

## Purpose And Boundary

Add the minimum stable evaluation surface needed to name and reduce
`C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9` while reusing all existing corrected
metric mathematics. Legacy `GT_VISIBLE` and `C4D_DYCROWD_COMMON` outputs remain
unchanged.

## State

- State: complete
- Completed: 2026-08-19
- Blocker: none.

## Residence

- Planned design:
  `docs/design/tasks/virtualcrowd-corrected-population-profile/README.md`.

## Result And Artifacts

- Real-data set probe: old common 167,243; filtered common 159,405; removed
  7,838; GroupRec-ge9 outside old common 6,255.
- Added the independent selected-view schema, JSON round trips, exact reducer,
  and the named `make_coco17_visible_ge9_common_mask` selection leaf.
- Legacy common evaluation parity is exact in the data-free smoke suite.

## Handoff

Completed; it unblocked the GroupRec producer and all three method reductions.
