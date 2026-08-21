# Task: GroupRec Corrected Producer

## Purpose And Boundary

Implement a GroupRec-owned scene producer that composes finalized result
loading, persistent native interpretation, a VirtualCrowd GT side path, and the
stable corrected-evaluation input. It does not own metrics or runner startup.

## State

- State: complete
- Completed: 2026-08-19
- Blocker: none.

## Residence

- Planned design:
  `../../../hj-tpa-grouprec/docs/design/tasks/grouprec-corrected-evaluation-producer/README.md`.

## Handoff

The producer supplied one corrected scene input/summary per finalized GroupRec
scene to the completed full-evaluation task. Its evaluation interpreter uses
the persistent `SMPL_Full.forward(...)[j3d]` path, which matched an independent
`SMPL_Full` call exactly; it does not regress already-translated vertices.

## Result And Artifacts

- Data-free GroupRec smoke: 48 passed.
- One-scene pilot: 10,196 selected/matched occurrences and 9,941 triples.
- Full GroupRec result: 159,405 selected/matched occurrences and 156,263
  triples; zero failed method-scene workers.
- Parity receipt:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-19/task/grouprec-virtualcrowd-corrected-eval/evaluation-parity-receipt.json`.
