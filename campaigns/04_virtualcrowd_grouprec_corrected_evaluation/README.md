# Campaign 04 - VirtualCrowd GroupRec Corrected Evaluation

## Status

- State: complete
- Owner: `hjlib-evaluation`
- GroupRec native-integration owner: `hj-tpa-grouprec`
- Completion date: 2026-08-19
- Result: all three methods completed eight-scene reduction on the exact
  159,405-occurrence selected population.

## Goal And Boundary

Compare Crowd4D, DyCrowd, and GroupRec with the existing corrected VirtualCrowd
metric mathematics on one honestly named population:
`C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9`. It is the frozen 167,243-key
Crowd4D/DyCrowd common set filtered by at least 9 of the mapped COCO-17 source
channels being greater than zero; author-relayed `0.5` values count. The
observed exact count is 159,405 occurrences.

The campaign reuses accepted native outputs. It does not rerun inference,
decode images for evaluation, change the 15 corrected metrics, mutate the old
167,243-key manifest/results, introduce a population registry, or make
`hjlib-evaluation` understand GroupRec-private schemas or start its runner.

## Completion Criteria

1. The evaluation contract can name and reduce the 159,405-key population
   without changing legacy schema-v1 result semantics.
2. A GroupRec-owned producer loads finalized scenes, persistently interprets
   compact SMPL parameters, joins GT by explicit source identity, and produces
   the nearest accepted corrected-evaluation input without image decode.
3. Executable parity evidence closes SMPL joint order, translation, projection,
   scale, unit, and camera-coordinate semantics used by the evaluator.
4. Crowd4D, DyCrowd, and GroupRec all complete eight-scene reductions on the
   same population and publish source-bound machine-readable results.
5. A three-method table, support/completeness statement, and full receipt
   reconcile the exact population and every method-scene worker.

## Task Index

| Task | State | Purpose |
| --- | --- | --- |
| [Corrected population profile](task_corrected_population_profile/) | complete | Express the filtered common population honestly while reusing the existing metric implementation |
| [GroupRec corrected producer](task_grouprec_corrected_producer/) | complete | Compose finalized result loading, persistent interpretation, VirtualCrowd GT, and the stable evaluation input |
| [Three-method full evaluation](task_three_method_full_evaluation/) | complete | Re-reduce existing native outputs and publish the common three-method result |
| [Corrected world dynamics](task_world_dynamics_metrics/) | complete | Add four world-space temporal residuals and recompute GroupRec with existing metrics |

## Decisions

- `hjlib-evaluation` owns the stable schema, population meaning, metrics, and
  reduction. `hj-tpa-grouprec` owns request/result interpretation and the
  one-way producer adapter.
- The existing scene-level `Corrected_Crowd_Sequence` and per-scene summary
  reduction remain the efficient loader/evaluator architecture. No second
  generic loader is introduced.
- The GroupRec producer wraps `load_collected_scene`, a generalized persistent
  native-result interpreter, and a scene-level VirtualCrowd GT side path. It
  joins by `(scene_id, frame_id, source_track_id)`, never positional rows.
- Evaluation must not use `Frame_Accessor.get_frame`, because numerical
  evaluation does not need images. Dataset metadata/labels are loaded once per
  active scene and released after summary publication.
- The 159,405 population is a default comparison selection for this campaign,
  not a registry entry or an automatic rejection/safety policy.
- Crowd4D and DyCrowd inference are not rerun. Their accepted native outputs are
  re-normalized/reduced because old aggregate summaries cannot be losslessly
  filtered from 167,243 to 159,405.

## Inputs And Provenance

- Existing common manifest SHA-256:
  `1126680d81a505522162397a2ed169ba0cf9625175c8ca03986df2e24dacda74`.
- Accepted GroupRec full-run receipt SHA-256:
  `dcffbbaca5250ac52757f756dc8507647ec5649a05032bcddca54903feac5816`.
- Accepted GroupRec result residence:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-18/task/grouprec-virtualcrowd-full-output/results`.
- Pre-implementation source heads: `hjlib-evaluation`
  `1606d60319ab9de582057f2dc9c2eba94ad5f590` and `hj-tpa-grouprec`
  `a640d8fd74871d6accf5d1a86fe4de957ba11363`.
- Both worktrees contained unrelated in-progress changes at activation. This
  campaign preserves them and records its own delta without resetting or
  claiming those changes.

## Unattended Interval

The user explicitly authorized unattended Sequential Task Level 3 execution on
2026-08-19 through completed evaluation. Ordinary implementation, dependency,
performance, test, and run failures are in scope for autonomous closure. Stop
only for a material protocol/coordinate/dataset identity conflict, missing
authority, or a condition that prevents a credible result. The interval ends
when the completion criteria are met or such a blocker is recorded.

## Headline Status Consumer

- [`../README.md`](../README.md)

## Completed Result

- All method-selected and matched counts: 159,405 each.
- All method acceleration triple counts: 156,263 each.
- All method-scene workers: 24/24 successful.
- Three-method receipt:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-19/task/grouprec-virtualcrowd-corrected-eval/three-method-evaluation/receipt.json`.
- Comparison JSON:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-19/task/grouprec-virtualcrowd-corrected-eval/three-method-evaluation/comparison.json`.
- Comparison SHA-256:
  `d07cd03b92911e06f29dddcbc25c96fc9e27a0580eb771b84b5178d482eb805a`.
- GroupRec operation receipt:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-19/task/grouprec-virtualcrowd-corrected-eval/grouprec-full-evaluation/receipt.json`.
- GroupRec world-dynamics result:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-19/task/grouprec-virtualcrowd-corrected-eval/grouprec-world-dynamics-evaluation/world_dynamics_result.json`.
- World-dynamics result / receipt SHA-256:
  `62d8090fc82d9fd357949b72c37331caae9b4247e974fedb10337cc5f48cd2c7` /
  `bc1b9496631b1f1a0e07804df7addc65bc17f870371bebaee87f31c292c34776`.
- Native/interpreter parity receipt:
  `/mnt/data1/hj/.home2/hj/Repo/Code_as_Libs/tmp/2026-08-19/task/grouprec-virtualcrowd-corrected-eval/evaluation-parity-receipt.json`.

## Modification History

- 2026-08-19: Created the unattended campaign after the user selected the
  `C4D_DYCROWD_COMMON_COCO17_VISIBLE_GE9` population and authorized completion
  through a three-method evaluation. Adopted the completed GroupRec full-run and
  corrected two-method results without renaming legacy artifacts.
- 2026-08-19: Completed the additive selected-view contract, persistent
  GroupRec producer/interpreter, real parity and pilot gates, full GroupRec
  reduction, and source-bound three-method comparison. The legacy 167,243-key
  common artifacts remained unchanged.
- 2026-08-19: Reopened only for an additive world-dynamics task selected by the
  user: `ACC-JOINT`, `ACC-ROOT`, `JERK-JOINT`, and `JERK-ROOT`, followed by a
  source-bound GroupRec re-evaluation. The prior completion and artifacts remain
  accepted and immutable.
- 2026-08-19: Completed the additive task with 159,405 matched occurrences,
  156,263 exact acceleration triples, 154,883 exact jerk quadruples, zero failed
  workers, and byte-identical recomputation of all existing GroupRec metrics.
