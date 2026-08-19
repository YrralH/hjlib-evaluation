# Task - Corrected World Dynamics Metrics

- State: complete
- Design residence:
  [`../../../docs/design/tasks/corrected-crowd-world-dynamics/README.md`](../../../docs/design/tasks/corrected-crowd-world-dynamics/README.md)

Add the four GT-relative world-space dynamics metrics without changing the
frozen corrected schema-v1, then recompute GroupRec's existing selected-view
metrics and the additive dynamics result on the accepted 159,405-occurrence
population. Completion requires eight scenes, zero worker failures, source-bound
receipts, and exact `ACC-JOINT`/legacy `ACCEL-WORLD` parity.

Completed result:

- selected/matched: 159,405 / 159,405;
- exact acceleration triples: 156,263;
- exact jerk quadruples: 154,883;
- dynamics result SHA-256:
  `62d8090fc82d9fd357949b72c37331caae9b4247e974fedb10337cc5f48cd2c7`;
- receipt SHA-256:
  `bc1b9496631b1f1a0e07804df7addc65bc17f870371bebaee87f31c292c34776`;
- legacy result remained byte-identical to accepted SHA-256
  `f7c36b7b7d038002a5b1fd5accbb566b3a9208afa7803139768312c69e8c2c36`.
