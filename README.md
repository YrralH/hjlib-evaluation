# hjlib-evaluation

动态场景评测协议:给定一个数据集 + 一批预测(已存 dump,或一个待评 ckpt),它负责
**逐 test segment 把预测对齐世界空间 GT → 归约出指标表**(MPJPE / T-MPJPE / Jitter;
2D OKS 占位)。推理与归约由 per-segment 预测 dump 解耦——**评已有 dump 不需要 live 网络**。

定位:**与 hjlib-trainer 对称的执行层**。trainer 编排训练,本仓编排评测;上面由
`hjlib-experiments` 编排。本仓**不持有任何模型 / 数据集定义**,只做评测 harness +
per-dataset GT/testset 接线；另提供 unit-neutral scalar trajectory residual
summary/reduction leaf API。

> 从 monolith `lib_dynamic_hvip/test/protocol_dynamic/` 迁出(见
> [docs/design/migration.md](docs/design/migration.md))。**端口 Phase 1-5 done**(脚手架 +
> testset + GT + tester/eval_reducer/dump_reader,real-data 绿)+ **Phase 6 parity/behavior
> 已在 `hjlib-migration-tests/evaluation/` 跑绿**(2026-06-25,只差 SHA pin);commit/push 待 family 收敛。

## 安装

```bash
conda activate hjlib_py312
pip install -e .
```

## 依赖方向

```
hjlib-experiments → { hjlib-evaluation, hjlib-network, vis 仓 }
hjlib-evaluation  → { hjlib-dataset-assembly, hjlib-dataset-std,
                      hjlib-skeleton, hjlib-geometry }   # 已 pin
                    （+ hjlib-network / hjlib-smpl:live driver 落地时再 pin）
```

已核实 **无环**:assembly / dataset-std / skeleton 均纯下游(network 不反向依赖)。详见
[docs/design/README.md](docs/design/README.md)。

## 更多

- [docs/usage/](docs/usage/) —— 怎么调用(评已有 dump / 评新 ckpt 的端到端流程)
- [docs/design/](docs/design/) —— 怎么修改(四层架构 + 与 assembly 的 divider 注入接线 +
  迁移记录)
- [campaigns/](campaigns/) —— 跨 task 的持久工作状态与交付入口
- Corrected crowd metric public schema and leaves are documented in
  [docs/usage/](docs/usage/) and the Campaign 03
  [Layered Design](docs/design/tasks/virtualcrowd-corrected-metric-protocol/README.md).
- The additive selected-population API used by the three-method VirtualCrowd
  comparison is documented in
  [corrected_crowd_selected_population.md](docs/usage/corrected_crowd_selected_population.md).
- Stable VirtualCrowd default/native profile names and their composition are
  documented in
  [virtualcrowd_evaluation_profiles.md](docs/usage/virtualcrowd_evaluation_profiles.md).
- GitHub remote:`YrralH/hjlib-evaluation`(建仓后)
- family 入口:[../CLAUDE.md](../CLAUDE.md)
