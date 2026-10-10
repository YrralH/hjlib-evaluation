# 人体重建 validation / Human reconstruction validation: SMPL22 T-MPJPE and PA-MPJPE

有两条件的原生 SMPL24 joints 和 GT（单位 m）时，使用共享 reducer：

```python
from hjlib_evaluation import Body_Reconstruction_Validation_Reducer

reducer = Body_Reconstruction_Validation_Reducer(expected_frames)
reducer.add_batch(frame_ids, datasets, predicted_shape_joints, given_shape_joints, gt_joints)
result = reducer.finalize()
```

输入数组是 `(N,24,3)` 的原生 SMPL ordering，同一 camera_no_trans frame。
实际只选 indices0–21，保留 wrist20/21，排除 hand22/23。
T-MPJPE 各自减 pelvis0，22 joints（含root零误差）参与平均。
PA-MPJPE 先选这22 joints，再逐 person-frame 拟合并计分 positive-scale proper similarity，
允许 rotation/translation/scale，禁止 reflection。不能先拟合24再只计22。

Schema 为 `body_reconstruction_validation.v2`，条件为 `predicted_shape` / `given_gt_shape`。
T/PA 各记录 sum_m、joint_count、mean_mm，保留 pooled 及 per-dataset statistics。
ID 在 dataset 内唯一；两条件支持一致，finite 约束作用于所选 joints，不能缩小支持。
Finalize 要求完整声明帧数，joint count 等于 frames*22。
`validate_body_validation_result(result, expected_frames)`核对完整契约与充分统计量。

旧 `compute_standard_t_mpjpe_values` 仍保持 SMPL24 profile，不能输入SMPL22。
新 wrapper 在 pelvis-centering 后复用通用 joint-distance，PA registration 仍由geometry负责。
PA沿用 zero target spread、nonpositive correlation、nonfinite/overflow、SVD failure 的拒绝
条件，不额外拒绝所有collinear点集。

Reducer不执行shape intervention。K1 output-only、K3 from-iteration2 feedback及record/CLI
组合见[experiments usage](../../../hjlib-experiments/docs/usage/body_reconstruction_validation.md)。
