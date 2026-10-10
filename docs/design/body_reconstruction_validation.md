# 双条件人体重建 validation

`body_reconstruction_validation.py`持有method-neutral metric wrapper、唯一ID与充分
统计量reducer。原生SMPL24输入先选0–21，再计算root-centred T与proper positive-scale PA。
旧standard T的SMPL24 profile保持不变；新22 wrapper复用通用joint-distance leaf。
PA fitting与score都只使用22 joints，registration由geometry负责。

V2 contract明确SMPL22/root0含root/frame/units/alignment及两个shape条件；两条件完整支持，
按joint sum/count汇总，不平均batch means。跨dataset与T/PA计数、finite统计、重复ID、
完整支持均有可失败控制；任一条件metric失败不提交reducer state。
Hand22/23扰动、root分母和先选后fit控制接入主smoke。

Shape injection、checkpoint/live orchestration、legacy SMPL24 T archive对照和resume由
experiments拥有。Network只拥有given-shape IEF与output行为，不依赖evaluation。
旧v1/retained普通结果保留原schema，不能通过v2 validator作为SMPL22结果。
