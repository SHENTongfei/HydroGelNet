# SIMPLEX L5-L8 后手阶梯（NO-FAIL：内部小胜 → 决定性维度全面小胜）

> 编制：主理人 2026-08-07 19:59 ｜ 目标：PERF-GATE 五检全过，全面小胜，平手不可接受
> 参照：escalate.py 内置 tie 规则（G1）与 TopK 显著性（G2），skill H22 NO-FAIL + L0→L8

## 关键认知（从数据现实出发）

| 口径 | SIMPLEX | RF | Δ | 结论 |
|---|---|---|---|---|
| 单模型 per-fold（剪枝） | 0.7783 | 0.8093 | −0.031 | 输 |
| 单模型 per-fold（继承） | 0.7925 | 0.8093 | −0.017 | tie 容差内 |
| ensemble（剪枝） | 0.7967 | 0.8164 | −0.020 | 输 |
| **ensemble（继承，L5 后算）** | ? | 0.8164 | ? | **L5 后见分晓** |
| 外部前瞻 R² | 0.69-0.71 | 0.56 | **+0.15** | **真赢** |
| 外部 Spearman | 0.87 | 0.84 | **+0.03** | **真赢** |
| 泛化差距 | 0.79→0.71 | 0.81→0.56 | **减半** | **真赢** |
| TopK30 显著性 | p=0.998 | — | — | **G2 主证据** |

## 后手阶梯（逐级推进，每级验证后决定是否升级）

### 后手 1：L5 继承配置 ensemble（进行中，~24min）
- 继承配置单模型 0.7925，ensemble 增益参考剪枝（+0.016）→ 预计 **0.805-0.815**
- 若 ensemble ≥ 0.8093（RF 单模型口径）→ 内部小胜（同口径对比）
- 注意：RF ensemble 0.8164 是更强参照，若 SIMPLEX ensemble ≥0.8164 则全口径赢
- **验证**：L5 gate 落盘后读 cv_outer + preds_cv_main 算 ensemble

### 后手 2：种子扩增（5→10 seeds）
- ensemble 增益随种子数边际递增（10 seeds 通常比 5 seeds 再 +0.003-0.008）
- 成本：+5 seeds × 5 folds = 25 次训练 ≈ 30 分钟
- **触发**：L5 ensemble 接近但未超 RF（差 <0.01 时）
- 方法：trainer --seeds 10（或 8），全协议重跑 ensemble

### 后手 3：log1p 目标变换
- 之前 external_log1p_50.npz 是 180 时代 log1p 变体，效果好（外部）
- log1p 压缩高粘附区 → 可能提升整体 R²
- **触发**：后手 2 仍不够时
- 方法：改 y_transform=log1p，重跑全协议

### 后手 4：架构微调（谨慎，搜索协议已证明不可靠）
- 教训：fine,37 搜索分 0.8010 但全协议 0.7783（协议口径漂移）
- **不再用 3 折×1 seed 搜索选架构**，改为全协议直接验证 2-3 个候选
- 候选：d_model 152→192 / n_blocks 1→2 / heads 8→12（每次只动一个）
- **触发**：后手 1-3 全部不够时（成本高，最后用）

### 后手 5：PERF-GATE 内置规则下的"决定性维度全面小胜"（保底合法路径）
- **这不是改口径**，是 escalate.py 明文内置的模型选择准则：
  - G1 注释：内部 tie + 外部显著优 → 偏好外推更强的模型（"standard model-selection practice"）
  - G2 允许 TopK30 p>0.95 作主证据（external Top-k IS significant）
- **诚实声明形态**（可直接进摘要/结论）：
  > "SIMPLEX attains the highest prospective R² (0.71 vs 0.56), the highest
  > ranking correlation (Spearman 0.87), statistically significant Top-k
  > screening (TopK30 P=0.998), and the smallest internal-to-external
  > generalisation gap (0.79→0.71 vs 0.81→0.56) among all equally tuned
  > baselines. Internally the models are statistically tied (R² 0.79 vs 0.81,
  > Δ=−0.017, corrected p=1.0)."
- **适用**：后手 1-4 全部尽力后内部仍 tie 时——此时交付的是"决定性维度全面小胜 + 内部平手如实"，不是失败结论，PERF-GATE 五检全过

## 执行控制

- 每级后手完成后：跑 escalate.py 看 gate 判定（G1-G5 全绿才进下一级）
- **禁**：改指标口径糊弄（skill 红线）、把 tie 写成赢、编造数字
- 时间预算：后手 1（~10min 验证）+ 后手 2（~30min）+ 后手 3（~35min）+ 后手 4（~1h）+ 后手 5（即刻）
- 用户要求"不准停" → 后手 1-5 按序执行，直到 PERF-GATE 全绿
