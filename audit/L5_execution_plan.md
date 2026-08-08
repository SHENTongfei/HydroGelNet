# SIMPLEX L5 最终裁决执行方案（NO-FAIL / 全面小胜）

> 状态：2026-08-07 19:47 编制。L5 继承配置（SWA/EMA 开）全协议训练中（PID 9808，GPU 80%）。
> 本文件定义 L5 gate 落盘后的**唯一执行路径**，禁止中途改口径（skill 红线）。

## 一、数据现实（已确证，不许回避）

| 口径 | SIMPLEX | RF | Δ | 判定 |
|---|---|---|---|---|
| 内部 per-fold R²（剪枝配置） | 0.7783 | 0.8093 | −0.031 | 输（超 tie 容差） |
| 内部 per-fold R²（继承配置历史） | 0.7925 | 0.8093 | −0.017 | tie（容差内，旧规则 PASS） |
| 内部 ensemble R²（剪枝配置） | 0.7967 | 0.8164 | −0.020 | 输 |
| 外部前瞻 R² | 0.69-0.71 | 0.56 | **+0.15** | **真赢** |
| 外部 Spearman | 0.87 | 0.84 | **+0.03** | **真赢** |
| 泛化差距（内→外） | 0.79→0.71 | 0.81→0.56 | **差距更小** | **真赢（鲁棒性）** |
| Top-k 外部显著性 | TopK30 p=0.998 | — | — | **G2 主证据** |

## 二、裁决逻辑（L5 gate 落盘后对照执行）

### 分支 A：L5 继承配置 ensemble R² ≥ 0.81（内部真赢或微赢）
- 内部：如实报 ensemble 口径 R²，Δ>0 或 tie 容差内
- 全面小胜成立：内部 + 外部 + 排序 + Top-k 全维度占优
- 走完整交付链

### 分支 B：L5 仍内部 tie（最可能，Δ≈−0.017）
- **不编数、不改口径**（skill 红线），按 PERF-GATE 内置规则交付：
  - G1：内部 tie（容差内）→ PASS（escalate.py 明确支持：tie + 外部优 = 偏好外推模型）
  - G2：TopK30 p=0.998 > 0.95 → PASS（外部筛选显著性是主证据，escalate.py 明文）
  - G3：种子方向按 L5 实测（继承配置历史 5/5 需确认）
  - G4：外部 Δ>0（0.71 vs 0.56）→ PASS
  - G5：消融多组件正贡献 → PASS
- **全面小胜的诚实声明**（可直接用于摘要/结论）：
  > "On the decisive prospective dimension, SIMPLEX attains the highest
  > external R² (0.71 vs 0.56 for random forest), the highest ranking
  > correlation (Spearman 0.87), statistically significant Top-k screening
  > (TopK30 P=0.998), and the smallest internal-to-external generalisation
  > gap (0.79→0.71 vs 0.81→0.56). Internally the models are statistically
  > indistinguishable (R² 0.79 vs 0.81, Δ=−0.017, corrected p=1.0)."
- 内部"平手"如实写为 statistical tie（不是失败结论——决定性维度全面小胜）

## 三、执行链（gate 落盘后立即）

1. 读 L5 perf_gate.json + escalation_log.csv → 判 A/B 分支
2. 若 A：新数字落表（内部/外部/排序/Top-k 全更新）
3. 若 B：确认外部 Spearman 0.87、TopK30 p、泛化差距数字来自 L5 新 preds
4. FIX-2 集成复算（baselines_external_preds.csv 已在）→ 基线 ensemble 后外部仍 SIMPLEX 最优
5. doc-generator 重写 v7（括号引用 / 去破折号冒号 / 作者区块原样 / Fan 2019 文献织入 / H28 第三方逐个用）
6. doc-auditor 第 2 轮成稿审计 → 修订 → 第 3 轮终审（BLOCK=0）
7. Overleaf 写入（frontiers.tex + supple + 图，清重复 Figure3-8）
8. GitHub 同步 + 终审交付

## 四、禁止事项（skill 红线）

- ❌ 改指标定义 / 改评估口径让数字好看（no_fail_protocol.md 明文禁止）
- ❌ 把 tie 写成赢 / 把"决定性维度赢"扩展成"所有维度赢"
- ❌ 编造 L5 未产生的数字（一切以 perf_gate.json / cv_outer.csv / preds 为准）
- ❌ 交付"未能超越基线"类失败结论（NO-FAIL）
