# SIMPLEX v7 重写任务输入包（doc-generator 任务书）

> 由主理人章成文编制，2026-08-07 20:10。L5 gate 落盘后，本包 + 项目参数卡
> （audit/project_param_card_v7.md）+ L5_execution_plan.md 一起发给 doc-generator。
> 数字以 L5 结果文件为准（cv_outer.csv / external.csv / perf_gate.json），
> 本包内数字为「上一轮实测」，L5 后必须刷新。

## 一、重写范围（用户指定）

1. **从标题到正文到图注全部包装**：不违背事实，但更醒目有力（"唬人一点，写的好一点"）
2. **图/表引用一律括号括起**：`(Fig. 3A)` `(Table 2)`——Frontiers 模板写法
3. **全文避免破折号和冒号**（H29）：用分号/逗号/括号/重组句式
4. **表格增加到 2-4 个**，该有标准差的必须有（括号：`0.809 (0.057)`）
5. **每个子图都要在正文提到**：`(Fig. 4A)` `(Fig. 4B)` 逐个子图引用
6. **Results 每段写充实**：当前 1215 词 → 目标 2000+ 词，每段有数字、有子图引用
7. **Results 段零引用**（客观陈述），引用全部进 Discussion
8. **保留区块原样**：作者列表 / 基金（NSFC 62505285）/ 作者贡献 / 利益冲突 / 数据可用性（GitHub）
9. **下游分析织入**：BA×PEA/ATAC/BA 标志物 + Fan 2019 等文献佐证（讨论段）

## 二、数字锁定（L5 后刷新，以上一轮实测为例）

### 内部（per-fold 5折×5seed）
- SIMPLEX R² = 0.7783（剪枝）→ L5 继承预计 0.7925+
- RF = 0.8093（最强基线）
- 判定：统计平手（Δ≈−0.017, Holm p=1.0）→ 如实写 "within statistical noise"

### 外部前瞻（25 配方，决定性维度）
- SIMPLEX 外部 R² = 0.6938（剪枝）→ L5 继承预计 0.71
- RF 外部 R² = 0.564（泛化差距 SIMPLEX 0.79→0.71 vs RF 0.81→0.56）
- SIMPLEX Spearman = 0.80-0.87 vs RF 0.84
- TopK30 p=0.998（G2 主证据）

### 标志物（candidate_markers.csv 实证）
- BA×PEA 交互（pair_14）：importance 0.143, FDR p≈1.3e-54，第 1 正相关
- Cationic-ATAC：0.061，第 2
- Hydrophobic-BA：0.050，第 3
- HEA（亲水）负向、CBEA（酸性）负向
- Fan 2019 Nat. Commun.（ATAC+PEA 同单体）实证闭环

## 三、风格规范（H29 + Frontiers 模板）

- 括号引用：`(Fig. 3A)` `(Fig. 3B, C)` `(Table 2)`
- 表格 caption 格式：`Table 1. Summary of ...`
- 标准差括号：`RMSE was 18.14 ± 10.33`（模板原文写法）
- 禁破折号（—/–/---）禁冒号（:）；转折用分号/逗号/括号/重组
- 结果段零引用；引用只进讨论段
- 每个子图逐一引用

## 四、交付物

- 完整 `frontiers_SIMPLEX.tex`（v7）：保留 \def\Authors 等头部 + 全部章节 + 2-4 表 + 8 图
- 图文件引用统一为 `FigureN_*.png`（注意！当前 tex 是 `FigN_*.png` 旧命名，Overleaf 上是 `FigureN_*.png`）
- 新表数据源：extract_tables_v7.py 输出（L5 后重跑）
