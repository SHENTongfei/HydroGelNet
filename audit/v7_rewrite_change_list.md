# SIMPLEX v7 重写改造清单（doc-generator 执行，含行号锚点）

> 基于 current_full.tex（33632 chars）逐行审计，2026-08-07 21:25
> 数字以 L5v3 实测为准（本清单用现有值标注改造点，L5v3 后统一刷新）

## 一、图文件命名统一（\includegraphics）

| 行号 | 现值 | 改为 |
|---|---|---|
| 60 | Figures/Fig1_pipeline.png | Figures/Figure1_pipeline.png |
| 82 | Figures/Fig2_architecture.png | Figures/Figure2_architecture.png |
| 106 | Figures/Fig3_dataset.png | Figures/Figure3_dataset.png |
| 117 | Figures/Fig4_internal_cv.png | Figures/Figure4_internal_cv.png |
| 148 | Figures/Fig6_external.png | Figures/Figure6_external.png |
| 159 | Figures/Fig5_benchmark.png | Figures/Figure5_benchmark.png |
| 170 | Figures/Fig7_ablation.png | Figures/Figure7_ablation.png |
| 181 | Figures/Fig8_interpretation.png | Figures/Figure8_interpretation.png |

## 二、括号引用法（Figure~\ref → (Fig. X)）

- 全文 `Figure~\ref{fig:XXX}` 全部改为 `(Fig. N)` 形式：
  - fig:pipeline → (Fig. 1)、fig:modelarch → (Fig. 2)、fig:dataset → (Fig. 3)
  - fig:cv → (Fig. 4)、fig:bench → (Fig. 5)、fig:ext → (Fig. 6)
  - fig:abl → (Fig. 7)、fig:interp → (Fig. 8)
- **每个子图逐一提**：(Fig. 4A) (Fig. 4B) ... (Fig. 4I) —— Results 段每段至少提到所属图的所有面板
- 表格引用：(Table 1) (Table 2) ...

## 三、H29 去破折号去冒号（全文 32 破折号 + 43 冒号）

- `---`（em-dash）→ 重组句式或分号
- `--`（en-dash，数字区间）→ 保留区间语义但检查是否可换 "to"（如 `$0.60$--$0.69$` → `0.60 to 0.69`）
- `:`（冒号）→ 重组（"including/namely/such as" 或括号）
- 每节写完 grep 自查：`--` 和 `:` 计数

## 四、Results 段零引用（引用全进 Discussion）

- 第 144 行（前瞻结果段）：无 \cite ✓（保持）
- **第 177 行（Interpretation 段）有 \cite{cai2021...zhang2020catechol}** → 这段是"结果呈现"但含引用，需拆分：
  - 结果客观陈述（BA×PEA 0.143 第一、ATAC 0.061 第二、HEA/CBEA 负向）→ 留在 Results，去掉引用
  - 机理解释（静电络合/疏水交互驱动）→ 移到 Discussion 相应段，带引用
- 第 190/194/198 行（Discussion）引用保留 ✓

## 五、Results 段充实（当前 1215 词 → 目标 2000+ 词）

每个子节：
- 开句用 Claim-first 或 Method-first 模板（nature-writing 05-results.md）
- 每段有具体数字 + 子图引用 (Fig. XA)
- 不写解读（解读进 Discussion）

## 六、表格增加（1 表 → 2-4 表，含括号 SD）

| 表 | 内容 | 数据源 |
|---|---|---|
| Table 1（已有） | 前瞻 25 配方筛选指标 | baselines_external.csv |
| Table 2（新增） | 内部 CV 每模型 R²/RMSE/MAE（mean (SD)） | extract_tables_v7.py Table 1 |
| Table 3（新增） | 泛化差距（内部→外部 R² per model） | extract_tables_v7.py Table 3 |
| Table 4（新增） | 消融贡献（ΔR², Holm p） | ablation_results.csv |

## 七、保留区块（原样不动）

- 第 6-18 行 \def\Authors 等（作者/地址/通讯）
- 第 206-224 行 COI/贡献/基金/致谢/数据可用性
- 基金 NSFC 62505285、GitHub 链接

## 八、标题/摘要/图注包装（不违背事实）

- 标题保持 SIMPLEX 主体，可微调副标题措辞更醒目
- 摘要用 Claim-first 开句（"SIMPLEX attains the highest prospective R² (0.71)..."）
- 图注已 9 面板齐全（figure_contract_v7.md 定稿），逐面板描述与图一致
