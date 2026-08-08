# Overleaf 6a6a0834 交付清理清单（v7 重写后执行）

> 编制：主理人 2026-08-07 20:20 ｜ 项目树已拉取确认

## 一、要删除的旧图（tex 引用统一为 FigureN 后，FigN 全部删除）

| 文件 | 项目树 ID | 备注 |
|---|---|---|
| Figures/Fig1_pipeline.png | 6a747c804a2ae32a2f4da2cd | 旧命名（v7 改 Figure1_pipeline） |
| Figures/Fig2_architecture.png | 6a747c862f9e70c125ac3f5d | 旧命名（v7 改 Figure2_architecture） |
| Figures/Fig3_dataset.png | 6a747c8d4a2ae32a2f4dad05 | 旧命名（v7 改 Figure3_dataset） |
| Figures/Fig4_internal_cv.png | 6a747c95d6565336f5ab6ded | 旧命名（v7 改 Figure4_internal_cv） |
| Figures/Fig7_ablation.png | 6a747dd212f91e15e1f85425 | 旧命名（v7 改 Figure7_ablation） |
| Figures/Fig8_interpretation.png | 6a747ddb33ed1ea89b515a21 | 旧命名（v7 改 Figure8_interpretation） |

> 注：Fig5/6 旧版（swap 事故那对）之前已删，不在当前树。

## 二、保留的图（FigureN_*.png + .pdf，与 v7 tex 引用一致）

- Figures/Figure1_pipeline.png（若 v7 重画后重新上传）
- Figures/Figure2_architecture.png（同上）
- Figures/Figure3_dataset.png + .pdf ✓
- Figures/Figure4_internal_cv.png + .pdf ✓
- Figures/Figure5_benchmark.png + .pdf ✓
- Figures/Figure6_external.png + .pdf ✓
- Figures/Figure7_ablation.png + .pdf ✓
- Figures/Figure8_interpretation.png + .pdf ✓

## 三、v7 tex 引用统一（\includegraphics）

- 当前 tex 用 `Figures/FigN_*.png`（无 e）→ **全部改为 `Figures/FigureN_*.png`**
- 图注（caption）描述 9 面板 (A)-(I) 的必须与图实际面板一致（figure_contract_v7.md 已定稿）
- 每张图正文逐子图引用：(Fig. 3A) (Fig. 3B) ...（H29 括号引用法）

## 四、supple（frontiers_SupplementaryMaterial.tex）同步

- 与主文数字一致（内部/外部/排序/Top-k）
- 补充表格：per-fold 全指标（RMSE/MAE/NRMSE/Pearson/Spearman/CCC/TopK20/TopK30）
- 补充材料也用括号引用、去破折号冒号

## 五、GitHub（SHENTongfei/HydroGelNet）同步

- paper/frontiers_SIMPLEX.tex（v7）→ 仓库
- paper/reference_final.bib（93 条）→ 仓库
- figures/（FigureN_*.png+pdf 16 个）→ 仓库
- code/（figures_v2.py 等更新）→ 仓库
- 提交信息：v7 rewrite（全面小胜口径 + 括号引用 + H29 风格）
