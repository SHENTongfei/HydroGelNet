# Final QA — 三轮审计终审报告（H35 / Phase 14.5）

日期：2026-08-09 | 项目：HydroGelNet (SIMPLEX) | 版本：v13

## R1 — Qwen 图片审计（本地 qwen36:latest）

| 图 | 首轮 | 问题 | 修复 | 复审 |
|---|---|---|---|---|
| Fig1 | FAIL | 用数字 1-5 非 A-E 字母；SIMPLEX 框文字被框边裁 | 加 `_pbox`/`_parrow` 等别名（潜伏 bug 修复）；阶段标签 1-5 → A-E；7/7 过时数字更新（180→316、161→25、0.50→0.87、0.25→0.90）；补 FancyArrowPatch import；outdir 修复 | **PASS**（21 元素 0 重叠） |
| Fig2 | PASS | — | — | PASS |
| Fig3 | PASS | — | — | PASS |
| Fig4 | PASS | — | — | PASS |
| Fig5 | PASS | — | — | PASS |
| Fig6 | FAIL | G 图例 XGB/KNN 挤；E "CV 0.792" 压蓝条 | G labelspacing 0.30→0.45；E pad 0.018→0.030 | **PASS** |
| Fig7 | PASS | — | — | PASS |
| Fig8 | FAIL | B 橙点盖 "Hyd-BA" 末尾；H xlabel 贴右缘 | B 点 60→45、值标签 0.06 偏移、字号 5.6 | **PASS** |

**R1 结论：8/8 PASS**（剩余均为 LOW/MED 可接受级，已记录）

## R2 — 恶意视角文字审计（机检）

| 检查项 | 结果 |
|---|---|
| H29 em-dash / en-dash = 0 | ✅ 0 |
| H29 正文散文冒号 = 0 | ✅ 0（首轮 75 为 \Fref{fig:/eq:/label 语法+主标题冒号，改进正则后 0） |
| Results 段零 cite | ✅ 0 |
| 引用 0 悬空 / bib ≥ cited | ✅ 0 悬空，189 bib ≥ 110 cited |
| 关键数字一致性（0.7924/0.6946/0.6342/0.87/0.0724/0.0631/316/25/50） | ✅ 全部命中 |
| 无占位符 / 无 ? 引用 / 无 null / 无 TODO | ✅ 0 |
| 结构 begin/end 1:1、8 图 4 表 | ✅ |
| bibtex 干净（本地编译 blg） | ✅ 0 issue |

**R2 结论：22/22 PASS**

## R3 — 交叉终审（引用真实性以 citation-finder 为准，H37）

- **引用核验**：33 个 recent* 新文献 + 8 个经典文献 = **41 条全部 Crossref DOI 有效**（10.1038/s41524...、10.1038/s41467...、10.1002/pol... 等全部真实）
- citation-finder `search_all.py` 脚本异常（exit 1 不落盘）→ **降级使用其内部数据源 OpenAlex 直查 + Crossref 交叉**（H37 允许，注明降级）
- **R1/R2 修复复审**：Fig1/6/8 重渲染后 qwen PASS；tex 复审 title/引用/结构全过
- **云端 vs 本地一致性**：发现云端 title short 为 `[SIMPLEX]`（旧版残留）→ **已推送 v13 最新 tex 同步**，编译 latexmk-errors: 0、latex-runs: 2

**R3 结论：PASS**（41/41 引用真实 + 云端本地一致）

## 最终判定

**三轮全 PASS → 允许交付。** 遗留说明：
- Fig6G "SIMPLEX" 标签与图例轻微接近（LOW，可接受）
- Fig8F y 轴标签贴近轴线（MED，可接受）
- citation-finder search_all.py 脚本 bug 已记录（下次修），本次按 H37 降级 OpenAlex 直查

## 变更清单（v13）
- `code/figures_v2_backup.py`：fig1 别名修复 + FancyArrowPatch import + outdir → paths.FIGURES_DIR + A-E 字母 + 7 处数字更新
- `code/figures.py`：Fig6G labelspacing、Fig6E pad、Fig8B 点/标签
- `audit/rewrite_baseline/v7_merged.tex`：已推 Overleaf（frontiers.tex v13）
