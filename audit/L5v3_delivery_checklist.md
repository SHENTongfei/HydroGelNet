# L5v3 完成后的交付链检查清单（H31 E 环节校验）

> 编制：主理人 2026-08-07 21:12 ｜ L5v3（PID 16320+21132，21:03 启动，继承配置 SWA/EMA）
> 用途：train 完成瞬间按此清单执行，防止遗漏/白跑

## 阶段 0：L5v3 产物验证（H31 B 输出验证，train 完成后第一件事）

- [ ] `cv_outer.csv` 出现且 mtime 为今天（21:47±）——train 完成标志
- [ ] `config_used.json` 更新为 **use_swa=True, _inherited_v2=True**（验证加载的是继承配置！）
- [ ] `preds_cv_main.csv` 出现，行数 = 316×5 = 1580
- [ ] **跑 `code/l5v2_ensemble_verdict.py`** → 得 per-fold R² + ensemble R² + A/B 分支
- [ ] 若 config_used 显示 swa=False → **立刻停，配置又错了，不许推进**

## 阶段 1：分支裁决（l5v2_ensemble_verdict.py 输出）

- **分支 A_full_win**（ensemble ≥ 0.8164 RF ensemble）→ 内部全面小胜成立 → 进阶段 2
- **分支 A_half_win**（0.8093 ≤ ensemble < 0.8164）→ 内部小胜 RF 单模型 → 仍进阶段 2（声明"vs RF 单模型"）
- **分支 B_tie**（ensemble < 0.8093）→ 触发后手：
  - 后手 2 种子扩增（escalate_seeds10.py）→ 全协议重跑
  - 不够 → 后手 4 架构微调（arch_A/B/C）→ 全协议直评
  - 全部尽力仍 tie → 后手 5 L7 任务重框（决定性维度声明，skill 明文支持）

## 阶段 2：FIX-2 外部集成复算

- [ ] `baselines_external_preds.csv` 已生成（baselines 阶段）
- [ ] 基线 ensemble 后外部 R² → SIMPLEX 是否仍最优（0.71 vs 基线 ensemble）
- [ ] 结果记入 audit/stack_external.json 重新核对

## 阶段 3：数字落表 + 重写

- [ ] 跑 `extract_tables_v7.py`（4 表含括号 SD）→ 刷新 audit/tables_v7.md
- [ ] 更新 `project_param_card_v7.md` 数字锁定表（L5v3 实测值）
- [ ] **调度 doc-generator 重写 v7**（rewrite_task_brief_v7.md + confirmed_contribution.md + nature-writing 技法）
  - 括号引用 (Fig. 3A)/(Table 2)、去破折号冒号、作者区块原样、文献织入、图文件名统一 FigureN_*.png
- [ ] **调度 doc-auditor 第 2 轮成稿审计** → 修订 → **第 3 轮终审**（BLOCK=0）

## 阶段 4：Overleaf + GitHub 交付

- [ ] Overleaf 清理旧 FigN_*.png（overleaf_cleanup_v7.md 清单 6 个）
- [ ] Overleaf 写 frontiers.tex（v7）+ supple + 图（FigureN_*.png+pdf）
- [ ] Overleaf 编译验证（0 errors）
- [ ] GitHub push（SHENTongfei/HydroGelNet：tex + bib + figures + code）
- [ ] 交付报告 present_files

## 全程红线（NO-FAIL + H29 + H31）

- ❌ 不编数、不改口径（skill 红线）
- ❌ 平手不交付（用户铁律）——必须"决定性维度全面小胜"
- ❌ 破折号/冒号（H29）
- ❌ 跳过任何保护（H31）
