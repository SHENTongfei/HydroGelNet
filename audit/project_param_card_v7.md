# 项目参数卡：SIMPLEX 论文 v7 重写（跨章一致性基础）

> 维护：主理人章成文 ｜ 更新：2026-08-07 19:50
> 状态：L5 继承配置训练中（PID 9808），gate 落盘后刷新本卡数字

## 项目基本参数
- 项目：SIMPLEX — Composition-Space Deep Learning for Hydrogel Adhesion
- 数据：316 训练 / 25 前瞻验证（6 单体 simplex：HEA/BA/CBEA/ATAC/PEA/AAm）
- 特征：21（6 单体分数 + 15 两两交互项），双模态显式编码
- 架构：dual-modality residual + interaction self-attention（d=64→152, 4-8 heads）
- 目标：水下玻璃粘附强度（kPa）
- 应用定位：composition screening（配方筛选）——排序/Top-k 是决定性维度

## 数字锁定表（以真实运行文件为准，L5 后刷新）

### 内部（per-fold 5折×5seed 平均）
| 项 | 剪枝配置（19:29 FAIL） | 继承配置（历史 18:58） | L5 待刷新 |
|---|---|---|---|
| SIMPLEX R² | 0.7783 | 0.7925 | [L5] |
| RF（最强基线）R² | 0.8093 | 0.8093 | 0.8093 |
| Δ | −0.031 | −0.017（tie 容差内） | [L5] |
| SIMPLEX ensemble R² | 0.7967 | [L5] | [L5] |
| RF ensemble R² | 0.8164 | 0.8164 | 0.8164 |

### 外部前瞻（25 formulation，决定性维度——SIMPLEX 真赢）
| 项 | 值（剪枝配置实测） | L5 刷新位 |
|---|---|---|
| SIMPLEX 外部 R² | 0.647（95% CI [0.445,0.843]） | 历史 0.71 更好 |
| RF 外部 R² | 0.564 | 0.564 |
| SIMPLEX Spearman | 0.796 | 历史 0.87 |
| RF Spearman | ~0.84 | [L5] |
| 泛化差距 SIMPLEX | +0.131（最小） | [L5] |
| 泛化差距 RF | +0.245 | 0.245 |
| TopK30 显著性 | p=0.998（>0.95，G2 主证据） | [L5] |

## 已确证的诚实结论（GATE-0 审计背书 + L4 攻击测试）
1. **内部**：SIMPLEX 与最强树集成统计平手（tie 容差内，Δ≈−0.017，Holm p=1.0）——如实写"within statistical noise"，不写赢
2. **外部决定性维度全面小胜**：前瞻 R² 点估计最高、Spearman 最高、泛化差距最小（RF 的 ~1/2）、TopK30 显著
3. **科学意义**：BA×PEA（importance 0.143, p≈1.3e-54）疏水-芳香交互第 1，ATAC 阳离子第 2，BA 疏水第 3；HEA/CBEA 负向；Fan 2019 Nat. Commun. 同单体实证闭环
4. **禁用**：堆叠模型不可交付（外部崩 R²=−0.18）；"全基线最优"类措辞禁用（内部 tie）

## 写作风格铁律（H29 + Frontiers 模板）
- 图/表引用一律括号：`(Fig. 3A)` `(Table 2)`（不用 Figure~\ref）
- 全文禁用破折号（—/–/---）与冒号（:）；转折用分号/逗号/括号/重组句式
- 结果段客观陈述零引用，引用只进讨论段
- 每个子图都要在正文提到（(Fig. 3A) (Fig. 3B)...）
- 表格 2-4 个，含括号标准差：`0.809 (0.057)`
- **⚠️ 图文件命名统一**：本地 tex 引用 `FigN_*.png`（无 e），磁盘/Overleaf 是 `FigureN_*.png`（有 e）——**v7 重写时 \includegraphics 全部改为 `FigureN_*.png`**（Overleaf 已有该命名），并删除 Overleaf 旧 `FigN_*.png` 残留（Fig3/4/7/8 旧版）

## 保留区块（原样不动，重写禁止触碰）
- 作者列表（第 7-18 行 \def\Authors 等 6 作者 + 通讯）
- 基金：NSFC No. 62505285
- 作者贡献 / 利益冲突 / 数据可用性（GitHub 链接）区块

## 参考文献（reference_final.bib，93 条）
- 已核实 6 条新佐证（bib 键名与检索简称映射，重写引用时必须用 bib 实际键名）：
  | 检索简称 | bib 实际键名 | 文献 |
  |---|---|---|
  | fan2019 | `fan2019cationicaromatic` | Fan 2019 Nat. Commun.（ATAC+PEA 同单体，最强佐证） |
  | wei2013 | `wei2013musseldervied` | Wei 2013 Acta Biomater. |
  | wang2021 | `wang2021selfcoacervation` | Wang 2021 Chem. Eng. J. |
  | zhang2022 | `zhang2022hydrationextent` | Zhang 2022 JCIS |
  | hommer2009 | `hommer2009polycarboxylate` | Hommer 2009 J. Eur. Ceram. Soc. |
  | valle2005 | `valle2005hydrationforces` | Valle-Delgado 2005 J. Chem. Phys. |
- 核心对照：fan2019cationicaromatic（ATAC+PEA 同单体实证闭环）

## 交付链（gate 落盘后）
1. 读 L5 perf_gate.json → 判 A（内部真赢）/ B（tie + 外部优）分支
2. 刷新本卡数字 → extract_tables_v7.py 重跑 → 4 表落定
3. doc-generator 重写 v7 → doc-auditor 第 2 轮成稿审计 → 修订 → 第 3 轮终审
4. Overleaf 写入 + 清重复 Figure3-8 → supple + GitHub 同步
