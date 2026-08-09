# Figure 1 — Study pipeline 重画详细指示（v13）

> 目标：重画 `Figure1_pipeline.png`（研究流程图，5 阶段横向流水线）。
> 现状问题：旧图使用 v1 过时数字（341/180/161 样本、Spearman 0.50、Top-20 0.25），
> 且代码存在 `_pbox` vs `_p_pbox` 命名不匹配 bug（fig1 从未跑通过当前版本）。
> 本指示给出：正确数字、布局规范、配色、文字内容、代码修改点。

---

## 1. 画布与整体布局

| 项 | 值 |
|---|---|
| 画布 | `figsize=(9.8, 4.8)`，坐标 `xlim=(0,106)`, `ylim=(0,44)`，`axis("off")` |
| 阶段数 | 5 段横向流水线（Data → Training region → SIMPLEX → Prospective validation → Screening & insight） |
| 每阶段 | 1 个主框（彩色标题框）+ 3-4 个细节框（同色系）+ 竖向箭头 |
| 阶段间 | 横向粗箭头（`-|>`，lw=1.8）连接 |
| 阶段标签 | 顶部居中（y≈41.6），下带细横线 |

## 2. 配色（沿用现有 PAL，保持全图一致）

| Key | 阶段 | 浅底 | 深边 |
|---|---|---|---|
| `blue` | 1 · Data | `#CCE4FC` | `#2E6DA4` |
| `green` | 2 · Training | `#E4FCFC` | `#2E8B57` |
| `orange` | 3 · SIMPLEX | `#FCE4CC` | `#E67E22` |
| `red` | 4 · Prospective | `#FCE4E4` | `#C0392B` |
| `purple` | 5 · Screening | `#FCE4FC` | `#8E44AD` |

（SIMPLEX 主框建议加粗 `bold=True, fs=8.5` 突出）

## 3. 每阶段文字内容（用 v13 正确数字！）

### Stage 1 · Data（blue，x≈1）
- 主框：`Public dataset` / sub `Nature 2025 · MIT licence`
- 细节框：
  1. `341 formulations`（总 316 内 + 25 外 = 341）
  2. `6 monomers on the composition simplex`
  3. `Target: adhesion strength (kPa)`

### Stage 2 · Training region（green，x≈20）
- 主框：`Training set` / sub `n = 316 · internal cohort`
- 细节框：
  1. `5-fold grouped CV`
  2. `10 seeds · 50 models`（10 seeds × 5 folds，非旧的 5×25）
  3. `Ablation-gated components`

### Stage 3 · SIMPLEX（orange，x≈41）
- 主框：`SIMPLEX` / sub `dual-modality encoder`
- 细节框：
  1. `Monomers + pairwise terms`（6 + 15 = 21 features）
  2. `ResBlock x2 + interaction attention`
  3. `Mixup · SWA · domain constraint`

### Stage 4 · Prospective validation（red，x≈62）
- 主框：`Prospective cohort` / sub `n = 25 · model-discovered`
- 细节框：
  1. `Held-out during all tuning`（未被模型选择接触过）
  2. `Evaluated once, after freezing`
  3. `High-adhesion region (62–251 kPa)`

### Stage 5 · Screening & insight（purple，x≈83）
- 主框：`Ranking + insight`
- 细节框：
  1. `Spearman ρ = 0.87`（前瞻相关系数，非旧 0.50）
  2. `Top-20 precision 0.90`（非旧 0.25）
  3. `Permutation importance → composition synergy`
  4. 底部主框：`Accelerated screening`

## 4. 正确数字速查（全文统一，禁止旧值）

| 指标 | v13 正确值 | 旧图错误值 |
|---|---|---|
| 内部样本 | **316** | 180 |
| 外部样本 | **25** | 161 |
| 总样本 | **341** | 341（对） |
| CV R² | **0.79** | — |
| 前瞻 R² | **0.69** | — |
| Spearman | **0.87** | 0.50 |
| Top-20 | **0.90** | 0.25 |
| 种子×折 | **10×5 = 50** | 5×5 = 25 |

## 5. 代码修改点（figures_v2_backup.py L196-251）

1. **修复命名 bug**：在 `fig1` 前加别名：
   ```python
   _pbox = _p_pbox
   _parrow = _p_parrow
   _stage_label = __stage_label
   _psave = _p_psave
   ```
2. **替换所有旧数字**（341/180/161/Spearman 0.50/Top-20 0.25/47→154 → 上表正确值）
3. Stage 2 细节框 "5 seeds · 25 models" → **"10 seeds · 50 models"**
4. Stage 4 主框 "n = 161 · SMBO-discovered" → **"n = 25 · model-discovered"**；细节框 "Target-value shift (mean 47 → 154 kPa)" → **"Held-out during all tuning"**
5. 用 `_psave` 输出 PNG(600dpi) + PDF
6. 渲染后跑 `Layout.check()` 确认 0 重叠

## 6. 验证流程

```bash
cd C:\Users\TS\WorkBuddy\HydroGelNet\code
python figures.py --only 1        # 渲染 Figure1
python figures.py --only 2        # 确认 fig2 不受影响
```
- 目视检查：5 阶段横向排布、箭头连通、无框重叠
- Qwen36 视觉 QA 防重叠（本地 ollama）
