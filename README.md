# HGDNet

Research project

Scaffolded on 2026-08-06 08:46 by the
`do-sci-research` skill.

## Project environment (isolated clone of the ready-made CUDA env)

```
env name : HydroGelNet
python   : C:\Users\TS\.conda\envs\HydroGelNet\python.exe
```

本项目有**自己专属的 conda 环境**（从 py311 用 robocopy 克隆而来，torch +
CUDA 现成）。所有命令都用上面这个解释器，不要用别的 python。

## Run everything

```bash
cd "C:/Users/TS/WorkBuddy/HydroGelNet\code"
"C:\Users\TS\.conda\envs\HydroGelNet\python.exe" run_all.py --all        # GPU 自动检测，无需任何配置
"C:\Users\TS\.conda\envs\HydroGelNet\python.exe" escalate.py             # PERF-GATE：结果够不够格写论文
```

`paths.py` 会打印 `DEVICE`（cuda/cpu）和 `PYTHON_EXE`；看到 `DEVICE = cuda`
就说明在显卡上跑。想强制指定：`SCI_DEVICE=cpu` 或 `SCI_DEVICE=cuda`。

## NO-FAIL 规则

主模型必须显著打赢最强基线（PERF-GATE 五项检查全过）才允许写论文。
没过就按 `audit/PERF_GATE.md` 里给出的升级阶梯（L0→L8）继续做，
不许交"诚实但失败"的报告。

## Layout

| Path | Contents |
|------|----------|
| `C:/Users/TS/WorkBuddy/HydroGelNet/code` | all Python scripts |
| `C:/Users/TS/WorkBuddy/HydroGelNet/data/raw` | untouched downloads |
| `C:/Users/TS/WorkBuddy/HydroGelNet/data/processed/dataset.npz` | internal cohort |
| `C:/Users/TS/WorkBuddy/HydroGelNet/data/external/dataset_external.npz` | independent cohort |
| `C:/Users/TS/WorkBuddy/HydroGelNet/results` | metrics, predictions, ablation, statistics |
| `C:/Users/TS/WorkBuddy/HydroGelNet/figures` | publication figures (PNG + PDF, 600 dpi) |
| `C:/Users/TS/WorkBuddy/HydroGelNet/tables` | publication tables (CSV + LaTeX) |
| `C:/Users/TS/WorkBuddy/HydroGelNet/paper` | manuscript source and final PDF |
| `C:/Users/TS/WorkBuddy/HydroGelNet/audit` | Gate-1 and Gate-2 audit reports |
