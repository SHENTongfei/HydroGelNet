# IDEA CARD — HydroGelNet (HGN)

## RESEARCH_SPEC
- **领域**: 软材料（水凝胶）力学性能预测
- **子方向**: composition→property（配方组成→水下粘附强度）
- **任务类型**: 回归（单目标 Glass (kPa)_max；多目标扩展）
- **模态数**: 2（6 单体摩尔分数 + 15 两两交互特征）
- **目标刊物**: 中上（IF 3-8，Frontiers in Materials / Soft Matter / npj）
- **核心卖点**: 训练在低性能配方空间（df_180, mean=47kPa），外推预测 SMBO 引导的高性能配方空间（161 新配方, mean=154kPa）——组合外推挑战

## 模型名（候选）
- **HGN = HydroGel-Net**（待查重）
- 展开: H(ydrogel) G(raph-enhanced) N(etwork) 或 H(ydrogel) G(eneration) N(etwork)
- 备选: GELD / SIMPLEX / COHESION

## 数据
- sheng-hu/hydrogels (MIT, Nature 2025, doi:10.1038/s41586-025-09269-4)
- internal: df_180 (180 配方, y mean=46.9, max=146.6)
- external: df_341 − df_180 (161 SMBO 配方, y mean=154.2, max=353.3)
- 特征: 6 单体摩尔分数 (Nucleophilic-HEA / Hydrophobic-BA / Acidic-CBEA / Cationic-ATAC / Aromatic-PEA / Amide-AAm)
- 目标: Glass (kPa)_max（水下玻璃粘附强度）

## 创新点
1. **组合外推范式**: 首次把"时间外推"（训练早期低性能空间→测试 SMBO 后期高性能空间）作为水凝胶 ML 的评估协议
2. **双模态组成编码**: 单体分数 + 显式两两协同特征（多项式展开），配合注意力捕捉高阶交互
3. **炫技组合**: Transformer + 稀疏注意力 + FiLM + ModalityGate + 对比预训练 + Mixup + SWA + SAM + EMA + 不确定性加权 + 域约束
4. **红牌规避**: composition→property（非曲线提取）、Ridge 外推 R²=-1.18（天花板未压缩）、外部 n=161

## 风险
- 6 维输入低维，过拟合风险高 → 需强正则 + 对比预训练
- 外推极难（RF 都崩 R²=-0.35）→ 需要域约束/单调性先验帮忙
