# Confirmed Contribution — SIMPLEX v7（paperspine Contribution-First）

> 编制：主理人章成文 2026-08-07 20:20 ｜ 依据 third_party/paperspine/references_contribution.md
> 数字基准：L5v2 跑完后刷新；当前用已验证的继承配置历史值（0.7925 / 0.71）

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | 在 316 配方训练 / 25 模型发现前瞻配方的**筛选应用**上，SIMPLEX——一种带交互自注意力的双模态残差网络——取得了所有同等调参基线中最高的前瞻绝对精度（R² 点估计 0.71）与最高的排序相关（Spearman 0.87），同时是唯一能从组成空间挖出**可独立文献佐证的粘附机理标志物**（疏水-芳香 BA×PEA 交互，Fan 2019 同单体实证闭环）的模型。 |
| Contribution type | new method + new empirical finding（新方法 + 新实证发现；筛选应用范式） |
| One-sentence reviewer payoff | 给定 k 个合成预算，SIMPLEX 从**前所未见、模型自行发现的配方**中恢复最强粘附候选的排序能力优于所有同等调参基线，且其挖掘的组成规律与已发表的湿粘附化学机理直接对应——这是深度模型在数据稀缺软材料领域"外推筛选 + 可解释发现"的双重价值。 |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | 生物混合界面用功能水凝胶设计依赖试错实验：每个候选配方必须合成并力学表征才能评估水下粘附，成本极高。 |
| Specific gap | 数据稀缺（316 配方）下，尚无模型能证明其排序能力可迁移到**模型发现的前瞻配方**（未被模型选择触碰的真正留出队列）。 |
| Concrete challenge | 前瞻队列 n=25 太小，指标估计带大不确定性（bootstrap CI 半宽 ≈0.2）；且组成空间外推（composition-space extrapolation）使所有模型的绝对误差指标天然恶化。 |
| Why prior work leaves it unresolved | 树集成（RF 0.81）内部可与深度模型匹敌但前瞻迁移差（0.81→0.56）；浅层模型（Ridge/SVR）外部点估计次之且给不出组成规律；现有深度表格方法在小样本下过拟合，且未在真正的前瞻协议下评估。 |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | 双模态显式编码（6 单体分数 + 15 两两交互项）使组成协同可直接学习；交互自注意力 + 残差细化；Mixup/SWA/EMA/域约束小数据正则化带；前瞻队列在全部超参冻结后一次性评估。 |
| Evidence required | ① 前瞻队列上 vs 最强基线的点估计与 CI ② 排序指标（Spearman）③ Top-k 筛选精度 ④ 泛化差距（内→外）⑤ 组成规律的独立文献佐证 ⑥ 内部统计诚实性（tie 如实写） |
| Evidence available | ① 前瞻 R² 0.71 vs RF 0.56（点估计 +0.15）② Spearman 0.87 vs 0.84 ③ Top-10 1.00 / TopK30 p=0.998 ④ 泛化差距 0.79→0.71 vs RF 0.81→0.56（SIMPLEX 差距更小）⑤ BA×PEA=Fan 2019 Nat. Commun. 同单体实证闭环 + 5 机制真实文献 ⑥ 内部 R² 0.79 vs RF 0.81（Δ−0.017, Holm p=1.0，如实写 tie） |
| Evidence missing | ① 前瞻 CI [0.46,0.86] 与基线重叠（不可统计分离）② 独立实验室队列（当前同实验室 SMBO 后期）③ 内部 R² 未超 RF（统计平手）→ 贡献声明必须限定在"决定性筛选维度 + 可解释发现"，不宣称全维度超越 |

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | 前瞻点估计最高（R² 0.71, Spearman 0.87）；泛化差距最小；Top-10 满恢复；TopK30 显著（P=0.998）；唯一挖出可文献佐证组成规律（BA×PEA 疏水-芳香 = Fan 2019 实证）；内部与最强树集成统计平手（如实写 within statistical noise） |
| Claims to soften or avoid | "全面最优/所有基线最优"（内部 tie）；"显著优于"（CI 重叠→"点估计最高"）；"因果机制"（→"统计关联，假设生成"）；"独立队列"（→"模型发现的前瞻队列"） |
| Novelty risk | "DL 在表格数据上不如树模型"的普遍质疑 → 回答：本论文的贡献不在内部 R² 超越树模型，而在前瞻排序迁移 + 可解释组成规律（树模型给不出） |
| Significance risk | "n=25 太小，点估计无显著性" → 回答：TopK30 配对 bootstrap P=0.998 提供显著证据；且筛选应用的价值在于排序，Spearman/Top-k 是部署时真正关心的指标（skill L7 明文支持的任务重框） |
