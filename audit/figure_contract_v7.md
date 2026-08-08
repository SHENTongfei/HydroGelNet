# SIMPLEX Fig3-8 图件契约（nature-figure 五要点，v7 重画前定稿）

> 依据 nature-figure static/core/contract.md 五要点契约 + stance（NMI pastel 低饱和、
> hero panel 原则、白色背景、直接标签优先、一个 figure 一个 restrained 调色板）。
> 后端：python（matplotlib，项目既定，无需再问）。数据源：results/ 真实文件。

## 图件契约逐条

### Fig3 数据集特征（fig:dataset）
- 核心结论：内部(316)与前瞻(25)队列靶标范围重叠但组成不同，前瞻验证是决定性测试
- 证据链：9 面板每格唯一证据——(A)队列规模 (B)靶标分布 (C)缺失率 (D)特征相关结构 (E)原始特征空间 (F)条件组成 (G)KS 协变量偏移 (H)组大小分布 (I)靶标范围重叠
- 原型：quantitative grid（3×3）
- hero panel：B（靶标分布，双队列对比）
- 配色：内部=蓝系 / 前瞻=橙系，2 族即可

### Fig4 内部 CV（fig:cv）
- 核心结论：5 折×5 seed 分组 CV 内部 R² 与最强树集成统计平手（L5 数字待刷新）
- 证据链：(A)每折 R² (B)OOF 预测-实测 (C)残差-拟合 (D)误差分布 (E)学习曲线 (F)指标热图 (G)hexbin 密度 (H)seed 稳定性箱线 (I)逐 target R² 斜率
- 原型：quantitative grid（3×3）
- hero panel：A（每折 R² 分布，SIMPLEX 高亮）
- 配色：SIMPLEX=红系 accent，基线=灰/蓝

### Fig5 基准对比（fig:bench）
- 核心结论：内部匹配最强树集成 + 前瞻 R²/Spearman 点估计最高（L5 数字待刷新）
- 证据链：(A)内部 R² 均值±95%CI (B)配对逐折分数 (C)Top-20 筛选精度 (D)ΔR² Holm 显著性 (E)跨折排名 (F)模型质量图 (G)cluster bootstrap CI (H)permutation 检验 (I)critical-difference 排名图
- 原型：quantitative grid（3×3）
- hero panel：A（均值±CI 排序条）
- 配色：SIMPLEX 红 accent 高亮，基线蓝系

### Fig6 前瞻验证（fig:ext）
- 核心结论：25 配方前瞻队列 SIMPLEX 外部 R² 点估计最高 + Spearman 最高 + Top-k 达队列上限（L5 数字待刷新）
- 证据链：(A)预测-实测 (B)Bland-Altman (C)Top-k 恢复(头条) (D)校准 (E)泛化差距 内→外 (F)预测排名分位误差 (G)外部基准对比 (H)外部残差分布 (I)top-50% ROC
- 原型：quantitative grid（3×3）
- hero panel：C（Top-k 恢复曲线，筛选应用头条）
- 配色：SIMPLEX 红 accent + 基线蓝

### Fig7 消融（fig:abl）
- 核心结论：多模态融合贡献最大，剪除中性组件，每个保留组件挣得席位
- 证据链：(A)组件贡献瀑布 (B)R² 排序 (C)融合策略 (D)统计贡献 (E)变体误差条 (F)per-variant 效应 (G)剪枝日志决策 (H)边际 vs 交互累积重要性 (I)剪枝汇总
- 原型：quantitative grid（3×3）
- hero panel：A（瀑布图）
- 配色：正贡献=蓝，中性=灰，负=红

### Fig8 解释与标志物（fig:interp）
- 核心结论：疏水-芳香 BA×PEA 交互主导预测（0.143, p≈1.3e-54），ATAC 阳离子第 2，HEA/CBEA 负向；Fan 2019 同单体实证闭环
- 证据链：(A)Top 置换重要性(前3红) (B)stability 频率 (C)CLS 注意力归因 (D)条件注意力 (E)latent 按靶标着色 (F)latent 按条件着色 (G)Top5 偏依赖 (H)标志物 volcano (I)组成规则符号
- 原型：quantitative grid（3×3）
- hero panel：A（置换重要性条，BA×PEA 红 accent）
- 配色：正=红，负=绿（方向线索），中性=灰

## 导出契约（每张）
- 尺寸：89mm 单栏 / 183mm 双栏宽度对应
- 格式：PNG(≥300dpi) + PDF(可编辑文字) 双交付
- 字体：Helvetica/Arial 系 sans-serif，全部可编辑
- 每张图 footer 注 n、误差条定义、统计显著性标注
- 无静默采样、无彩虹色、无图内文字重叠（figure-visual-qa 预检 14 项）
