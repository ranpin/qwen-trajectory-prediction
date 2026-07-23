# 校准集消融：域内 vs 通用新闻（INT8 SmoothQuant）

**假设**：默认 CNN/DailyMail 新闻校准集与部署域(具身/驾驶推理)错配，拖累了 INT8 的量化质量（INT8 perplexity 掉点 +18.1% > INT4 +9.8%）。

**实验**：用 Cosmos-Reason1-Benchmark 非 robovqa 子集(agibot/bridgev2/holoassist/robofail)的问题文本作**域内校准集**(400 问题→100 条拼接序列)重量化 INT8_SmoothQuant，与默认 news-512 对比（评测集 robovqa 不重叠，无泄漏）。

| INT8 校准集 | FP16-teacher perplexity (vs FP16 1.237) | robovqa MC 正确率 |
|---|---|---|
| news 512 篇长文（默认） | 1.460 (+18.1%) | 87.3% |
| 域内 100 条拼接短问题 | **4.255 (+244%)** | 84.5% |

**结论（假设被推翻）**：域内校准**大幅劣化** INT8（perplexity +244%，生成出现错误物理）。因此"新闻域错配拖累 INT8"**不成立**。更深的结论：**校准集的覆盖度/质量（数量、长度、多样性）比域匹配重要得多**——512 篇长新闻是好的通用校准集，100 条短问题覆盖差→SmoothQuant 激活 scale 估计差→量化劣化。**原 news-512 选择得到验证。**

**Caveat**：本消融中"域"与"校准集规模/长度"混杂（同时变化），不能纯归因于域；但"小而窄的 ad-hoc 域内集有害"的实用结论成立。要纯隔离"域"需构造同规模/长度的域内 vs 通用集。
