[← 上一课](../L10-直接偏好优化DPO/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L12-医疗数据集详解/README.md)

---

# L11 ORPO 与 GRPO 前沿方法

> **一句话精髓**：*「更高效：让对齐更接近『一步到位』或『更省显存地强化学习』。」*

---

## 写给「刚学完 L08–L10」的你

你已经走过 **偏好对齐** 的主干道：

- **RM + RLHF（PPO/RLOO）**：显式奖励 + 在线优化，强但重。  
- **DPO**：去掉显式 RM，把偏好监督化，轻且稳，但 **极吃数据质量**。

本课介绍两条「进一步压缩流程或显存」的路线：

1. **ORPO**：把 **SFT 与偏好优化** 更紧密地绑进 **单体目标**，叙事上 **弱化单独参考模型 checkpoint 的依赖**（实现细节以 TRL 为准）。  
2. **GRPO**：在 **可验证奖励** 场景，用 **同一 prompt 的 k 个样本组内对比** 构造优势，**砍掉 Critic**，显著降低显存与工程复杂度。

读完本课，你应该能 **口述 ORPO/GRPO 的核心创新**、**对照 MedicalGPT 脚本**、并背下 **算法选型决策树**（面试高频）。

---

## 本课你将学会什么

1. 解释 **ORPO** 的核心思想：**不需要单独参考模型** 的叙事、**SFT+对齐一步**、**优势比（odds ratio）** 直觉。  
2. 说明 ORPO 如何 **缓解灾难性遗忘**（在同一目标里保留语言建模分量）。  
3. 对照 **`orpo_training.py`**：`DPOTrainer` + `loss_type="orpo"`、`orpo_beta`。  
4. 解释 **GRPO**：DeepSeek 等推动的 **组内相对策略优化**；**组内对比替代价值网络**。  
5. 说明 **为何显存可降 30–50%（经验区间）**：少一整份 Critic 与其优化器状态（**以实测为准**）。  
6. 描述 GRPO **工作原理**：同一输入生成 k 个候选 → 组内归一化得优势 → 策略梯度更新。  
7. 掌握 **k 值选择经验**（从 4 smoke test 到 8/16）。  
8. 说明 GRPO **特别适合数学推理与代码生成** 的原因（可验证奖励）。  
9. 了解 **MedicalGPT 对 GRPO 的支持**（`grpo_training.py`、`run_grpo.sh`）。  
10. **DAPO** 一句话定位。  
11. 背 **算法选择决策树**（资源/数据/任务形状）。  
12. 画 **多种对齐方法对比 ASCII**。  
13. 回答：**GRPO 与 PPO 核心差异？为何越来越多模型选 GRPO？**

---

## 目录

1. [总览：对齐方法「越来越省」的趋势](#1-总览对齐方法越来越省的趋势)
2. [ORPO：Odds Ratio Preference Optimization](#2-orpoodd-ratio-preference-optimization)
3. [MedicalGPT：`orpo_training.py`](#3-medicalgptorpo_trainingpy)
4. [`run_orpo.sh` 参数导读](#4-run_orposh-参数导读)
5. [GRPO：Group Relative Policy Optimization](#5-grpogroup-relative-policy-optimization)
6. [MedicalGPT 对 GRPO 的支持](#6-medicalgpt-对-grpo-的支持)
7. [k 值怎么选（经验谈）](#7-k-值怎么选经验谈)
8. [DAPO 简介](#8-dapo-简介)
9. [算法选择决策树（面试必背）](#9-算法选择决策树面试必背)
10. [ASCII：各种对齐方法对比](#10-ascii各种对齐方法对比)
11. [面试高频题](#11-面试高频题)
12. [小结与自测](#12-小结与自测)

---

## 1. 总览：对齐方法「越来越省」的趋势

把前几课串成一条 **工程负担** 轴：

```
PPO-RLHF (多模型、在线采样、重)
    → DPO (去掉 RM+RL 内环，偏监督化)
    → ORPO (强调与 SFT 合并、弱化 ref 叙事)
    → GRPO (RL 家族里砍掉 Critic，用组内基线省显存)
```

### 1.1 类比三连

- **DPO**：错题本直接改作文。  
- **ORPO**：边练字帖（语言能力）边改语病（偏好），减少「对照字帖」式 ref 依赖的叙事。  
- **GRPO**：同一道题交 **k 份作业**，小组内 **相对排名** 定奖惩，不再雇「全年级估值老师」（Critic）。

---

## 2. ORPO：Odds Ratio Preference Optimization

论文：[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)（题目以官方为准）。

### 2.1 核心思想（口语）

- 经典 DPO 需要 **参考策略 \(\pi_{\text{ref}}\)**（常是 SFT 拷贝）构造 log 比率。  
- ORPO 试图在 **单一阶段** 把 **负对数似然（SFT/LM 项）** 与 **偏好 odds ratio 项** 结合，从而 **减弱对单独参考模型 checkpoint 的依赖**（实现路径依 TRL 版本而定）。

### 2.2 「SFT + 对齐一步完成」意味着什么？

工程上可能减少「先 SFT 再 DPO」的 **显式阶段切换**，适合：

- 想 **缩短 Pipeline**；  
- 从中等质量基座出发，**同时拉能力曲线与对齐曲线**。

**注意：** 「一步」不是魔法——数据仍要同时支撑 **示范学习** 与 **偏好对比**；医疗场景要非常小心 **安全偏好** 与 **事实性** 的权重平衡。

### 2.3 优势比（Odds Ratio）直觉

回顾 odds：\(\text{odds}(p)=\frac{p}{1-p}\)。  
偏好建模里用 **odds ratio** 连接「模型对赢家 vs 输家的相对倾向」，与 **logit 差** 有亲缘关系，用于 **强调 margin** 与稳定训练（细节以论文为准）。

### 2.4 缓解灾难性遗忘（Catastrophic Forgetting）

**痛点：** 强偏好优化后，模型在 **通用能力** 或 **SFT 领域事实** 上退步。

**ORPO 叙事：** 在同一目标里保留 **LM 的 NLL / SFT 分量**，使优化 **不要只围着偏好转**，从而 **缓和** 只训偏好时的能力坍塌（程度依赖超参与数据混合）。

### 2.5 与 DPO 的边界（面试说法）

> ORPO 属于 **偏好优化** 家族，强调 **单体（monolithic）目标** 与 **参考模型弱化**；DPO 是「显式 ref + BT 推导」的经典基线。二者 **不是简单谁永远更好**，看数据与阶段划分。

---

## 3. MedicalGPT：`orpo_training.py`

[MedicalGPT](https://github.com/shibing624/MedicalGPT) 中 ORPO 常通过 **TRL 的 `DPOTrainer` + `loss_type="orpo"`** 实现。

### 3.1 数据管道

与 `dpo_training.py` **同构**：

- `return_prompt_and_responses` 从 `system` / `history` / `question` / `response_chosen` / `response_rejected` 构造 `prompt`, `chosen`, `rejected`。

### 3.2 训练配置要点

- `DPOConfig(..., loss_type="orpo", beta=args.orpo_beta, ...)`  
- `orpo_beta` 在脚本注释中常解释为 **ORPO 目标里 SFT/对齐权衡**（参数语义以仓库与 TRL 版本为准）。  
- `DPOTrainer(...)`：**未传入 `ref_model`** 的路径与「ORPO 不依赖单独参考模型」的实现一致（数学在 TRL 内部完成）。

### 3.3 实操提醒

- 关注 **长度过滤**、`full_max_length`、`gradient_checkpointing`。  
- ORPO 与 DPO **共用数据格式**，便于 **A/B**。

### 3.4 目标「两项相加」直觉（非严格符号）

```text
L_ORPO ≈ L_SFT(依赖实现：常在 chosen 上保留 LM 损失) + λ · L_odds-preference(chosen vs rejected)
```

**读法：** 左边拉住 **语言能力**，右边拉住 **相对偏好**；`orpo_beta` 类似 **缰绳松紧**。医疗场景往往 **不能让偏好项完全压死语言项**。

---

## 4. `run_orpo.sh` 参数导读

与 `run_dpo.sh` 接近，额外关注：

| 参数 | 含义 |
|------|------|
| `--orpo_beta` | SFT/对齐权衡（与 TRL 内部 `beta` 联动，以版本为准） |
| `--model_name_or_path` | 起点常为 Instruct / 经 SFT 的模型 |
| `--train_file_dir ./data/reward` | 与 RM/DPO **同目录习惯** |
| `--per_device_train_batch_size` / `--gradient_accumulation_steps` | 按显存调 |
| `--max_steps` | 示例可能偏短，仅供 smoke test |

**对比记忆：** `run_dpo.sh` 无 `orpo_beta`；ORPO **必须先理解 beta 再调**，否则易出现「只对齐不长能力」或相反。

---

## 5. GRPO：Group Relative Policy Optimization

**来源：** DeepSeek 等在 **推理强化学习** 场景推广 **组内相对奖励**；公开讨论常与 **R1 类训练** 相关。GRPO 强调：**用一组样本上的相对表现构造优势，替代显式价值网络（Critic）**。

### 5.1 核心创新：组内对比替代价值网络

经典 PPO-RLHF：

- Critic \(V_\psi(s)\) 估计状态价值，降低方差。

GRPO（直觉）：

- 对 **同一 prompt** 采样 **k 个完成**；  
- 用 **组内统计量**（如减均值、除标准差）定义 **相对优势**；  
- **不再维护大型 Critic**。

**白话：** 同一道题小组 k 人交卷，以 **组平均为基准** 看谁更好，而不是请「全年估值专家」给每题估绝对价值。

### 5.2 显存为何常能降 30–50%（经验区间）？

粗算（概念）：少加载 **一整份与 policy 同量级的价值模型** 及其优化器状态。  
**面试必补一句：** 比例依赖 **模型规模、实现、是否 vLLM、并行策略**；**以实测为准**。

### 5.3 工作原理（步骤化）

```text
1) 采样 prompt x
2) 用当前 policy 生成 k 个完成 {y_1,...,y_k}
3) 对每个 y_i 计算奖励 R_i（规则 / 可验证 / 模型奖励）
4) 组内归一化：Ã_i = (R_i - mean(R)) / (std(R) + ε)  （形式因实现而异）
5) 策略梯度类更新，使高于组平均的完成概率上升
```

**关键：** 优势是 **相对的**，弱化绝对标定难度，适合 **对错分明** 的任务。

### 5.4 特别适合数学推理与代码生成

- 常有 **可验证答案**（单测、数值比对），奖励噪声低。  
- 组内 k 个样本覆盖多种错误，相对排序更稳。

### 5.5 与 PPO 的核心差异（浓缩表）

| 维度 | 经典 PPO-RLHF | GRPO |
|------|----------------|------|
| Critic | 通常需要 | **不需要（组内基线）** |
| 优势来源 | GAE + Value | **组内归一化奖励** |
| 奖励 | RM 标量常见 | 常配合 **可验证/规则** 奖励 |
| 典型场景 | 通用聊天对齐 | **推理、代码、可验证任务** |

### 5.6 白板伪代码

```text
for epoch:
  for batch of prompts X:
    for each x in X:
      samples = [ rollout_k_times(π_θ, x) for _ in range(k) ]
      rewards = [ R(x, y_i) for y_i in samples ]
      advs = normalize_within_group(rewards)
      loss += policy_gradient_term(π_θ, samples, advs)
    update θ
```

**对照 PPO：** 把「Critic 估计的 A」换成「组内归一化后的 A」；**k** 显式控制采样次数。

---

## 6. MedicalGPT 对 GRPO 的支持

仓库提供 **`grpo_training.py`** 与 **`run_grpo.sh`**（以主分支为准）。

### 6.1 脚本做什么？

- 使用 `trl.GRPOConfig`, `trl.GRPOTrainer`, `trl.ModelConfig`, `trl.TrlParser` 等（以版本为准）。  
- 可从 Hub 加载如 **GSM8K** 类数据集，或 `--train_file_dir` 指向本地 JSON。  
- 将样本映射为 **chat prompt**（常含固定 `SYSTEM_PROMPT` 引导推理格式）+ `answer` 字段供奖励函数使用。

### 6.2 奖励函数（示例实现）

- **`accuracy_reward`**：数学类解析 `####`；否则 LaTeX 解析 + verify（依赖 `math_verify` 等）。  
- **`format_reward`**：正则检查 **标签格式**（推理/答案分节），鼓励可解析输出。

**医疗迁移：** 需要 **可验证奖励**（结构化 JSON、选择题选项、知识库匹配）。**开放域问诊** 很难自动奖罚，GRPO **不是银弹**。

### 6.3 `run_grpo.sh` 参数导读

| 参数 | 含义 |
|------|------|
| `torchrun --nproc_per_node 2` | 分布式数据并行 |
| `--model_name_or_path` | 基座 / Instruct |
| `--num_generations 4` | **k**：每组采样条数（常见 4–16） |
| `--per_device_train_batch_size` | 与生成并行度强相关，过大易 OOM |
| `--max_completion_length` | 生成长度上限 |
| `--beta` | KL/正则类系数（以 TRL 文档为准） |
| `--learning_rate` | GRPO 常 **偏小**（示例或到 `5e-7` 量级） |
| `--use_peft` + LoRA | 降可训练参数 |
| `--use_vllm False` | 环境支持时可开 vLLM 加速生成 |

### 6.4 数据路径补充

- **Hub 模式：** `dataset_name` 如 `openai/gsm8k`，`subset_name`、`dataset_splits` 控制子集。  
- **本地模式：** `--train_file_dir` → `load_dataset("json", data_dir=..., split="train")`。  
- **划分：** `train_test_split(test_size=0.1)`。  
- **字段映射：** `question` / `answer` → messages（system + user），与 **格式奖励** 强耦合。

### 6.5 医疗场景迁移检查表

1. **奖励是否可自动判定？** 否则组内优势学噪声。  
2. **是否 shortcut？** 误把关键词当对会诱发投机。  
3. **安全拒答怎么奖？** 需单独模板奖励。  
4. **合规：** 自动奖励不能替代临床责任。

---

## 7. k 值怎么选（经验谈）

| k 较小（2–4） | k 较大（8–16） |
|---------------|----------------|
| 显存/时间压力小 | 组内基线 **更稳** |
| 方差估计更噪 | 训练慢、显存暴涨 |

**经验：** 从 **k=4** smoke test；稳定后试 **8**；奖励极可靠且算力足再升。  
**监控：** 每组奖励全相同（学不到相对序）与 OOM。

---

## 8. DAPO 简介

**DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）** 等工作关注 **GRPO/PPO 类训练中的动态采样与解耦 clip**，目标常是 **提高长链推理训练的有效梯度比例、减少无效样本**。

**面试一句话：**  
> DAPO 属于 **RL 推理对齐** 生态里对 **采样与 clip 策略** 的改进方向，与 GRPO **同一条「可验证奖励 + RL」延长线**；细节以论文/技术报告为准。

**医疗侧：** 若缺乏可验证奖励，DAPO/GRPO 优先级通常 **低于** DPO/ORPO + 强偏好数据。

---

## 9. 算法选择决策树（面试必背）

### 9.1 一表对比（背诵版）

| 方法 | 典型数据 | 在线生成 | 参考模型 | Critic | 第一印象 |
|------|-----------|----------|----------|--------|-----------|
| PPO-RLHF | prompt + rollout | 是 | 要 | 要 | 重、经典 |
| DPO | 偏好对 | 否 | 要* | 否 | 性价比 |
| ORPO | 偏好对 + LM 项 | 否 | 弱化 | 否 | 少阶段 |
| GRPO | prompt + 可验证标签 | 是（k 路） | 依实现 | 否 | 推理/代码 |

\* PEFT 下 ref 可能隐式处理。

### 9.2 决策树（刻意简化）

```
开始
  |
  +-- 资源充足 + 要走经典叙事/复杂奖励？
  |       └─ 倾向：PPO-RLHF（或企业内同类 RL 栈）
  |
  +-- 偏好数据高质量 + 快速迭代 + 显存有限？
  |       └─ 倾向：DPO（L10）
  |
  +-- 客观可验证任务 + 想省 Critic 显存？
  |       └─ 倾向：GRPO（本课）
  |
  +-- 希望 SFT 与对齐更少手工切分 + 接受 ORPO 调参？
          └─ 倾向：ORPO（本课）
```

### 9.3 用户要求速记（必背四句）

- **资源充足 → PPO**  
- **数据高质 + 资源有限 → DPO**  
- **客观任务 + 要效率 → GRPO**  
- **SFT 和对齐一步完成 → ORPO**

---

## 10. ASCII：各种对齐方法对比

```
           方法            参考模型        在线采样       Critic/价值网
        ------------    ------------    ------------    --------------
        PPO-RLHF           通常要            要               通常要
        DPO                要*             不要             不要
        ORPO             弱化/无(叙事)      不要             不要
        GRPO             依实现/β项        要(k 完成)       不要

        * PEFT 下 ref 可能隐式处理
```

```
                数据与任务形状决定上限
                           |
        +------------------+------------------+
        |                  |                  |
   开放域偏好            可验证答案          混合目标
   (聊天/问诊)           (数学/代码)        (能力+偏好)
        |                  |                  |
      DPO/ORPO            GRPO              ORPO/多阶段
```

---

## 11. 面试高频题

**Q1：GRPO 和 PPO 的核心差异？**

> GRPO 用 **同一 prompt 的 k 个样本** 做 **组内归一化** 得优势，**不需要 Critic**；经典 PPO-RLHF 常用 **价值网络 + GAE** 并配合 RM。GRPO 更贴合 **可验证奖励** 与 **推理链**。

**Q2：为什么越来越多模型选 GRPO（或组内 RL）？**

> **显存与工程复杂度下降**（无 Critic）；**可验证任务** 奖励清晰；长思考链里 **相对排序** 常比绝对标量稳；TRL 等基础设施成熟。**开放域纯聊天** 仍要审慎。

**Q3：ORPO 还需要偏好数据吗？**

> 需要。ORPO 解决的是 **目标整合与 ref 依赖**，不是「凭空不需要偏好」。

**Q4：ORPO vs DPO 怎么选？**

> 想 **少阶段、从中等基座拉齐** 可试 ORPO；成熟 SFT + 熟悉 DPO 则 **DPO 更稳**。最终以 **离线评测 + 安全红队** 为准。

**Q5：MedicalGPT 里 ORPO 怎么实现？**

> `DPOTrainer` + `loss_type="orpo"` + `orpo_beta`；数据管道与 DPO 相同。

---

## 12. 小结与自测

### 12.1 背四句

1. ORPO：**单体目标**，SFT 与偏好 **绑在一起训**，弱化 ref。  
2. GRPO：**k 样本组内相对优势**，**砍掉 Critic**，省显存，适合 **可验证推理**。  
3. DAPO：**动态采样 / decoupled clip** 一类 **工程增强**。  
4. 选型：**算力+经典 RL → PPO；好数据+省事先 DPO；客观题+省显存 GRPO；少阶段 ORPO**。

### 12.2 自测

1. 画出 GRPO 一组 k=4 时的奖励流。  
2. 解释 GRPO 在开放域问诊为何可能吃亏。  
3. 打开 `run_grpo.sh`，找到 `num_generations` 与 `learning_rate`。

### 12.3 参考

- [ORPO 论文](https://arxiv.org/abs/2403.07691)  
- DeepSeek 相关技术传播材料（GRPO / R1，注意甄别二手解读）  
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)  
- [MedicalGPT](https://github.com/shibing624/MedicalGPT)

---

**课程段收尾：** L08–L11 已覆盖 **RM → RLHF/PPO → DPO → ORPO/GRPO** 的「偏好对齐光谱」。后续 L12 起进入 **数据与工程深水区**，请随身携带本课 **选型表**。

---

*前沿论文与 TRL API 迭代快；以本地 `pip show trl` 版本与仓库 commit 交叉验证。*
