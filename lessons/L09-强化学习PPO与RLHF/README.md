[← 上一课](../L08-奖励模型RM/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L10-直接偏好优化DPO/README.md)

---

# L09 强化学习 PPO 与 RLHF

> **一句话精髓**：*「用『老师』的评分不断改进『学生』的回答。」*

---

## 写给「刚学完 L08」的你

在 L08 里，你训练了一个 **奖励模型（RM）**：它能把「提示 + 回答」映射成一个 **标量分数**，用来近似人类偏好。  
但 RM 本身 **不会生成文本**；真正面对用户、逐 token「做决定」的是 **策略语言模型（Policy LM）**。

**本课核心问题：** 如何利用 \(r_\phi(x,y)\) 去 **更新** 这个会生成的模型？

答案是 **强化学习（RL）** 框架下的 **RLHF（Reinforcement Learning from Human Feedback）**。面试与论文叙述里，**PPO（Proximal Policy Optimization）** 是最常被拿来当「标准参照」的算法；工程上你也会看到 **RLOO** 等变体——**MedicalGPT 的 `ppo_training.py` 实际实现可能与经典四模型 PPO 不完全一致**，本课会 **先讲清经典**（你面试靠它），再 **对齐你本地仓库**（你跑代码靠它）。

---

## 本课你将学会什么

1. 用 **Agent / Environment / Action / Reward** 描述「语言模型 + RM」在 RL 里分别是谁。  
2. 准确解释 **RLHF** 的含义与动机。  
3. 画出并讲解 **SFT → RM → PPO** 三步流水线，每步解决什么问题。  
4. 理解 **PPO** 的动机：限制策略更新幅度，避免重要性采样失效。  
5. 掌握 **clip 机制**：为什么 PPO 强调「不要走太远」。  
6. 背出 **经典 PPO-RLHF 四模型**：Policy / Reference / Reward / Value（Critic）。  
7. 解释 **KL 散度惩罚**：防止策略偏移到 RM 不可靠区域（reward hacking / OOD）。  
8. 对照 **MedicalGPT `ppo_training.py`**：数据、`RLOOTrainer`、奖励模型加载方式。  
9. 逐项理解 **`run_ppo.sh`** 中与显存、生成长度、学习率相关的参数。  
10. 说出 **PPO / RLHF 的工程缺点**：不稳定、显存、调试难。  
11. 能画 **RLHF 完整流程 ASCII 图**。  
12. 回答面试题：**四模型分工？为何 KL？PPO 与 DPO 区别？**

---

## 目录

1. [类比：学生、老师与缰绳](#1-类比学生老师与缰绳)
2. [强化学习基础：Agent / Environment / Action / Reward](#2-强化学习基础agent--environment--action--reward)
3. [什么是 RLHF](#3-什么是-rlhf)
4. [RLHF 三步走：SFT → RM → PPO](#4-rlhf-三步走sft--rm--ppo)
5. [把语言模型放进 RL 框架](#5-把语言模型放进-rl-框架)
6. [PPO 算法详解（直觉 + 公式骨架）](#6-ppo-算法详解直觉--公式骨架)
7. [Clip：核心思想是「不要走太远」](#7-clip核心思想是不要走太远)
8. [经典设置：PPO 需要的四个模型](#8-经典设置ppo-需要的四个模型)
9. [KL 散度惩罚：防止模型偏离太远](#9-kl-散度惩罚防止模型偏离太远)
10. [MedicalGPT：`ppo_training.py` 核心逻辑](#10-medicalgptppo_trainingpy-核心逻辑)
11. [关键参数与训练命令：`run_ppo.sh` 解析](#11-关键参数与训练命令run_pposh-解析)
12. [PPO 的缺点与 RLHF 工程挑战](#12-ppo-的缺点与-rlhf-工程挑战)
13. [ASCII：RLHF 完整训练流程](#13-asciirlhf-完整训练流程)
14. [面试高频题与白板伪代码](#14-面试高频题与白板伪代码)
15. [小结与自测](#15-小结与自测)

---

## 1. 类比：学生、老师与缰绳

- **学生（Policy）**：正在训练的 **生成模型** \(\pi_\theta\)，会写作文（回答用户）。  
- **环境**：不是物理世界，而是「给定 prompt，按策略采样 token，直到结束符」这条 ** rollout 过程**。  
- **动作（Action）**：每一步在词表中选一个 token（或子词）。  
- **奖励（Reward）**：常见是 **序列级**——整段生成完后，用 RM 打分；也常加上 **KL 惩罚** 等 shaping。  
- **老师（RM）**：L08 的打分器 \(r_\phi(x,y)\)。  
- **缰绳（Reference + KL）**：通常是 **冻结的 SFT 模型** \(\pi_{\text{ref}}\)，用来约束 \(\pi_\theta\) 不要跑到 **胡言乱语但 RM 误给高分** 的区域。

```
     Prompt 进教室
           |
           v
    ┌──────────────┐        生成完整回答 y
    │  学生 Policy  │ ----------------------+
    └──────────────┘                       |
           ^                                 v
           |                          RM 给分 r(x,y)
           |                          （常 − β·KL）
           +-------- PPO 类更新 θ --------+
```

---

## 2. 强化学习基础：Agent / Environment / Action / Reward

### 2.1 四要素对照表（LM RLHF 典型）

| RL 概念 | 在 LM RLHF 里常对应什么 |
|---------|-------------------------|
| **Agent** | 策略模型 \(\pi_\theta\)（Causal LM） |
| **Environment** | 提供 \(x\)；维护已生成前缀作为状态转移 |
| **Action** | 选下一个 token |
| **Reward** | RM 分数（稀疏，常在序列末）；可叠加 KL 等 |

### 2.2 策略梯度一句话

若目标是最大化期望回报 \(J(\theta)\)，策略梯度思想可写成：

\[
\nabla_\theta J(\theta)\propto \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a|s)\cdot A\big]
\]

其中 \(A\) 是 **优势（Advantage）**：这个动作比「平均水平」好多少。  
**朴素 REINFORCE** 也能训，但方差大；PPO 等算法通过 **裁剪更新**、**价值网络** 等降低不稳定。

### 2.3 小白直觉

若某次生成让奖励变高，就 **增大** 那次采样到的 token 的概率；反之则压小。  
但步长太大时，**新策略** 与 **采样数据时的旧策略** 差太远，梯度方向不可信——这就是 PPO 要解决的痛点之一。

---

## 3. 什么是 RLHF

**RLHF（Reinforcement Learning from Human Feedback）**：  
用 **人类反馈**（常先变成 RM）定义奖励，再用 **强化学习** 微调语言模型，使其在 **高维离散决策（选词）** 上提升该奖励。

**与 SFT 的边界：**

- **SFT**：模仿示范轨迹（行为克隆）。  
- **RLHF**：允许探索生成分布，用 **标量奖励** 拉向人类更喜欢的区域；可处理「没有唯一字符串答案」的对齐目标。

---

## 4. RLHF 三步走：SFT → RM → PPO

| 阶段 | 输入 | 输出 | 解决的问题 |
|------|------|------|------------|
| **SFT** | 高质量指令-回答 | \(\pi_{\text{SFT}}\) | 基本能力、格式、遵循指令 |
| **RM** | 成对偏好数据 | \(r_\phi(x,y)\) | 把「好」变成可优化信号 |
| **PPO（RL）** | prompt 数据 + RM + ref | \(\pi_\theta\) | 在真实生成分布上提升奖励且可控 |

**医疗直觉：** SFT 让模型「像助手那样说话」；RM 学会「哪种说法更安全、更有帮助」；RL 阶段在 **长尾问法** 上继续塑形——但 **医疗合规** 要求你在奖励与 KL 上非常谨慎。

---

## 5. 把语言模型放进 RL 框架

### 5.1 状态与动作

- **状态 \(s_t\)**：prompt + 已生成的前缀。  
- **动作 \(a_t\)**：下一个 token。  
- **策略 \(\pi_\theta(a_t|s_t)\)**：LM 给出的条件分布。

### 5.2 奖励在哪里给？

常见是 **稀疏奖励**：EOS 之后才用 RM 打分。  
也可设计 **过程奖励**（工具调用格式、步骤得分），但系统更复杂。

### 5.3 RM 不可微怎么办？

RM 对 **离散采样路径** 打分；用 **策略梯度类方法** 绕过「端到端反向穿过采样」难题。

---

## 6. PPO 算法详解（直觉 + 公式骨架）

### 6.1 从「大步更新」到「信任域」

若直接用旧数据对新策略做大步梯度更新，**重要性采样比率** 可能爆炸或趋零，估计失真。  
**PPO** 思路：在同一批样本上做多轮更新，但用 **ratio + clip** 把更新限制在「离旧策略不太远」的范围内。

### 6.2 重要性采样比率

记采样用 **旧策略** \(\pi_{\text{old}}\)，当前优化 \(\pi_\theta\)：

\[
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
\]

### 6.3 Clipped Surrogate Objective（骨架）

对每个位置有优势估计 \(\hat{A}_t\)。PPO-Clip 目标（只看结构）：

\[
L^{\text{CLIP}}(\theta)=\mathbb{E}\Big[\min\big(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\big)\Big]
\]

**读法：**

- 若 \(\hat{A}_t>0\)（好动作），无 clip 时会拼命增大 \(r_t\)。clip 上限 **\(1+\epsilon\)** 限制「一次涨太猛」。  
- 若 \(\hat{A}_t<0\)（坏动作），clip 下限 **\(1-\epsilon\)** 限制「一次压太狠」。

### 6.4 完整训练常见的附加项

工程实现里总 loss 还常包含：

- **价值损失**：Critic \(V_\psi\) 拟合回报（MSE）。  
- **熵 bonus**：鼓励探索，减缓过早塌缩到单一语气。

面试答出 **clip + value + entropy** 结构即可；系数因框架而异。

---

## 7. Clip：核心思想是「不要走太远」

### 7.1 地图类比

你用 **旧导航风格** 采了一批路况数据来规划新策略；若你一步把开车风格改成 **与采样时完全不一致**，旧数据上的梯度 **不再可信**。  
Clip 像 **方向盘转角限幅**：每次别拐太急。

### 7.2 与 reward hacking 的关系

没有约束时，策略可能快速学会利用 RM 盲点（奉承、冗长、触发词），拿到 **虚高奖励** 但人类体验变差。Clip + KL 都是 **系统工程上的刹车片**。

---

## 8. 经典设置：PPO 需要的四个模型

> **面试必背**：下面这张表描述 **经典 PPO-RLHF（如 InstructGPT 叙述）**。**MedicalGPT 当前脚本若用 RLOO，则组件数量可能不同**——见第 10 节。

| 模型 | 角色 | 是否训练 | 作用 |
|------|------|----------|------|
| **Policy（策略）** | 当前要优化的 LM | **训练** | 生成回答 |
| **Reference（参考）** | 常为冻结 SFT 快照 | **冻结** | 计算 KL，防止跑太偏 |
| **Reward Model** | L08 训练 | RL 阶段常 **冻结** | 给生成结果打分 |
| **Value / Critic** | 估计状态价值 | **训练** | 降方差，估计优势 |

**记忆口诀：** 一个 **敢改**（policy），一个 **敢打**（RM），一个 **敢拽**（ref+KL），一个 **敢估**（critic）。

---

## 9. KL 散度惩罚：防止模型偏离太远

### 9.1 为什么需要 KL？

RM 只在 **有限偏好数据** 上训练，对 **新分布** 的打分可能 **不可靠**。  
若 policy 自由探索，容易进入 RM **自信但人类觉得很糟** 的区域。

对参考策略 \(\pi_{\text{ref}}\)（常取 SFT）加入惩罚项：

\[
R_{\text{total}}=r_\phi(x,y)-\beta\cdot D_{\text{KL}}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)
\]

\(\beta\) 控制「多听话于 SFT」 vs 「多讨好 RM」。

### 9.2 工程上 KL 怎么估？

可在生成序列的每步近似累加 KL；实现细节依 TRL / DeepSpeed-Chat 等而定。  
**面试答法：** 约束 policy 不要离 SFT 太远，减轻 **分布偏移** 与 **reward hacking**。

### 9.3 医疗场景额外提醒

KL 不是「医学正确性保证」，只是 **分布锚定**。安全仍需 **数据、规则、评测、红队**。

---

## 10. MedicalGPT：`ppo_training.py` 核心逻辑

### 10.1 读源码前的重要说明（避免课码脱节）

在 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 主分支常见实现里，`ppo_training.py` 文件头注释会写明：使用 **TRL 的 `RLOOTrainer`（REINFORCE Leave-One-Out）** 作为 **PPO 的替代**，并说明 **RLOO 不需要单独价值模型或参考模型**（与经典四模型 PPO 不同）。

**你应该这样回答面试官：**

- **原理层**：我会讲经典 **PPO + KL + 四模型**。  
- **工程层**：我们仓库 RL 阶段实际调用 **`RLOOTrainer`**，属于 RLHF 家族里的 **变体实现**，组件更省。

### 10.2 典型导入与 Trainer 构造（概念）

```text
from trl import RLOOConfig, RLOOTrainer

RLOOTrainer(
  args=training_args,
  processing_class=tokenizer,
  model=policy,
  reward_funcs=reward_model,
  train_dataset=...,
  eval_dataset=...,
  peft_config=...,
)
```

具体参数名以你本地 `trl` 版本为准。

### 10.3 数据与模板

- 从 `train_file_dir` / `validation_file_dir` 读取 json/jsonl。  
- `preprocess_function` 用 `get_conv_template(template_name)` 把多轮对话转成 **prompt** 字段。  
- 过滤空 prompt，保证训练稳定性。

### 10.4 奖励模型加载

常见写法：

```text
AutoModelForSequenceClassification.from_pretrained(
  reward_model_path, num_labels=1, ...
)
```

作为 `reward_funcs` 传入 Trainer；**前向调用链在 TRL 内部**。

### 10.5 与 L08 的衔接检查

- `reward_model_path` 必须指向 **你训练好的 RM**，不要用占位基座糊弄。  
- **chat template** 必须与 SFT/RM 数据一致。

---

## 11. 关键参数与训练命令：`run_ppo.sh` 解析

下面给出 **典型含义**；请以你本地 `run_ppo.sh` 为准。

| 参数 | 含义 |
|------|------|
| `--sft_model_path` | 策略初始化（常为 SFT 后模型） |
| `--reward_model_path` | L08 产出目录 |
| `--template_name` | 与数据一致，如 `qwen` |
| `--max_source_length` | prompt 长度上限 |
| `--max_completion_length` | 生成长度上限（显存敏感） |
| `--per_device_train_batch_size` | 常设较小：生成+打分昂贵 |
| `--gradient_accumulation_steps` | 累积步数 |
| `--dtype bfloat16` | 混合精度 |
| `--max_steps` / `--num_train_epochs` | 训练长度 |
| `--output_dir` | 输出目录 |

**实操建议：** 先用 **极小步数 smoke test**，打印或人工查看生成样例，确认 **不崩、不乱码、奖励方向合理**，再拉长训练。

---

## 12. PPO 的缺点与 RLHF 工程挑战

### 12.1 PPO 相关缺点（经典题）

- **训练不稳定**：奖励噪声、优势方差、超参敏感。  
- **显存与算力**：四模型栈昂贵；生成并行进一步吃显存。  
- **reward hacking**：对 RM 投机。  
- **调试难**：loss 好看但生成变差并不罕见。

### 12.2 RLHF 工程挑战（可写简历）

- **分布偏移**：policy 更新后 RM OOD。  
- **模板一致性**：错一位，奖励全歪。  
- **医疗合规**：奖励要覆盖 **拒答、转诊、隐私**。  
- **可复现性**：采样随机导致同样配置波动大。

### 12.3 为什么面试仍爱考 PPO？

许多工业系统仍用 **PPO 系** 或共享同一套概念的变体；论文与课程以 PPO 为 **基准坐标系**。

---

## 13. ASCII：RLHF 完整训练流程

```
        ┌─────────────────────────────────────────────────────────┐
        │ ① SFT：指令数据 → 会聊天的 π_ref / π_SFT                  │
        └───────────────────────────┬─────────────────────────────┘
                                    v
        ┌─────────────────────────────────────────────────────────┐
        │ ② 偏好数据 (chosen/rejected) → 训练 RM 得 r_φ(x,y)        │
        └───────────────────────────┬─────────────────────────────┘
                                    v
        ┌─────────────────────────────────────────────────────────┐
        │ ③ RL：采样 y ~ π_θ(·|x)                                   │
        │     奖励 ≈ r_φ(x,y) − β·KL(π_θ || π_ref) （经典形式）      │
        │     PPO-clip 更新 θ（+ Critic 等）                        │
        └───────────────────────────┬─────────────────────────────┘
                                    v
                           对齐后的 π_θ 部署 / 再迭代
```

**MedicalGPT 脚本层（RLOO）理解：** ③ 步可能换成 **更省组件** 的优化实现，但 **RM 与数据模板** 仍处同一生态位。

---

## 14. 面试高频题与白板伪代码

### Q1：PPO 的 4 个模型分别是什么？

> Policy、Reference（冻结 SFT）、Reward Model、Value/Critic。分工见第 8 节表。

### Q2：为什么需要 KL 惩罚？

> RM 仅在有限数据上可靠；无 KL 时 policy 可偏移到 OOD 区域 **刷假高分**。KL 把更新锚在 SFT 附近。

### Q3：PPO 和 DPO 的区别？（L10 预习）

> PPO：**显式 RM + 在线采样 + RL**；链路长、工程重。DPO：用偏好数据 **直接优化策略相对参考模型** 的目标，常 **无需在线 RM**，更轻（L10 展开）。

### Q4：PPO 的 clip 在防什么？

> 防止一次更新把策略改到 **离采样策略太远**，重要性采样失效导致崩溃。

### Q5：你们 `ppo_training.py` 用的是 PPO 吗？

> 以主分支为例常是 **`RLOOTrainer`**；答法：**RLHF 家族实现，变体与经典四模型 PPO 不完全一致**——体现你读过代码。

### 14.1 教科书版 PPO 一步伪代码（助记）

```text
# 用 π_old 采样 {a_t}，并预计算 Â_t
for 优化轮次 in range(K):
    for minibatch:
        r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)
        L_clip = min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t)
        L = -mean(L_clip) + c_v * L_value - c_e * Entropy
        θ ← Adam(∇_θ L)
```

---

## 15. 小结与自测

### 15.1 背清单

1. RLHF = **SFT 能力 + RM 偏好信号 + RL 优化生成分布**。  
2. PPO = **clip 限步 +（通常）Critic 降方差 + KL 约束**。  
3. MedicalGPT `ppo_training.py` = **以 TRL RLOO 实现为准阅读**。

### 15.2 自测

1. 画一张图：从 prompt 到 token 到 RM 打分。  
2. 解释 \(r_t(\theta)\) 过大时 clip 如何起作用。  
3. 打开本地 `run_ppo.sh`，找出 batch 与 `max_completion_length`。  
4. 用一段话解释 **为何 RL 阶段仍需要参考模型（经典叙事）**。

### 15.3 参考阅读

- [InstructGPT](https://arxiv.org/abs/2203.02155)  
- [PPO 原论文](https://arxiv.org/abs/1707.06347)  
- [TRL 文档](https://huggingface.co/docs/trl)  
- [MedicalGPT](https://github.com/shibing624/MedicalGPT)

---

*若上游将 `ppo_training.py` 切回经典 `PPOTrainer`，请以仓库 diff 更新第 10 节描述。*
