[← 上一课](../L09-强化学习PPO与RLHF/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L11-ORPO与GRPO前沿方法/README.md)

---

# L10 直接偏好优化 DPO（Direct Preference Optimization）

> **一句话精髓**：*「不要『在线老师』了，直接从好坏对比里学。」*

---

## 写给「刚学完 L09」的你

你已经理解：经典 **RLHF** 常常是 **SFT → 训 RM → 在线采样 → PPO 更新策略**。这条路很强，但也很「重」：**显式奖励模型**、**rollout 生成**、**价值网络**、**分布式调试** 都会吃掉团队精力。

**DPO（Direct Preference Optimization）** 提出一个更「监督化」的路线：在特定数学推导下，可以把「带 KL 约束的 RL 目标」与 **Bradley-Terry 偏好模型** 结合，**消去显式 RM**，直接用 **偏好对** 更新策略 \(\pi_\theta\)。

本课目标：你能 **对比 DPO vs RLHF**、**口述直觉推导链**、**写出 DPO loss 结构**，并能对照 **MedicalGPT `dpo_training.py` / `run_dpo.sh`** 说明数据字段与训练配置。

---

## 本课你将学会什么

1. 复述 **DPO 论文核心思想**（奖励重参数化、隐式奖励）。  
2. 清晰说明 **DPO 去掉了什么、简化了什么**。  
3. 用 **直觉版数学** 把 RM 的 BT 形式与 DPO 的 log-ratio 形式连起来。  
4. 写出 **标准 DPO Loss**（含 \(\beta\) 与 `logsigmoid`）。  
5. 理解 **只需两模型**：Policy \(\pi_\theta\) + Reference \(\pi_{\text{ref}}\)（PEFT 下 ref 可能隐式）。  
6. 列举 **DPO 优点**（简单、稳定、易实现）与 **缺点**（极依赖数据质量）。  
7. 对照 **`dpo_training.py`**：`return_prompt_and_responses`、长度过滤、`DPOTrainer`。  
8. 逐项理解 **`run_dpo.sh`**。  
9. 明确 **偏好数据格式** 与 L08 RM 的字段同构性。  
10. 画 **DPO vs RLHF ASCII 对比图**。  
11. 回答面试题：**DPO loss？DPO vs PPO 效果？何时选 DPO？**

---

## 目录

1. [类比：从陪练到错题本](#1-类比从陪练到错题本)
2. [DPO 论文核心思想](#2-dpo-论文核心思想)
3. [DPO vs RLHF：去掉了什么？简化了什么？](#3-dpo-vs-rlhf去掉了什么简化了什么)
4. [数学推导（直觉版）](#4-数学推导直觉版)
5. [DPO 的 Loss 函数](#5-dpo-的-loss-函数)
6. [两个模型：策略 + 参考](#6-两个模型策略--参考)
7. [优点、缺点与数据质量](#7-优点缺点与数据质量)
8. [MedicalGPT：`dpo_training.py` 核心逻辑](#8-medicalgptdpo_trainingpy-核心逻辑)
9. [关键参数与训练命令：`run_dpo.sh` 解析](#9-关键参数与训练命令run_dposh-解析)
10. [DPO 的偏好数据格式](#10-dpo-的偏好数据格式)
11. [ASCII：DPO vs RLHF](#11-asciidpo-vs-rlhf)
12. [训练排障与监控清单](#12-训练排障与监控清单)
13. [面试高频题](#13-面试高频题)
14. [小结、自测与延伸阅读](#14-小结自测与延伸阅读)

---

## 1. 类比：从陪练到错题本

**RLHF + PPO** 像「考试时现场请阅卷老师给分，你根据分数当场改写作习惯」——老师（RM）要一直在旁边；考试过程（采样）还要反复多轮。

**DPO** 像「老师提前把『这篇比那篇好』写在错题本上；你回家只对着错题本改，不再每场考试都请老师」——**偏好信息直接进 loss**，通常 **不需要在线 RM 推理**。

```
RLHF:  生成 → RM 打分 → 更新 → 再生成 → …
DPO:   偏好对 (y_w, y_l) 一次性进 batch → 直接梯度更新
```

---

## 2. DPO 论文核心思想

DPO（Rafailov et al.）指出：在 **Bradley-Terry + KL 约束下的 RL 目标** 与 **隐式奖励** 之间可建立变换，从而 **消去显式奖励模型**，直接用偏好对优化 \(\pi_\theta\) 相对 \(\pi_{\text{ref}}\) 的 **对数比率**。

**一句话：** 把人类偏好编码进 **策略与参考模型的似然差**，用分类式 loss **拉近赢家、推远输家**。

论文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

---

## 3. DPO vs RLHF：去掉了什么？简化了什么？

| 维度 | 经典 RLHF（PPO 路径） | DPO |
|------|------------------------|-----|
| **显式 RM** | 需要单独训练与部署 | **不需要**（奖励信息隐式） |
| **在线采样** | policy rollout | **通常不需要**（数据已有 \(y_w,y_l\)） |
| **价值网络** | 常见需要 Critic | **不需要** |
| **优化器** | PPO 等多步 RL | 偏好 loss，**更稳定** |
| **工程复杂度** | 高 | **相对低** |
| **数据** | 偏好训 RM + prompt 训 RL | **高质量偏好对** 更关键 |

**去掉了什么？** 显式 RM、RL 内环采样、（经典 PPO 里的）Critic。  
**简化了什么？** 训练栈从「多阶段多服务」收缩为 **主模型 + 参考（冻结拷贝）+ 偏好数据**。

---

## 4. 数学推导（直觉版）

### 4.1 从 BT 模型开始（与 L08 衔接）

人类偏好可写为：

\[
P(y_w \succ y_l \mid x)=\sigma\big(r^*(x,y_w)-r^*(x,y_l)\big)
\]

\(r^*\) 是真实人类奖励函数，我们 **无法直接访问**。

### 4.2 RL 阶段想做什么（口头）

在 KL 约束下最大化期望「奖励」时，最优策略 \(\pi^*\) 与 \(r^*\) 之间存在 **闭式关系**（论文称 **reward reparameterization**）：可把 \(r^*\) 与 \(\log \pi^* - \log \pi_{\text{ref}}\) 联系起来（差配分函数项；在 **成对偏好** 中会抵消关键部分）。

### 4.3 关键直觉（背这个就够面试）

- 若 \(\pi_\theta\) 对 **赢家** 赋更高概率、对 **输家** 更低概率（都 **相对** \(\pi_{\text{ref}}\)），就更符合 BT 偏好。  
- 把该要求写成 **log-sigmoid** 形式，即得 DPO loss——你会看到 **L08 的 sigmoid 影子**，只是 margin 从 **显式 RM 分数** 换成 **log \(\pi\) 比率**。

### 4.4 小白背公式策略

第一次不必推导配分函数。先记结构：

- 里面有 \(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}\) 与 \(\log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\)。  
- 两者相减，乘 **\(\beta\)**，再套 **\(-\log\sigma(\cdot)\)**。

---

## 5. DPO 的 Loss 函数

### 5.1 标准形式（一条样本）

\[
\mathcal{L}_{\text{DPO}}=-\log\sigma\Big(\beta\big[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big]\Big)
\]

- \(y_w\)：winner / chosen；\(y_l\)：loser / rejected。  
- \(\beta\)：**温度**：越大越强硬偏好赢家；过大可能不稳定或过拟合噪声。  
- 实现常用 **`logsigmoid`**。

### 5.2 与 L08 RM loss 的形似

- RM：\(-\log\sigma(r_\phi(y_w)-r_\phi(y_l))\)  
- DPO：\(-\log\sigma(\beta\cdot(\text{logit-ratio 差}))\)

**本质都在拉大赢家与输家的 margin**，只是 DPO 的 margin 定义在 **策略空间**。

### 5.3 玩具数字直觉

设：

- \(\Delta_w=\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}=+0.4\)  
- \(\Delta_l=\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}=-0.1\)

则 \(\Delta_w-\Delta_l=0.5\)。\(\beta=0.1\) 时 sigmoid 自变量为 \(0.05\)，loss 较小——说明策略已更倾向赢家。若 margin 为负，loss 变大，梯度推动修正。

### 5.4 logprob 怎么实现？

对回答部分每个 token 的对数概率 **求和**（正确 mask 掉 prompt）：

```text
log π_θ(y|x) ≈ Σ_t log π_θ(y_t | x, y_<t)
```

**工程坑：** assistant 标签是否进入 loss mask 错一位，等价于 **教错目标**。

---

## 6. 两个模型：策略 + 参考

### 6.1 标准配置

- **Policy \(\pi_\theta\)**：正在训练（全参或 LoRA）。  
- **Reference \(\pi_{\text{ref}}\)**：常取 **SFT 快照**，**冻结**，提供 KL 意义的锚点。

### 6.2 LoRA 时的常见处理

在 TRL `DPOTrainer` 中，若启用 `peft` 且 `ref_model=None`，参考分支可能通过 **禁用 adapter** 等方式从同一骨干得到 ref 行为（以 TRL 版本为准）。

**面试说法：** 「参考模型提供无适配器下的基座分布；policy 带 LoRA 偏移。」

### 6.3 显存直觉

要算 chosen 与 rejected 两路 logprob，**不是零成本**；但通常仍 **远轻于四模型 PPO 栈**。

---

## 7. 优点、缺点与数据质量

### 7.1 优点

- **实现简单**：HF + TRL 易跑通。  
- **相对稳定**：无 PPO 式 rollout 方差主导。  
- **易调试**：loss、margin 监控接近分类任务。

### 7.2 缺点

- **极依赖偏好数据质量**：噪声标签直接推偏策略。  
- **chosen 不够强**：天花板低。  
- **rejected 太弱**：学不到难负例（与 RM 同理）。  
- **长度敏感性**：logprob 累加与长度相关；需注意截断与模板。

### 7.3 何时选 DPO（简版，L11 有决策树）

- 资源有限、想快速对齐；  
- 已有 **可靠** 偏好对；  
- 不想维护 RM 服务与 RL 内环。

---

## 8. MedicalGPT：`dpo_training.py` 核心逻辑

以下与 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 主分支结构对齐（以本地为准）。

### 8.1 依赖与入口

- `trl.DPOTrainer`, `trl.DPOConfig`  
- `transformers.AutoModelForCausalLM`  
- `template.get_conv_template` 统一对话格式

### 8.2 数据：`return_prompt_and_responses`

从样本字段构造：

- `prompt`：`system` + `history` + `question` 模板拼接。  
- `chosen`：`response_chosen`  
- `rejected`：`response_rejected`

`dataset.map` 后得到三列供 `DPOTrainer` 使用。

### 8.3 长度过滤

```text
full_max_length = max_source_length + max_target_length
filter: len(prompt+chosen) 与 len(prompt+rejected) 在 (0, full_max_length]
```

避免空串或超长。

### 8.4 模型与 Trainer

- 支持 **QLoRA**（`BitsAndBytesConfig`）与全参。  
- `DPOConfig`：`max_length=full_max_length`、batch、学习率、bf16/fp16 等。  
- `DPOTrainer(model, ref_model=None if use_peft else deepcopy(model), ...)`：全参时常 **深拷贝** ref；PEFT 时常 `ref_model=None`。

### 8.5 训练循环

`trainer.train()`，保存 adapter 或全量到 `output_dir`。

---

## 9. 关键参数与训练命令：`run_dpo.sh` 解析

| 参数 | 含义 |
|------|------|
| `CUDA_VISIBLE_DEVICES=0,1` | 多卡（视启动方式） |
| `--model_name_or_path` | 常为 **SFT 后** 模型 |
| `--template_name` | 与数据一致，如 `qwen` |
| `--train_file_dir` / `--validation_file_dir` | 偏好 json/jsonl（可与 RM 同源） |
| `--per_device_train_batch_size` | 每卡 batch |
| `--gradient_accumulation_steps` | 累积 |
| `--max_train_samples` / `--max_eval_samples` | 调试截断 |
| `--max_steps` | 总步数（示例可能偏短） |
| `--max_source_length` / `--max_target_length` | 长度预算 |
| `--use_peft True` + LoRA | 省显存 |
| `--torch_dtype bfloat16` + `--bf16 True` | 混合精度 |
| `--gradient_checkpointing True` | 省显存 |
| `--remove_unused_columns False` | 保留自定义列 |
| `--output_dir` | 输出 |

**实战：** MedicalGPT Colab 提供 **PT+SFT+DPO** 快速流水线，适合先跑通再换医疗数据。

---

## 10. DPO 的偏好数据格式

| 字段 | 说明 |
|------|------|
| `system` | 系统提示（可为空字符串） |
| `history` | 多轮列表，如 `[[q1,a1],[q2,a2],...]` |
| `question` | 当前用户问题 |
| `response_chosen` | 更优回答 |
| `response_rejected` | 较差回答 |

**医疗建议：** chosen 引用 **指南级** 表述；rejected 覆盖 **常见误区**（自行用药、绝对化结论）。同一题多组 rejected 可增强鲁棒性，但避免 **假负例**（rejected 其实更好）。

### 10.1 与其它偏好方法坐标（了解）

| 方法 | 一句话 |
|------|--------|
| **IPO** | 改进噪声偏好下的目标形态。 |
| **SimPO 系** | 序列级 margin 变体。 |
| **KTO** | 二元好/坏标签，适合不同标注形态。 |

先把 **DPO** 讲透，再扩展阅读。

---

## 11. ASCII：DPO vs RLHF

```
                    经典 RLHF (PPO 路径)
    ┌─────────────────────────────────────────────────────────┐
    │ SFT → 训 RM → 在线采样 → RM 打分 → PPO 更新 Policy        │
    │          ↑                    ↑                          │
    │      偏好数据              需要 RM 服务                   │
    └─────────────────────────────────────────────────────────┘

                         DPO
    ┌─────────────────────────────────────────────────────────┐
    │ SFT 得 π_ref ──冻结或隐式参考                             │
    │        +                                                  │
    │ 偏好对 (prompt, chosen, rejected)                         │
    │        ↓                                                  │
    │  DPOTrainer：直接优化 π_θ（无显式 RM）                     │
    └─────────────────────────────────────────────────────────┘
```

---

## 12. 训练排障与监控清单

| 现象 | 可能原因 | 可尝试 |
|------|-----------|--------|
| loss ~0.693 | margin≈0 | 检查 chosen/rejected；调 \(\beta\)；增强 rejected |
| 生成变短、复读 | 过强偏好 | 减小 \(\beta\)；清洗数据；混合 SFT |
| OOM | 双路长序列 | 降 `max_target_length`；checkpointing；LoRA |
| 能力崩 | 对齐过强 | 混合 SFT；减步数；网格搜 \(\beta\) 与 lr |

**监控建议（除 loss 外）：** 学习率、eval 人工抽检表、生成长度分布、（若日志提供）implicit reward / margin。

---

## 13. 面试高频题

**Q1：DPO 的 loss 怎么写？**

> \(-\log\sigma(\beta[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}])\)；实现用 `logsigmoid`。

**Q2：DPO 和 PPO 哪个效果好？**

> **无普适答案**；取决于数据、规模、任务。高质量偏好下 DPO **性价比高**；复杂奖励 shaping 时部分团队仍走 PPO。医疗要看 **安全标注与评测**。

**Q3：什么时候选 DPO？**

> 想省 RM+RL、偏好可靠、迭代快；资源有限或首版对齐。

**Q4：DPO 还需要参考模型吗？**

> 理论上要 \(\pi_{\text{ref}}\) 锚定；实现上 **冻结拷贝** 或 PEFT 下 TRL 隐式 ref。

**Q5：DPO 会不会过拟合 chosen？**

> 会。需 \(\beta\)、多样化数据、held-out 评测、能力保留监控。

---

## 14. 小结、自测与延伸阅读

### 14.1 三句话

1. DPO 把 BT+KL-RL 中的奖励 **重参数化** 进 **策略对数比率**，省去显式 RM。  
2. Loss 仍是 **sigmoid margin** 形式，与 RM pairwise loss **同源不同域**。  
3. MedicalGPT 用 **`DPOTrainer` + 模板化三列** 与 RM 数据 **同构**。

### 14.2 自测

1. 默写 DPO loss 括号内两项差。  
2. 解释 \(\beta\) 增大趋势。  
3. 打开 `dpo_training.py`，找到 `DPOTrainer` 构造处。

### 14.3 迷你练习

**A（讲给非技术朋友）：** 为什么 DPO 可以不要 RM？  
**要点：** 人类只给相对好坏；DPO 把它变成「相对参考模型，应把概率质量挪向赢家句子」——奖励藏在比率里。

**B：** 若 rejected 其实更医学严谨只是更啰嗦？  
**要点：** 模型会学 **错偏好**；标注指南必须定义「何为更好」。

### 14.4 数据复用说明

同一套 `response_chosen` / `response_rejected` 字段上，可 **先训 RM（L08）** 再走 **RLHF（L09）**，或 **跳过 RM 直接 DPO**——Pipeline 切换成本主要在脚本与超参。

### 14.5 参考

- [DPO 论文](https://arxiv.org/abs/2305.18290)  
- [HuggingFace TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)  
- [MedicalGPT](https://github.com/shibing624/MedicalGPT)

---

## 附录 A：把 DPO 讲成「五步故事」（背诵版）

下面这段可以当作你在面试里 **60 秒版本** 的腹稿：

1. **人类只给相对偏好**：在同一上下文下，\(y_w\) 比 \(y_l\) 更好。  
2. **BT 模型**：把这种偏好写成 \(\sigma(r(y_w)-r(y_l))\)。  
3. **但我们不想单独训 \(r_\phi\)**：RLHF 里 RM 训练与 RL 训练是两条重流水线。  
4. **DPO 的关键一步**：在 KL 正则的最优策略形式下，把未知的 \(r^*\) 与 \(\log\pi-\log\pi_{\text{ref}}\) 联系起来，使得 **偏好对比只需要策略与参考的似然比**。  
5. **落到实现**：对每个 token 累加 logprob，构造 margin，套 `-logsigmoid`，用 `DPOTrainer` 直接反传更新 \(\pi_\theta\)。

你可以把第 4 步说成「数学上把奖励藏进策略比率里」——面试官若追问配分函数，你诚实回答：**细节见论文，我实现层主要关心 loss 结构与数据 mask。**

---

## 附录 B：与 L08、L09 的「同一条偏好数据」怎么切换？

| 你想跑的路径 | 需要额外训练 | 典型入口脚本 |
|--------------|--------------|--------------|
| RM → PPO/RLOO | RM + RL 两阶段 | `reward_modeling.py` → `ppo_training.py` |
| 直接 DPO | 无 RM | `dpo_training.py` |

**字段一致时的收益：** 标注团队仍产出 `(chosen, rejected)`；算法团队可以 **A/B** 不同对齐路线而 **不重标**。  
**仍要做的事：** `template_name`、`max_source_length`、tokenizer、评测集必须对齐；否则比较不公平。

---

## 附录 C：DPO 实现里常见的 3 个「隐形 bug」

1. **prompt/chosen 边界切错**：把不该训练的部分算进 logprob，等价于教模型「背诵模板标签」。  
2. **参考模型不是 SFT 快照**：若 ref 与 policy 起点不一致，\(\beta\) 的含义漂移，表现为「怎么调都不对」。  
3. **验证集泄漏**：chosen 与训练集问题高度重复，会让 loss/metric **虚好**，上线后用户问法一变就崩。

---

*TRL 版本升级时参数名可能变化，以官方迁移指南为准。*
