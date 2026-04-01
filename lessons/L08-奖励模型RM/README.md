[← 上一课](../L07-LoRA与QLoRA高效微调/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L09-强化学习PPO与RLHF/README.md)

---

# L08 奖励模型 RM（Reward Model）

> **一句话精髓**：*「训练一个『老师』来给回答打分。」*

---

## 写给「刚学完 L01–L07」的你

你已经知道：**预训练（PT）** 让模型懂语言与世界知识，**有监督微调（SFT）** 让模型学会「按指令、按格式、按示范」说话。  
但真实产品里，用户很少只问「请把下面这句话翻译成英文」这种有标准答案的事；更多时候是开放域问答、多轮问诊、复杂决策辅助——**同一问题往往有多种「看起来都对」的回答**，人类关心的是：**哪一个更有帮助、更诚实、更安全**。

**本课要解决的矛盾：**

- 强化学习（下一课 L09）需要 **标量奖励** \(r\) 才能「往好的方向」优化生成模型。  
- 人类却很难对互联网上 **每一个** 候选回答都打一个 **绝对分数**（贵、慢、人与人之间尺度不一致）。  
- 折中方案：让人类只做 **二选一**（或排序），训练一个神经网络 **模仿人类这种相对比较**——这就是 **奖励模型（Reward Model, RM）**。

读完本课，你应该能 **用自己的话** 讲清 RM 是什么、数据长什么样、loss 为什么长那样，并能 **打开 MedicalGPT 的 `reward_modeling.py` / `run_rm.sh`** 说出数据怎么流、参数怎么调。

---

## 本课你将学会什么

1. 用 **阅卷老师 / 作文评分** 类比理解 RM 与 SFT 的差异。  
2. 准确说出 **Reward Model** 的输入输出，以及它在 **RLHF 流水线** 中的位置。  
3. 解释 **为什么人类不能给每个回答打分**，以及 **成对偏好** 为何更可行。  
4. 记住 **HHH 原则**（Helpful / Honest / Harmless）及医疗场景下的落点。  
5. 理解 **chosen vs rejected** 偏好数据的构造方式与常见坑。  
6. 从 **LM Head** 走到 **Value / Score Head**：架构上「生成」如何变「打分」。  
7. 掌握 **Bradley-Terry（BT）排序模型** 与偏好概率公式。  
8. 背下并实现 **Loss：\(-\log\sigma(r_{\text{chosen}}-r_{\text{rejected}})\)** 的直觉与 `logsigmoid` 写法。  
9. 对照 **MedicalGPT `reward_modeling.py`**：`preprocess`、`collator`、`compute_loss`、`metrics`。  
10. 逐项读懂 **`run_rm.sh`** 中与显存、精度、LoRA、数据路径相关的参数。  
11. 知道 **RM 的常见评估指标**（pairwise accuracy、margin、曲线监控）。  
12. 能画出 **RM 训练流程 ASCII 图**，并在面试中 **30 秒内复述**。  
13. 回答面试高频题：**RM 的 loss 是什么？为什么用对比训练？**  
14. 建立 **OOD / reward hacking** 等「下一课伏笔」概念，避免把 RM 分数当成客观真理。

---

## 目录

1. [类比开场：从「标准答案」到「作文评分」](#1-类比开场从标准答案到作文评分)
2. [什么是奖励模型（Reward Model）](#2-什么是奖励模型reward-model)
3. [为什么需要奖励模型：人类不能给每个回答打分](#3-为什么需要奖励模型人类不能给每个回答打分)
4. [HHH：Helpful / Honest / Harmless](#4-hhhhelpful--honest--harmless)
5. [偏好数据的构造：chosen vs rejected](#5-偏好数据的构造chosen-vs-rejected)
6. [架构：去掉 LM Head，加上 Value Head](#6-架构去掉-lm-head加上-value-head)
7. [Bradley-Terry 排序模型](#7-bradley-terry-排序模型)
8. [Loss：\(-\log\sigma(r_{\text{chosen}}-r_{\text{rejected}})\)](#8-loss-logσr_chosen--r_rejected)
9. [MedicalGPT：`reward_modeling.py` 核心逻辑](#9-medicalgptreward_modelingpy-核心逻辑)
10. [关键参数与训练命令：`run_rm.sh` 解析](#10-关键参数与训练命令run_rmsh-解析)
11. [奖励模型的评估指标与调试信号](#11-奖励模型的评估指标与调试信号)
12. [ASCII：奖励模型训练流程](#12-ascii奖励模型训练流程)
13. [面试高频题与速记卡片](#13-面试高频题与速记卡片)
14. [小结、自测与延伸阅读](#14-小结自测与延伸阅读)

---

## 1. 类比开场：从「标准答案」到「作文评分」

### 1.1 SFT 像「选择题 / 填空题」

SFT 的数据常常是：

\[
(x,\ y^\star)
\]

其中 \(y^\star\) 是 **示范答案**。训练目标很像：「请把老师的标准写法背下来。」这对 **格式、流程、术语** 很有效。

### 1.2 对齐像「作文」：没有唯一真解

用户问：「我最近总是胸闷，可能是什么问题？」

- 回答 A：先问伴随症状、危险因素，建议必要时就医检查。  
- 回答 B：直接断言「你就是心脏病」并给出具体药名剂量。  

从「语言模型会不会造句」角度，B 甚至可能 **更流利**；但从 **医疗安全与人类偏好** 角度，A 往往 **明显更好**。  
这类判断很难压缩成「唯一字符串监督」，更适合 **相对比较**。

### 1.3 RM 是谁？

RM 是一个 **可微的打分器** \(r_\phi(x,y)\)：输入 **提示+回答**，输出 **一个实数**。  
它不学「下一个词是什么」，而学 **「这段话在人类偏好意义下值多少分」**（在训练分布内、在标注规则下）。

```
        人类只做「A 更好」这种相对判断
                    |
                    v
    ┌───────────────────────────────────────┐
    │  大量样本：(prompt, chosen, rejected)   │
    └───────────────────────────────────────┘
                    |
                    v
    ┌───────────────────────────────────────┐
    │  RM：学会输出与偏好一致的标量排序        │
    └───────────────────────────────────────┘
                    |
                    v
         供 L09 强化学习当作 reward 信号
```

---

## 2. 什么是奖励模型（Reward Model）

### 2.1 定义（口语 + 符号）

**奖励模型** 是一个带参数 \(\phi\) 的函数：

\[
r_\phi(x,y)\in\mathbb{R}
\]

- \(x\)：**上下文**（system、history、用户问题等拼成的提示）。  
- \(y\)：**模型候选回答**（完整 assistant 回复）。  
- 输出：**标量奖励**；越大表示越符合训练时所编码的人类偏好。

### 2.2 它和「语言模型头」有什么不一样？

| 组件 | 典型输出 | 问题形式 |
|------|-----------|----------|
| **Causal LM + LM Head** | 词表上的概率分布 | 下一个 token 是什么？ |
| **RM + Value Head** | 标量 logit | 整段回答好不好？ |

### 2.3 RM 在 RLHF 里站哪一站？

经典三段式（L09 会展开）：

1. **SFT**：学会说话与遵循指令。  
2. **RM**：学会按人类偏好打分。  
3. **RL（如 PPO）**：让 **会生成的策略模型** 在 rollout 中获得更高 RM 分数（常配 KL 约束）。

你现在学的 RM，就是第 2 站。

---

## 3. 为什么需要奖励模型：人类不能给每个回答打分

### 3.1 组合爆炸

语言是 **组合空间**。对同一个 \(x\)，候选 \(y\) 的数量随长度指数增长。  
即使只考虑「模型实际可能采样到的 Top-K 空间」，也无法让人类对 **每一条** 都打绝对分。

### 3.2 绝对分的三个工程痛点

1. **成本高**：请医生/专家标注绝对分，比二选一贵得多。  
2. **不一致**：同一个人在不同疲劳程度下，7 分和 8 分的边界会漂。  
3. **跨人不可比**：A 标注员的「6 分」与 B 标注员的「6 分」不对齐。

### 3.3 成对比较更「人类友好」

大量实验与标注实践表明：人对 **「A 是否优于 B」** 的判断，往往比 **「A 是几分」** 更稳定。  
因此我们收集：

\[
(x,\ y_{\text{win}},\ y_{\text{lose}})
\]

并训练 RM 满足：**在 \(x\) 条件下，更偏好 \(y_{\text{win}}\)**。

### 3.4 RM 是「可微近似评委」

强化学习需要 **频繁** 的奖励信号。RM 把 **高维文本** 压缩成 **标量**，使得后续算法可以对 **生成策略** 做优化。  
没有 RM 时，你只能写规则（关键词、格式、长度），很难表达 **「是否夸大疗效」「是否足够谨慎」** 这类细腻目标。

---

## 4. HHH：Helpful / Honest / Harmless

工业界讨论对齐目标时，常把「好回答」拆成三块（公开材料中广泛引用）：

| 原则 | 直觉 | 医疗场景举例 |
|------|------|----------------|
| **Helpful（有用）** | 真的帮用户推进问题解决 | 给出可执行的就诊与检查边界，而不是空话套话 |
| **Honest（诚实）** | 不确定就说，不编造 | 不捏造剂量、不伪造指南条文与引用 |
| **Harmless（无害）** | 降低身体、心理、社会风险 | 避免替代急诊判断；避免歧视与羞辱性表述 |

### 4.1 重要澄清：RM 不会自动理解 HHH

RM **只学习数据里体现的偏好**。如果你的标注指南没有明确「Harmless 优先于简短」，RM 可能学会 **「短、像客服」** 但并不安全。  
**数据与标注规则** 才是「价值观」的来源，网络结构只是 **拟合器**。

### 4.2 当 HHH 冲突时怎么办？

真实产品里会有冲突：更「完全诚实」可能更「吓人」，更「无害」可能更「拒答」。  
这类优先级必须在 **标注指南** 写清楚；RM 训练阶段只能通过 **样本分布** 反映这些规则。

---

## 5. 偏好数据的构造：chosen vs rejected

### 5.1 一条样本常见字段

与 MedicalGPT 数据处理思想一致时，常见字段包括：

- `system`：系统提示（可为空）。  
- `history`：多轮对话历史。  
- `question`：当前用户问题。  
- `response_chosen`：**更好** 的回答。  
- `response_rejected`：**相对更差** 的回答。

### 5.2 rejected 不一定是「错误答案」

很多时候 rejected **语法正确、看似合理**，只是在 **安全性、完整性、遵循指南** 上输给 chosen。  
把 rejected 当成「一定胡说」会误导你对 loss 与评测的理解。

### 5.3 负例构造策略（工程向）

| 策略 | 作用 | 风险 |
|------|------|------|
| **模型采样差答案** | rejected 更贴近真实错误模式 | 若采样太弱，变成「稻草人对手」，RM 太好训但不泛化 |
| **人工改写** | 控制错误类型（遗漏追问、过度承诺） | 成本高；改写者偏见进入数据 |
| **对抗/难例挖掘** | 提升 RM 判别力 | 标注噪声上升；需要质检 |

### 5.4 极简 JSON 示意（概念）

```json
{
  "system": "你是一名谨慎的医学助手。",
  "history": [],
  "question": "发烧 38.5℃ 需要立刻去急诊吗？",
  "response_chosen": "不一定。需结合年龄、精神状态、呼吸困难等…若出现警示症状或持续加重应及时就医。",
  "response_rejected": "不用，多喝水就行。"
}
```

---

## 6. 架构：去掉 LM Head，加上 Value Head

### 6.1 从「预测下一个词」到「给整段话打分」

- **因果语言模型**：最后一层接 **LM Head**，维度 = 词表大小。  
- **序列分类（回归）式 RM**：骨干仍是 Transformer，但最后不接词表，而是接 **Score / Value Head**，输出 **1 个 logit**。

在 HuggingFace 生态里，常见类名是 **`AutoModelForSequenceClassification`**，并设置 `num_labels=1`。

### 6.2 池化：如何把一整段变成向量？

常见做法包括：

- 取 **最后一个非 pad token** 的隐状态；  
- 或对有效 token 做 **mean pooling**（视实现而定）。

你要记住的面试要点是：**RM 输出是标量奖励**，不是词分布。

### 6.3 为什么常用 SFT 模型初始化骨干？

RM 需要理解 **医学语义、对话格式、指令遵循**。用已经 **SFT 过** 的权重初始化，通常比随机初始化更容易学到「什么是更专业的回答」。  
直觉：**先会听懂人话，再学当评委**。

### 6.4 类比：同一套「大脑」，换「考试题型」

把 Transformer 骨干想成 **阅读理解能力很强的人**：

- 装上 LM Head：参加「接龙考试」。  
- 换上 Value Head：参加「给作文打分」——读的还是文字，但输出从「下一个字」变成 **一个分数**。

---

## 7. Bradley-Terry 排序模型

### 7.1 从「谁赢」到「概率」

对同一 \(x\) 下的两条回答 \(y_1,y_2\)，BT 模型把「\(y_1\) 优于 \(y_2\)」的概率写成：

\[
P(y_1 \succ y_2 \mid x)=\sigma\big(r(x,y_1)-r(x,y_2)\big)
\]

其中 \(\sigma\) 为 logistic 函数。  
**RM 的角色**：用 \(r_\phi\) 去逼近人类偏好的这种相对比较。

### 7.2 为什么用差分？

因为人类标注提供的是 **相对序**，不是绝对标度。用 **差分** 自动抵消许多「整体偏置」（例如某人普遍打分偏高）。

### 7.3 与 Plackett-Luce 等的关系（了解即可）

面试一般考到 BT  pairwise 就够。若数据是 **多条排序** 而非二元对，会有更一般的排序模型；工程上 **pairwise** 仍然最常见。

---

## 8. Loss：\(-\log\sigma(r_{\text{chosen}}-r_{\text{rejected}})\)

### 8.1 极大似然 → 负对数似然

若人类标注「chosen 优于 rejected」，我们希望：

\[
P(y_{\text{chosen}}\succ y_{\text{rejected}}\mid x)\rightarrow 1
\]

等价于最小化：

\[
\mathcal{L}=-\log \sigma\big(r_\phi(x,y_{\text{chosen}})-r_\phi(x,y_{\text{rejected}})\big)
\]

对一个 batch 再取平均即可。

### 8.2 PyTorch 稳定实现

```python
import torch.nn.functional as F

# rewards_chosen, rewards_rejected: shape [batch]
loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

`logsigmoid` 比 `torch.log(torch.sigmoid(...))` **数值更稳**，工程上应优先使用。

### 8.3 玩具演算（建立手感）

假设某个 batch 里：

- \(r_{\text{chosen}}=2.0\)，\(r_{\text{rejected}}=1.0\)，差为 \(1.0\)。  
- \(\sigma(1.0)\approx 0.73\)，\(-\log 0.73 \approx 0.31\)（较小，说明模型已经挺「同意」这条偏好）。

若差接近 0，则 \(\sigma(0)=0.5\)，\(-\log 0.5=\log 2\approx 0.693\)。  
所以你若在训练日志里经常看到 **loss 卡在 0.693 附近**，常意味着 **margin 拉不开**（数据太像、学习率太小、或标注噪声大）。

### 8.4 为什么用对比而不是「单条回归成 7.3 分」？

- 标注形态匹配：**相对比较** 直接对应 **差分 + sigmoid**。  
- 标注成本匹配：不需要统一绝对尺度。  
- 鲁棒性：减少不同标注员「分数漂移」的影响。

---

## 9. MedicalGPT：`reward_modeling.py` 核心逻辑

以下描述与 [MedicalGPT 开源仓库](https://github.com/shibing624/MedicalGPT) 的设计思想对齐；**参数名与类名以你本地克隆版本为准**。

### 9.1 数据预处理：`preprocess_reward_function`

- 读取 `system`, `history`, `question`, `response_chosen`, `response_rejected`。  
- 用 **同一对话模板**（如 `template.get_prompt`）分别拼出：

  - **chosen 侧**：历史 + 当前问题 + chosen 回答；  
  - **rejected 侧**：历史 + 当前问题 + rejected 回答。

- 分别 tokenize，得到 `input_ids_chosen`、`input_ids_rejected` 等张量。

**关键约束：** chosen/rejected 必须对应 **同一 `question` 上下文**，否则 RM 学到的是 **虚假相关**（例如把「更长」当成「更好」）。

### 9.2 批处理：`RewardDataCollatorWithPadding`

- 一个 batch 内对 chosen 序列 padding 对齐；  
- 再对 rejected 序列 padding 对齐；  
- 产出模型 forward 所需的 `input_ids`、`attention_mask` 等。

直觉：**左右两摞作业本分别码整齐**，再一起送进 GPU。

### 9.3 `RewardTrainer.compute_loss`（概念对齐）

核心计算链：

```text
r_chosen  = RM_forward(batch_chosen)
r_rejected = RM_forward(batch_rejected)
loss = -mean(logsigmoid(r_chosen - r_rejected))
```

### 9.4 评估：`compute_metrics`

验证集上除了 loss，还可能记录 **MSE/MAE** 类指标（具体取决于 labels 如何构造）。  
**面试说法：** BT pairwise loss 是 **主目标**；MSE/MAE 更多是 **监控是否数值爆炸**，不要机械等同为「人类满意度」。

### 9.5 与 LoRA 的衔接（承接 L07）

脚本常支持 `use_peft`、`TaskType.SEQ_CLS`，并在 `find_all_linear_names` 中 **跳过** 不应低秩化的头（如 `score`）。  
这与 L07 的思想一致：**只训练便签纸（LoRA），不动整本书（骨干）**——但 **score head** 往往仍要参与学习，具体以仓库实现为准。

### 9.6 你读源码时的「检查清单」

1. `chosen` 与 `rejected` 的模板拼接是否 **逐字段一致**？  
2. `labels` 或 `return_tensors` 是否与 `Trainer` 兼容？  
3. `remove_unused_columns=False` 是否配合自定义 collator？  
4. 验证集是否与训练集 **同分布**（避免「泄漏」）？

---

## 10. 关键参数与训练命令：`run_rm.sh` 解析

官方示例脚本会随仓库更新；下面给出 **典型参数语义** 与 **调参方向**。请以你本地 `run_rm.sh` 为准做 diff。

### 10.1 设备与分布式

| 参数/写法 | 含义 |
|-----------|------|
| `CUDA_VISIBLE_DEVICES=0` | 只暴露一张卡给进程 |
| `CUDA_VISIBLE_DEVICES=0,1` | 暴露两张卡；是否 DDP 取决于启动命令 |
| 注释若写明 **RM 暂不支持 torchrun** | 用单进程 `python` 启动，避免踩坑 |

### 10.2 模型与数据路径

| 参数 | 含义 |
|------|------|
| `--model_name_or_path` | 常为 **SFT 后** 的 instruct 模型 |
| `--train_file_dir` / `--validation_file_dir` | 放 json/jsonl 偏好数据 |
| `--output_dir` | RM checkpoint，供 L09 作为 `reward_model_path` |

### 10.3 长度与 batch

| 参数 | 含义 |
|------|------|
| `--max_source_length` / `--max_target_length` | 控制拼接后长度预算；过长样本可能被过滤或截断 |
| `--per_device_train_batch_size` | 每卡 batch；注意一次 step 往往 forward **两次**（chosen+rejected） |
| `--gradient_accumulation_steps` | 累积梯度，等价增大 batch |

### 10.4 精度、显存与稳定性

| 参数 | 含义 |
|------|------|
| `--bf16` / `--torch_dtype bfloat16` | 混合精度训练 |
| `--gradient_checkpointing True` | 以算换显存 |
| `--use_peft True` + LoRA 相关 | 参数高效微调 |
| `--remove_unused_columns False` | 保留自定义字段给 collator |
| `--ddp_find_unused_parameters False` | DDP 常见设置，避免部分参数未参与 loss 报错 |

### 10.5 实战建议（小白向）

1. 先用 `--max_train_samples` 做 **smoke test**，确认 **不 OOM、loss 可降**。  
2. 再放开全量；否则很难判断是 **数据** 还是 **配置** 问题。  
3. 医疗数据先检查 **模板是否与 SFT 完全一致**，否则 RM 学的是 **格式差异** 而非 **医学偏好**。

---

## 11. 奖励模型的评估指标与调试信号

### 11.1 Pairwise Accuracy（排序准确率）

在 held-out 偏好对上统计：

\[
\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\big[r_\phi(x_i,y^{\text{chosen}}_i)>r_\phi(x_i,y^{\text{rejected}}_i)\big]
\]

这是面试与论文里非常常见的 **RM 质量** 指标。

### 11.2 Margin 监控

训练时打印或记录：

\[
\Delta=\mathbb{E}[r_{\text{chosen}}-r_{\text{rejected}}]
\]

- \(\Delta\) 长期接近 0：模型 **学不动** 或数据 **不可分**。  
- \(\Delta\) 极大且验证集泛化差：可能 **过拟合** 或 rejected **太弱**。

### 11.3 校准与「分数绝对值」陷阱

BT 训练主要保证 **排序**，不保证 \(r_\phi\) 的 **绝对尺度** 具有跨任务意义。  
因此 **不要** 把「RM 输出 8.88」直接宣传为「人类分 8.88」。

### 11.4 常见坑表

| 现象 | 可能原因 |
|------|-----------|
| loss 不降 | lr 太小；chosen/rejected 太像；标注噪声 |
| 训练集 acc 很高但 RL 阶段变差 | RM-policy **分布偏移（OOD）**；需要 KL 约束、数据覆盖（L09） |
| rejected 太蠢 | RM 过拟合「简单模式」，对新错误不敏感 |

---

## 12. ASCII：奖励模型训练流程

```
  ┌─────────────┐     ┌─────────────────────────────────────┐
  │  SFT 模型    │     │  偏好数据 JSONL                      │
  │  (骨干初始化)│     │  system / history / question         │
  └──────┬──────┘     │  + response_chosen / response_rejected │
         │            └──────────────────┬────────────────────┘
         │                               │
         v                               v
  ┌──────────────────────────────────────────────────────────┐
  │  SequenceClassification 骨干 (num_labels=1)               │
  │  + 可选 LoRA (SEQ_CLS)                                      │
  └───────────────────────────┬──────────────────────────────┘
                              │
         for each batch:      │
         ┌────────────────────┴────────────────────┐
         v                                          v
  forward(chosen)                            forward(rejected)
         |                                          |
         v                                          v
     r_chosen                                   r_rejected
         └────────────────┬───────────────────────┘
                          v
              loss = -log sigmoid(r_chosen - r_rejected)
                          |
                          v
                   反向传播更新 φ
                          |
                          v
              保存 RM → L09 作为 reward 使用
```

---

## 13. 面试高频题与速记卡片

### Q1：奖励模型的 loss 是什么？

> **成对排序的负对数似然**：\(\mathcal{L}=-\mathbb{E}\log\sigma(r_{\text{chosen}}-r_{\text{rejected}})\)。实现上用 **`logsigmoid`**。

### Q2：为什么用对比方式训练？

> 人类标注提供 **相对序** 而非绝对分数；BT 模型把偏好概率写成 **奖励差分的 sigmoid**，与数据形态一致，并对 **标注尺度漂移** 更鲁棒。

### Q3：RM 输出是什么？能当客观真理吗？

> 标量 **偏好信号**；只在训练分布与标注规则下有意义。**不是**临床诊断结论。

### Q4：RM 过拟合会怎样？

> 强化学习阶段可能出现 **reward hacking**：策略钻 RM 漏洞拿高分但人类觉得差；需要数据多样性、KL 约束、迭代修复（L09）。

### Q5：为什么常见 `num_labels=1`？

> 输出 **单个 logit** 作为奖励；多分类头在此任务不常见。

### Q6：chosen/rejected 需要长度接近吗？

> 不强制，但若数据里 chosen **系统性更长**，RM 可能学到 **长度偏见**；应用层要监控。

---

## 14. 小结、自测与延伸阅读

### 14.1 三句话背下来

1. RM 把 **成对偏好** 学成 **标量打分函数** \(r_\phi(x,y)\)。  
2. 架构上是 **序列分类 + 单 logit**，训练目标是 **拉大 chosen 与 rejected 的 margin**。  
3. MedicalGPT 用 **自定义 Trainer** 实现 InstructGPT 同款的 **pairwise logsigmoid loss**。

### 14.2 自测（建议闭卷）

1. 写出 BT 偏好概率公式。  
2. 解释为何 RM 通常从 SFT 模型初始化。  
3. 打开本地 `reward_modeling.py`，指出 `compute_loss` 相关段落。  
4. 说明 pairwise accuracy 的定义。  
5. 若 loss 长期约 0.693，你会先排查哪三类原因？

### 14.3 延伸阅读

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)  
- [MedicalGPT 源码](https://github.com/shibing624/MedicalGPT)

### 术语小抄

- **Pairwise preference**：同一上下文两条回答，只标相对好坏。  
- **Bradley-Terry**：把「A 优于 B 的概率」写成奖励分数差的 sigmoid。  
- **Reward hacking**：刷奖励但不提升真实质量。  
- **OOD**：RM 训练分布与线上生成分布不一致，分数不可靠。

---

*文档与 MedicalGPT 开源脚本保持概念对齐；上游更新参数名时，请以仓库为准并在笔记中记录 diff。*
