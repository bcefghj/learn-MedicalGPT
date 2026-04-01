[← 上一课](../L05-增量预训练PT/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L07-LoRA与QLoRA高效微调/README.md)

---

# L06 有监督微调（Supervised Fine-Tuning, SFT）

> **一句话精髓**：*「读完书还不够，得做题才会答题。」*

在 **L05** 里，模型通过 **增量预训练（PT）** 大量「读」医学文本，更像一个 **熟悉医学语言的阅读者**。但现实产品需要的是：**听得懂用户问题、按格式回答、遵守安全边界** 的 **助手**。这一步主要靠本课的 **有监督微调（SFT）** 完成。

### 0.1 类比：从「图书馆」到「标准化考试培训班」

```
  L05 PT                L06 SFT
  -------               -------
  随便翻开一页书        拿到一张「标准试卷」
  猜下一个字            在固定题干下写出「标答」
  没人告诉你「题型」    每题都有人类写的示范答案
```

你不需要在 SFT 阶段 **发明医学知识**（那是数据与 RAG 的事），你要做的是：让模型学会 **在提示（prompt）约束下**，**稳定地产出** 你期望的 **语气、结构、安全边界与格式**。

### 0.2 本课唯一必须建立的「肌肉记忆」

SFT 仍然用 **自回归下一词损失**，但 **loss 往往只打在 assistant 段**——等价于告诉模型：

> 「题干随便看看就行，但 **答案里的每一个字** 你都要负责说对。」

这句话在面试里 **复述一遍**，分数通常不会低。

---

## 本课你将学会什么

1. 说清楚 **微调** 与 **有监督微调（SFT）** 的定义与边界。  
2. 用一张对比表讲透 **SFT vs PT 的本质区别**（数据、标签、目标、产物）。  
3. 掌握 **Alpaca 风格** 与 **ShareGPT 多轮** 两种常见数据格式。  
4. 了解 MedicalGPT 使用的 **中文医疗指令数据规模**（约 240 万条量级，以官方数据集说明为准）。  
5. 理解 **`template.py` 模板机制**：`vicuna` / `alpaca` / `chatglm` / `qwen` 等如何把对话渲染成训练串。  
6. 能对照 `supervised_finetuning.py` 解释 **labels mask**、`train_on_inputs`、与 **Trainer** 流程。  
7. 掌握 **全参数微调 vs PEFT（LoRA/QLoRA）** 的选型思路。  
8. 理解 **灾难性遗忘** 与常见缓解手段；理解 **Base vs Instruct(Chat)** 模型选型。  
9. 回答面试题：**SFT 数据要多少条？Epoch 设几个？**

---

## 1. 什么是微调（Fine-tuning）

### 1.1 直觉类比

预训练像 **通识教育**；微调像 **岗前培训**：

- 通识教育让你 **识字、懂语法、有常识**。  
- 岗前培训教你 **公司的话术、流程、回复模板**。

**微调** 泛指：在已有参数初始化下，用 **更小、更专门** 的数据集继续训练，使模型适配 **下游任务或风格**。

### 1.2 技术角度

微调通常意味着：

- **起点**：预训练权重 \(\theta_0\)。  
- **数据**：任务相关样本 \(\{(x_i, y_i)\}\) 或对话轨迹。  
- **优化**：在较小学习率下更新 **全部或部分参数**，最小化任务损失。

---

## 2. 什么是有监督微调（SFT）

### 2.1 「有监督」指什么

**监督信号来自标注**：人类（或高质量流水线）给出「在提示 \(x\) 下，理想输出 \(y\)」。

对生成式大模型，常见做法是：把 \((x, y)\) 拼进 **对话模板**，仍用 **自回归下一词损失** 训练，但对 **提示部分** 的 token **不计算 loss**（label 设为 `IGNORE_INDEX`），只对 **回答部分** 计算 loss。

### 2.2 SFT 在 RLHF 流水线中的位置

经典叙事：

```
PT（可选） → SFT（行为克隆，学「像人那样答」） → RM + PPO 或 DPO（偏好对齐）
```

SFT 常被称为 **Behavior Cloning（行为克隆）** 的第一步：模仿示范回答。

---

## 3. SFT vs PT 的本质区别

### 3.1 一张表背下来（面试够用）

| 维度 | PT（增量预训练） | SFT（有监督微调） |
|------|------------------|-------------------|
| 数据形态 | 纯文本、非结构化 | 指令-输入-输出 / 多轮对话 |
| 标注 | 自监督（下一词即标签） | 显式示范（人类写答案） |
| 主要目标 | 拟合领域语言分布 | 拟合「提示→期望回答」映射 |
| 典型 loss | CLM 全序列（除 pad） | CLM 但 **mask 掉 prompt** |
| 能力侧重 | 术语、风格、知识统计 | 指令遵循、对话格式、安全话术 |
| 失败表现 | 不像「医学文」 | 不听指令、答非所问、格式乱 |

### 3.2 一句话总结

- **PT**：让模型 **更会读医学话**。  
- **SFT**：让模型 **更会按人类规矩答医学题**。

---

## 4. 指令数据格式（一）：instruction / input / output（Alpaca 系）

### 4.1 常见 JSON 形态

单轮任务常用字段：

```json
{
  "instruction": "患者出现胸痛，可能原因有哪些？",
  "input": "男，55岁，高血压病史",
  "output": "胸痛病因较多，需结合病史与体征鉴别……（示范回答）"
}
```

- **`instruction`**：任务描述。  
- **`input`**：可选上下文；没有可置空或省略。  
- **`output`**：希望模型模仿的回答。

### 4.2 在训练时发生什么

模板把字段渲染成 **多段文本**（不同模型前缀不同），再 token 化。MedicalGPT 的 `supervised_finetuning.py` 更偏向 **ShareGPT 对话结构** + `template.get_conv_template`，但社区数据常混用 Alpaca 格式，需做转换。

---

## 5. 指令数据格式（二）：ShareGPT 多轮对话

### 5.1 结构直觉

ShareGPT 风格强调 **角色交替**：`human` / `gpt`（或类似别名），支持多轮。

```json
{
  "conversations": [
    {"from": "human", "value": "什么是糖尿病？"},
    {"from": "gpt", "value": "糖尿病是一组以慢性高血糖为特征的代谢性疾病……"},
    {"from": "human", "value": "有什么典型症状？"},
    {"from": "gpt", "value": "典型表现为多饮、多食、多尿和体重下降……"}
  ]
}
```

### 5.2 `supervised_finetuning.py` 里的关键约定

源码中 `roles = ["human", "gpt"]`，并会：

- 若首条是 `system`，抽出 **system prompt**。  
- 校验角色交替，组装成 `messages` 再两两配对。  
- 通过 `prompt_template.get_dialog(history_messages, system_prompt=...)` 得到 **渲染后的 dialog 片段列表**。

### 5.3 为什么要多轮数据

真实问诊常是 **追问型**；多轮 SFT 让模型学会：

- **承接上文**  
- **不重复啰嗦**  
- **分步骤澄清**（在数据示范足够好的前提下）

---

## 6. MedicalGPT 的 SFT 数据：240 万条中文医疗数据（量级认知）

主仓库 README / Wiki 与 HuggingFace 数据集页将医疗相关语料描述为 **大规模中文医疗数据**（常见宣传口径为 **约 240 万条** 指令数据，具体以 [shibing624/medical](https://huggingface.co/datasets/shibing624/medical) 说明为准）。

### 6.1 你应该记住的「不是数字本身」

面试时更有价值的是：

- **数据来自哪里**（公开集合、清洗规则、是否脱敏）。  
- **字段格式** 与 **模板** 是否匹配。  
- **质量筛选**（去重、去空、去有害、去幻觉示范）。

### 6.2 数据质量的「一票否决项」

1. **错误医学事实**的示范回答（模型会学错）。  
2. **自相矛盾**的多轮对话。  
3. **过度承诺**（包治百病类话术）。

---

## 7. Template 模板机制（vicuna / alpaca / chatglm / qwen）

### 7.1 为什么需要模板

不同基座在预训练/对齐阶段见的 **对话格式** 不同，例如：

- 是否有 `USER:` / `ASSISTANT:`  
- 是否用 **特殊 token** 包裹轮次  
- system 字段放哪里

**模板**就是把结构化对话 **编译成模型最熟悉的字符串**。

MedicalGPT 通过 `from template import get_conv_template` 与 `--template_name` 选择模板。

### 7.2 选型经验（口语版）

| 基座家族 | 常见 template_name |
|----------|---------------------|
| Vicuna / LLaMA 对话系 | `vicuna` |
| Alpaca 系数据兼容 | `alpaca` |
| ChatGLM 系 | `chatglm` |
| Qwen Instruct 系 | `qwen` |

**原则**：`template_name` 应与你加载的 **Instruct/Chat 模型** 的训练格式 **尽量一致**，否则等于 **用错键盘布局打字**。

### 7.3 `run_sft.sh` 中的示例

官方示例使用：

```bash
--model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
--template_name qwen \
```

这就是在说：**用 Qwen 对话格式去渲染训练串**。

---

## 8. `supervised_finetuning.py` 核心逻辑（对照读源码）

> 源码以 [MedicalGPT `supervised_finetuning.py`](https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py) 为准。下面用 **零基础可跟读** 的方式，把「数据怎么进、loss 怎么出」讲成一条故事线。

### 8.0 故事线版（读代码前先读这段）

1. **扫目录**：`train_file_dir` 下所有 `json/jsonl` 被拼成 `datasets` 的 `train` split。  
2. **逐条样本**：每条里若有 `conversations`，就走 **ShareGPT 风格**；脚本会把 `human/gpt` 轮次 **校验、配对**，必要时抽出 `system`。  
3. **模板渲染**：`get_conv_template(template_name)` 决定 **分隔符、角色名、特殊标记**；`get_dialog` / `get_prompt` 把多轮变成 **一串 token 可吃的文本块**。  
4. **切段与预算**：每一轮拆成 `source`（给模型看的上文）与 `target`（模型该生成的下文）；若超长，按 **比例切分** 并保留 **EOS** 语义（细节以源码为准）。  
5. **造 labels**：默认 `train_on_inputs=False` → **prompt 段整段 `IGNORE_INDEX`**，只在 **回答段** 保留真实 token id 作为监督。  
6. **过滤**：若某条样本 labels 全被 mask，**丢弃**（否则这一步梯度为 0，浪费算力还可能扰乱日志）。  
7. **Trainer**：`DataCollator` 做 padding 对齐；`forward` 里算 CLM loss；**optimizer 只通过未被 ignore 的位置** 更新参数（或等价实现）。

### 8.1 总流程 ASCII 图

```
  json/jsonl 数据（train_file_dir）
              │
              ▼
        load_dataset("json")
              │
              ▼
   preprocess_function（map）
     ├─ get_dialog：ShareGPT → 多轮文本对
     ├─ tokenizer.encode 拼接 source/target
     ├─ 控制总长 ≤ model_max_length
     └─ labels：prompt 部分置 IGNORE_INDEX（默认）
              │
              ▼
   filter_empty_labels（去掉全 mask 样本）
              │
              v
   AutoModelForCausalLM + (可选 LoRA/QLoRA)
              │
              v
   Trainer.train()
```

### 8.2 `train_on_inputs` 的含义（高频考点）

- **`train_on_inputs=False`（默认）**：**只对 assistant 回答部分算 loss**，prompt 部分 label 为 `IGNORE_INDEX`，loss 忽略。  
- **`train_on_inputs=True`**：prompt + 回答 **都训练**。适用于某些特殊场景（例如也要强化「提问方式」），但更常带来 **过拟合提示** 或 **泄露数据偏见**。

### 8.3 长度分配技巧（源码细节）

对每一轮，`source_ids` 与 `target_ids` 的总长若超过预算，会按 **各自原始占比** 分配 `max_source_len` 与 `max_target_len`，并保留 **EOS** 行为。目的是在 **有限上下文** 内尽量 **两端都保住一部分**。

### 8.4 评估集过大时的警告

脚本会对 `eval` 样本过多发出 warning，建议 `--max_eval_samples` 减小，否则 **验证极慢**。

### 8.5 Collator 与 LabelSmoother

使用 `DataCollatorForSeq2Seq` 风格处理（见源码 import），`IGNORE_INDEX` 默认取 `LabelSmoother.ignore_index`，用于在 loss 里 **屏蔽不需要学习的位置**。

---

## 9. 关键参数详解

### 9.1 `per_device_train_batch_size`

每张 GPU 每次迭代的样本数。SFT 数据序列更长，**batch 往往比 PT 更小**。

### 9.2 `gradient_accumulation_steps`

与 L05 相同：用时间换「大 batch 效果」。有效 batch：

\[
B_{\text{eff}} \approx \text{per\_device} \times N_{\text{gpu}} \times \text{accum}
\]

### 9.3 `learning_rate`

SFT 常用 **比 PT 更小的 LR**（全参微调尤其要小）。`run_sft.sh` 示例为 `2e-5` 量级（配合 LoRA 时仍属常见范围；**全参**可能要再降）。

**口诀**：**越接近「成品模型」，步子越小**。

### 9.4 `num_train_epochs`

- 数据 **少而精**：1～3 epoch 也可能够。  
- 数据 **大而杂**：1 epoch 或更少 + 强正则/早停。  
- **过拟合信号**：训练 loss 继续降，验证集对话质量 **变差**、复读机、胡编。

### 9.5 `model_max_length`

MedicalGPT `ScriptArguments.model_max_length`：单条样本最大 token。对齐 **显存** 与 **任务上下文**（问诊若长病历，需更大，但要硬件撑得住）。

`run_sft.sh` 示例：`4096`。

---

## 10. 全参数微调 vs PEFT 微调

### 10.1 全参数微调（Full Fine-tuning）

- **更新**：所有权重。  
- **优点**：上限高，彻底重塑行为空间大。  
- **缺点**：显存、时间、存储成本高；更易 **灾难性遗忘**。

### 10.2 PEFT（LoRA / QLoRA 等）

- **更新**：少量增量参数（+ 可选部分模块）。  
- **优点**：便宜、快、可插拔、可多版本并存。  
- **缺点**：极难任务或格式剧变时，**容量**可能不够。

**MedicalGPT 默认脚本**常设 `--use_peft True`，对学习者友好。

---

## 11. 训练命令示例：`run_sft.sh` 解析

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 supervised_finetuning.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_file_dir ./data/finetune \
  --validation_file_dir ./data/finetune \
  --template_name qwen \
  --use_peft True \
  --model_max_length 4096 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --flash_attn True
```

**解读要点**：

1. 选 **Instruct 基座**：已经会对话，SFT **更稳、更快收敛**。  
2. `template_name` 与基座家族匹配。  
3. `model_max_length` 与数据长度分布一致（过短会截断丢信息）。  
4. `flash_attn` 在环境支持时可显著加速（需安装 flash-attn）。

---

## 12. SFT 最佳实践：数据质量 > 数据数量

### 12.1 为什么质量优先

SFT 的梯度 **非常直接**：你写什么，它就 **往什么方向模仿**。错误示范会被 **高强度记忆**。

### 12.2 质量工程检查表

```
□ 医学事实：是否经过专业人员或权威来源校验？
□ 语气：是否安全、谨慎、避免绝对承诺？
□ 格式：是否与模板渲染后仍清晰？
□ 多样性：是否覆盖拒答、追问、无法判断？
□ 去重：是否去掉重复问答对？
□ 毒性：是否过滤歧视、隐私、违法内容？
```

### 12.3 数量与覆盖

在质量合格前提下，**覆盖关键意图**（主诉、鉴别、用药解释、检查解读、就医建议边界）比 **单纯堆百万条同质问答** 更有用。

---

## 13. 灾难性遗忘（Catastrophic Forgetting）

### 13.1 现象

模型在专注新任务后，**旧能力掉点**：通用推理变弱、其他领域问答变差、甚至 **格式能力回退**。

### 13.2 常见缓解

| 方法 | 思路 |
|------|------|
| 混合通用数据 | SFT 里掺入高质量通用指令 |
| 更小 LR / 更少 epoch | 减小对原分布的「拉扯」 |
| LoRA | 限制更新子空间 |
| 回放（replay） | 周期性加入旧任务样本 |
| 多阶段训练 | 先域内 SFT，再轻量通用修复（需谨慎设计） |

### 13.3 面试答法

**遗忘不是「坏了」，而是参数被优化到新任务的局部最优**。缓解核心是 **约束更新** 或 **保留旧任务信号**。

---

## 14. Base 模型 vs Chat/Instruct 模型：选择策略

### 14.1 Base 模型

- **特点**：更像「续写机」，不天然遵循对话指令。  
- **适用**：你 **有大量高质量对话 SFT**，想从头塑造对话形态；或做研究对比。

### 14.2 Instruct/Chat 模型

- **特点**：已对齐过对话格式与安全偏好。  
- **适用**：**医疗场景 SFT** 的大多数工程起点：更快、更稳。

### 14.3 MedicalGPT 示例脚本的暗示

`run_sft.sh` 使用 `Qwen2.5-0.5B-Instruct`：**实战优先选 Instruct**。

### 14.4 组合策略（经验）

```
若你数据少但质量极高：Instruct + LoRA（最常见）
若你数据极大且想重塑行为：可评估 Base + 全参（成本高）
若 PT 后接 SFT：PT 后权重 + Instruct 模板需一致化验证
```

---

## 15. ASCII 图：SFT 数据流与训练流程

### 15.1 数据流

```
 ShareGPT / 类对话 JSON
        │
        ▼
  get_conv_template(name)
        │
        ▼
 「渲染串」= prompt + answer 片段交替
        │
        ▼
 tokenize → input_ids
        │
        ▼
 labels：prompt 位置 = IGNORE
         answer 位置 = 真实 token
```

### 15.2 训练闭环

```
   ┌─────────────┐
   │  一个 batch   │
   └──────┬──────┘
          v
   ┌─────────────┐
   │  Forward CLM │
   └──────┬──────┘
          v
   ┌─────────────┐
   │ Loss 只在    │
   │ answer 区间  │
   └──────┬──────┘
          v
   ┌─────────────┐
   │  Backward    │
   └──────┬──────┘
          v
   ┌─────────────┐
   │  Optimizer   │
   └─────────────┘
```

---

## 16. 面试高频题

### Q1：SFT 数据需要多少条？

**标准答法**：**没有魔法数字**，取决于：

- 任务复杂度（单轮 vs 多轮、是否工具调用）  
- 数据噪声水平  
- 基座能力（Instruct 更强则更少）  
- 是否 PEFT

**可举例**：医疗场景常见 **数千到数十万** 高质量即可做出可用原型；**百万级** 往往需强清洗。声称「必须 240 万」不如强调 **质量与覆盖**。

### Q2：Epoch 设几个？

**答法**：

- 先从 **1 epoch** 与小验证集定性看生成质量。  
- 小数据可 **2～3 epoch**；大数据常 **1 epoch 或更少**。  
- 以 **验证集 loss + 人工抽检** 为准，避免过拟合。

### Q3：为什么 SFT 还要用 CLM loss？

**答法**：自回归模型天然输出 token 序列；**对整个回答串做下一词预测** 等价于 **最大化回答的条件概率**，与生成推理一致。

### Q4：SFT 能不能解决所有安全问题？

**答法**：**不能**。SFT 是必要基础，但对抗性输入、越狱、长链诱导仍需 **对齐（DPO/RLHF）**、**系统层防护**、**RAG 溯源** 等组合。

---

## 17. 小型「手写样本」示例（理解 mask）

假设渲染后：

- Prompt token：`[P1, P2, P3]`  
- Answer token：`[A1, A2, A3, EOS]`

则 `labels` 可能是：

```
[IGNORE, IGNORE, IGNORE, A1, A2, A3, EOS]
```

模型在答句位置计算 loss，从而学会 **在给定 prompt 下生成答句**。

### 17.1 再举一例：多轮对话的两段 loss（直觉）

假设两轮对话渲染后拼接（简化）：

```
[System...] [User1...] [Asst1...] [User2...] [Asst2...]
```

默认 `train_on_inputs=False` 时，**常见策略**是：**只有 Asst1、Asst2 段** 参与 loss；User/System 段 **全部 IGNORE**。  
直觉：你不希望模型在 SFT 里「学习怎么当用户提问」（除非你的数据刻意设计过），你希望它学 **怎么当助手回答**。

### 17.2 `train_on_inputs=True` 什么时候会想开

- 你想让模型 **强烈记住某种提问风格**（例如病历采集的标准问法）。  
- 你做 **续写预训练式** 的混合实验（少数团队会这么做）。  
**默认不要轻易开**：很容易把 **数据里的偏见提问** 也学成「标准」。

---

### 17.3 与 PT 的「同构不同质」（进阶对照）

PT 的 `labels` 往往 **整段都是监督信号**（除 padding）；SFT 则 **人为挖掉一大段 prompt 的监督**。  
因此：

- **同样下降的训练 loss**，语义不同：SFT 的 loss **只反映答句难度**。  
- **不要拿 PT 的 loss 数值与 SFT 横向比较**。

---

## 18. 常见踩坑

1. **模板与基座不匹配**：loss 能下降，但推理时 **格式全乱**。  
2. **验证集太大**：训练一天，eval 占一半。  
3. **全是短问答**：长病历上下文 **泛化差**。  
4. **把错误答案当金标准**：模型会 **自信地错**。  
5. **`max_train_samples` 忘记关掉**：上线发现只训了 1000 条（脚本里常有调试默认值）。

---

## 19. 与 L05 的衔接练习（建议口述）

用 60 秒回答：

> 我们团队已完成医学 PT，为什么还需要 SFT？能否只做 PT 上线？

**参考要点**：PT 不教对话与指令遵循；产品需要结构化交互与安全表达 → 需要 SFT；只做 PT 更像「续写引擎」，用户体验与可控性差。

---

## 20. 推荐阅读

1. [MedicalGPT](https://github.com/shibing624/MedicalGPT) — `supervised_finetuning.py`、`template.py`、`run_sft.sh`。  
2. [HuggingFace SFT 概念](https://huggingface.co/docs/trl/main/en/sft_trainer)（生态工具参考）。  
3. [Zephyr / UltraChat 数据讨论](https://arxiv.org/) — 理解「对话数据如何影响行为」（搜关键词即可）。  
4. 医疗数据页面：[HuggingFace medical dataset](https://huggingface.co/datasets/shibing624/medical)。

---

## 21. 课后自检清单

- [ ] 我能解释 **为什么要 mask prompt**。  
- [ ] 我能描述 **ShareGPT 多轮** 如何进 `get_dialog`。  
- [ ] 我能比较 **PT vs SFT** 的数据与目标。  
- [ ] 我知道 **Instruct 基座** 为什么是默认推荐起点。  
- [ ] 我能回答 **epoch 与数据量** 没有固定答案，要看验证与抽检。

---

## 22. 术语表

| 中文 | 英文 | 说明 |
|------|------|------|
| 有监督微调 | SFT | 用示范输入输出继续训练 |
| 行为克隆 | BC | 模仿示范轨迹/回答 |
| 提示 | Prompt | 模型可见的上文指令部分 |
| 掩码标签 | Label mask | 不计 loss 的位置 |
| 灾难性遗忘 | Catastrophic Forgetting | 新任务挤压旧能力 |

---

## 23. 扩展：SFT 与 RAG 的分工（预告 L17）

SFT 教 **表达方式与任务流程**；RAG 提供 **可更新外部知识**。二者互补：

- **事实易变**（指南更新）→ RAG。  
- **话术与流程**（如何问清过敏史）→ SFT。

---

## 24. FAQ

**问：能不能直接用 ChatGPT 生成 SFT 数据？**  
要谨慎：需 **强校验**，否则会把 **幻觉风格** 教进模型。

**问：SFT 需要 RLHF 吗？**  
不强制，但上线产品常再做 **DPO/RLHF** 提升「哪个回答更好」。

**问：多轮数据怎么保证角色不错乱？**  
靠数据规范与模板；训练前 **decode 打印** 几条样本最有效。

---

## 25. 和 MedicalGPT Wiki 的对齐建议

实操时养成习惯：

1. 先 **decode 看渲染**；  
2. 再 **小样本过拟合**（例如 100 条训到能背）；  
3. 最后 **全量 + 验证**。

这是工业界常用的 **三段式 sanity check**。

---

## 26. `run_sft.sh` 超参数与「显存/效果」三角关系（零基础版）

把下面当成 **调参地图**，而不是死规则：

```
                    想把上下文拉长
                           |
                           v
                 model_max_length ↑
                           |
            +--------------+--------------+
            |                             |
    per_device_batch ↓              gradient_checkpointing ↑
            |                             |
            +--------------+--------------+
                           |
                 flash_attn / bf16 / LoRA
```

- **先固定 template 与数据**，再动学习率；**同时改太多变量** 会导致无法归因。  
- **有效 batch** 不足时，梯度噪声大，SFT 可能 **欠拟合**；过大则可能 **过拟合小数据**。  
- **验证集** 一定要有 **人工抽检**（医疗场景尤甚）：loss 好看但 **胡编** 很常见。

---

## 27. 口述题：给非技术同事讲 SFT（30 秒版）

「我们先让大模型看很多 **问答示范**，告诉它：在用户这么问的时候，你应该这么答。训练时模型要 **逐字预测标准答案**；但我们 **不让它学习怎么重复用户的问题**，只让它学习 **怎么生成回答**，这样上线更像助手。」

---

## 28. 扩展阅读：SFT 数据构造的常见「工程配方」

| 配方 | 适用 | 风险 |
|------|------|------|
| 医生撰写 + 双人复核 | 质量最高 | 贵、慢 |
| 模型生成 + 人工修订 | 规模化 | 生成幻觉需过滤 |
| 指南/说明书改写为 QA | 事实较稳 | 语气可能刻板 |
| 真实对话脱敏 | 贴近场景 | 合规与清洗成本 |

**面试加分句**：「我更关心 **标注协议、复核流程、拒答与边界案例覆盖**，而不是单纯条数。」

---

## 29. 与本课相关的「安全」底线（医疗向）

SFT 会 **放大** 数据里的行为：

- 数据里爱 **下诊断**，模型就敢 **下诊断**。  
- 数据里爱 **给具体药量**，模型就敢 **给具体药量**。  

工程上常配合：**系统提示词约束**、**拒答模板**、**RAG 引用来源**、**后续 DPO/RLHF**。本课先把 **SFT 模仿性** 记牢：**它不会自动变得谨慎，谨慎要靠数据与流程**。

---

## 30. 一页纸复盘（建议抄在笔记本上）

```
SFT = 在预训练权重上，用「指令-回答」数据继续训练
损失  = 自回归下一词，但 prompt 常不算损失
数据  = Alpaca 字段 或 ShareGPT 多轮
模板  = template_name 必须和基座家族匹配
基座  = 工程上优先 Instruct/Chat
调参  = 有效 batch、LR、epoch 三者联动
遗忘  = 混合通用数据 / 小步更新 / LoRA / 回放
```

---

**下一课预告（L07）**：显存不够全参数？**LoRA / QLoRA** 让你在 **只训练少量参数** 的情况下，把 SFT 跑起来——就像 **在厚书上贴便签**，而不是 **重写整本书**。

---

[← 上一课](../L05-增量预训练PT/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L07-LoRA与QLoRA高效微调/README.md)
