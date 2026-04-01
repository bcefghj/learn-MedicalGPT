[← 上一课](../L04-MedicalGPT项目全景/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L06-有监督微调SFT/README.md)

---

# L05 增量预训练 PT（Continue Pre-training）

> **一句话精髓**：*「先让模型读一千万篇医学论文。」*

你已经学完 **L01～L04**：知道大模型是什么、Transformer 在干什么、环境怎么搭、MedicalGPT 仓库里 **PT / SFT / DPO** 各管哪一段。从本课开始，我们进入 **真正改参数的训练**：第一站是 **增量预训练（PT）**——让通用大模型在 **海量医疗文本** 上继续「读书」，把 **语言习惯与统计规律** 拉进医学世界。

---

## 本课你将学会什么

1. 用 **非技术语言** 说清：什么是 **预训练（Pre-training）**。  
2. 区分 **增量预训练** 与 **从头训练**，并理解 **成本与风险** 的差异。  
3. 解释 **为什么** 要把「通用模型」推进成「领域模型」。  
4. 掌握 PT 的 **训练目标**：**Next Token Prediction（下一词预测）** 与 **CLM**。  
5. 知道 PT 数据长什么样：**非结构化医疗文本**（txt / jsonl 等）。  
6. 能对照 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 的 **`pretraining.py`** 口述 **主流程**。  
7. 理解 **`run_pt.sh`** 里每个关键参数在 **显存与收敛** 上的作用。  
8. 把 **`tokenize → group_texts(packing) → batch → loss`** 串成一条线。  
9. 判断 **什么时候可以跳过 PT**，避免「为训而训」。  
10. 背下 **PT vs SFT** 的面试答法骨架。

---

## 0. 写给「零基础但已学完前 4 课」的你：本课在 Pipeline 里的位置

把 MedicalGPT 想象成培养一位 **医学生**：

```
  ┌──────────────────────────────────────────────────────────────┐
  │  L05 PT：图书馆阶段 —— 狂读教材、论文、病历叙述（不一定有问答）   │
  └────────────────────────────┬─────────────────────────────────┘
                               v
  ┌──────────────────────────────────────────────────────────────┐
  │  L06 SFT：习题阶段 —— 给「题目+标准答案」，学会按格式作答        │
  └────────────────────────────┬─────────────────────────────────┘
                               v
  ┌──────────────────────────────────────────────────────────────┐
  │  L07+ 对齐：价值观与偏好 —— 哪个答案更好、更安全               │
  └──────────────────────────────────────────────────────────────┘
```

**PT 不负责** 教模型说「我是医疗助手」；它负责让模型在 **续写医学文本** 时 **更像真的医学语料**：术语、缩写、句式、常见搭配、知识共现统计等。**对话与指令遵循** 主要在 **SFT** 解决——但 PT 常常能 **降低 SFT 难度**（模型更少「医学外行腔」）。

---

## 1. 什么是预训练（Pre-training）

### 1.1 生活类比：先学会「中文」，再学「科室术语」

- **预训练**阶段，模型读的是 **互联网级** 的混合文本：新闻、百科、论坛、书籍片段……  
- 它学到的是 **通用语言能力**：语法、常识、推理骨架、世界知识（嘈杂但广）。  
- 类比：一个人 **先读完九年义务教育教材**，识字、造句、读文章都没问题；但 **还不会写病程记录、不会背指南原文的口吻**。

### 1.2 技术一句话

**预训练** = 在大规模无标注文本上，用 **自监督目标**（最常见是 **下一词预测**）更新模型参数，使模型 **逼近真实文本的联合概率分布**。

### 1.3 自监督是什么意思（面试常问）

没有人类逐句标注「正确输出」，**标签来自文本自身**：  
当前词序列已知，模型要预测 **下一个 token**；**正确答案就是序列里的下一个 token**。所以叫 **自监督**。

---

## 2. 什么是增量预训练（Continue Pre-training）vs 从头训练

### 2.1 从头训练（Train from Scratch）

- **含义**：随机初始化（或简单初始化），用海量语料把 **全部参数** 练出来。  
- **特点**：算力与数据需求 **极大**；只有巨头或极特殊场景才现实。  
- **类比**：不从任何学校毕业，**从造字开始** 培养语言专家——理论上可行，工程上几乎不划算。

### 2.2 增量预训练（Continue Pre-training / Domain-adaptive Pre-training）

- **含义**：拿 **已经预训练好的通用大模型** 当初始点，在 **领域语料**（如医学书籍、论文摘要、脱敏病历叙述）上 **继续** 做同样的 **下一词预测** 训练。  
- **特点**：**站在巨人肩膀上**；成本远低于从头训；风险是 **分布偏移过大** 或 **数据脏** 时可能 **伤基座**。  
- **类比**：通识教育已完成，现在 **转专业读医学系教材**——不是重新认字，而是 **大量阅读专业语料**。

### 2.3 对比表（建议收藏）

| 维度 | 从头训练 | 增量预训练（本课 PT） |
|------|----------|------------------------|
| 初始化 | 随机/弱先验 | 强先验（通用 LLM） |
| 数据量需求 | 极大 | 大即可（仍要讲究质量） |
| 算力 | 极高 | 中高（可用 LoRA/QLoRA 降本） |
| 典型目标 | 造「基座」 | 把基座 **拉进某领域文本分布** |
| 失败表现 | 不收敛/胡言乱语 | 过拟合小语料、学脏数据、通用能力下降 |

---

## 3. 为什么需要增量预训练？（通用模型 → 领域模型）

### 3.1 问题从哪来

通用模型在公开互联网上训练，**医学高频表述** 未必足够：

- 指南式句式：「首选…」「禁忌证包括…」  
- 缩写与符号：「CrCl」「HbA1c」「NYHA 分级」  
- 中文医疗书写习惯与 **口语问诊** 的差异  

**SFT** 可以教「怎么答」，但若基座对 **领域语言与实体共现** 不熟，SFT 要 **硬背更多表面模式**，数据效率差，且长文本续写/填空类任务容易 **露怯**。

### 3.2 PT 带来的典型收益（口语版）

- **更顺的医学表达**：续写更像真实语料。  
- **更好的实体与搭配统计**：不是「真理解」，但 **预测更贴域**。  
- **给 SFT「减负」**：后续指令微调更容易收敛。

### 3.3 什么时候 PT 收益不明显（避免过度期待）

- 你用的基座 **已经极强** 且 **中文医疗覆盖已经很好**。  
- 下游只做 **短问答**，且 **RAG** 提供全部事实。  
- 领域语料 **又少又脏**——不如先洗数据或只做高质量 SFT。

---

## 4. 训练目标：Next Token Prediction（下一词预测）

### 4.1 直觉：手机输入法的「联想下一词」超级版

输入法根据已输入字词猜下一个字；LLM 根据 **前面所有 token** 预测 **下一个 token 的分布**。训练时，用 **真实出现的下一个 token** 当标签，算 **交叉熵损失**，反向传播更新参数。

### 4.2 符号速写（面试够用）

给定 token 序列 \(x_1, x_2, \ldots, x_T\)，模型学习最大化：

\[
\prod_{t=1}^{T-1} P(x_{t+1} \mid x_1,\ldots,x_t)
\]

**Causal（因果）** 的含义：预测 \(x_{t+1}\) 时 **只能看 \(x_1\) 到 \(x_t\)**，不能偷看未来——这与 GPT 类架构的 **掩码自注意力** 一致，叫 **Causal Language Modeling（CLM）**。

### 4.3 PT 与 SFT 在「目标函数形式」上的关系（重要）

二者 **常常是同一类 CLM loss**；差别主要在 **数据形态与 label 掩码**：

- **PT**：整段连续文本，**除 padding 外** 大多位置都算 loss。  
- **SFT**：往往 **mask 掉用户提示**，只对 **助手回复** 算 loss（L06 细讲）。

---

## 5. 训练数据：非结构化医疗文本

### 5.1 「非结构化」指什么

不是已经整理成「问 / 答」字段，而是 **自然语言段落**：书籍章节、论文段落、百科条目、脱敏后的病历叙述、指南正文等。存储形态常见为：

- **纯文本 `.txt`**：一行一段或一篇一文件（具体以你清洗规则为准）。  
- **JSON / JSONL**：常有一个 **`text`** 字段存正文（MedicalGPT 的 `load_dataset` 逻辑会找 `text` 或第一列）。

### 5.2 数据质量红线（比「一千万篇」更重要）

1. **隐私**：未脱敏真实姓名、电话、地址 → 一票否决。  
2. **错误医学知识**：模型会 **自信地复读错误**。  
3. **非医疗噪声**：大量广告、乱码、重复爬取，会 **稀释有效梯度**。  

### 5.3 规模直觉

「一千万篇」是 **数量级直觉**：强调 **多读**；真实工程要 **看 token 总量、重复率、清洗成本与验证指标**。面试时 **强调质量与多样性** 比报一个夸张数字更专业。

---

## 6. MedicalGPT 中的实现：`pretraining.py` 核心逻辑

以下描述以官方仓库 [`pretraining.py`](https://github.com/shibing624/MedicalGPT/blob/main/pretraining.py) 为准（若你本地版本略有差异，以你克隆的 commit 为准）。

### 6.1 总览：从 `main()` 出发

1. **`HfArgumentParser`** 解析四类参数：`ModelArguments`、`DataArguments`、`Seq2SeqTrainingArguments`、`ScriptArguments`（含 LoRA/QLoRA）。  
2. **加载 tokenizer**（可与模型同路径或单独指定）。  
3. **`block_size`** 与 `tokenizer.model_max_length` 取 **min**，防止超长。  
4. **`load_dataset`**：从目录递归收集 `txt/json/jsonl`（训练与验证 **文件类型需一致**）。  
5. **预处理两条分支**：  
   - **`packing=True`（默认）**：`tokenize_wo_pad_function` → `group_text_function`  
   - **`packing=False`**：`tokenize_function`（每条样本 **padding 到 block_size**）  
6. **加载模型**（可选 4bit/8bit、QLoRA 配置）。  
7. **`use_peft`**：构造 `LoraConfig`，`get_peft_model`（`target_modules=all` 时用 `find_all_linear_names` 自动找 Linear，**跳过 `lm_head` / `output_layer`**）。  
8. **`SavePeftModelTrainer`**：`Trainer` 训练；`do_eval` 时可算 **token 级 accuracy** 辅助观察。  
9. 训练结束 **保存** 权重与 tokenizer；评估可算 **perplexity**（由 `eval_loss` 指数得到）。

### 6.2 为什么 PT 脚本用 `Seq2SeqTrainingArguments`

历史与复用原因：文件改编自 HuggingFace 的 CLM 示例；**任务仍是因果语言模型**。读代码时 **以实际 `Trainer` 行为与数据列（`input_ids`/`labels`）为准**，不必被名字里的 Seq2Seq 吓到。

### 6.3 `SavePeftModelTrainer` 是干什么的

在 LoRA 训练结束时，用 **`model.save_pretrained`** 把 **适配器与可训练部分** 存到 `output_dir`（全参时行为以你修改后的代码为准）。工程上要分清：**存的是「基座+LoRA」还是仅 adapter**——以你加载推理时的路径约定为准。

### 6.4 评估指标：`compute_metrics` 在说什么

验证时把 **logits argmax** 与 **labels** 对齐（注意 **shift**：预测位置与标签位置错开一位），算 **token 级 accuracy**。  
**面试句**：PT 主看 **loss / perplexity**；accuracy 只是 **辅助观测**，不要迷信。

---

## 7. 关键参数详解

### 7.1 `learning_rate`（学习率）

- **含义**：每次更新参数的步伐大小。  
- **PT 特点**：继续预训练往往可以用 **比 SFT 更大** 的 LR（尤其 LoRA 时），但 **没有银弹**。  
- **`run_pt.sh` 示例**：`2e-4`（配合 LoRA；若全参通常要更谨慎）。  
- **调参直觉**：loss **震荡发散** → 降 LR；loss **几乎不动** → 检查数据与学习率是否过小、是否 dead batch。

### 7.2 `per_device_train_batch_size`（单卡 batch）

- **含义**：每张 GPU **一步** 喂多少条样本。  
- **PT**：`packing=True` 时，一条样本是 **固定长度 block** 的稠密序列，**显存压力与序列长度强相关**。  
- **OOM 时**：先降 `per_device_train_batch_size`，再配合梯度累积维持等效 batch。

### 7.3 `gradient_accumulation_steps`（梯度累积）

- **含义**：多步 forward-backward **累积梯度** 再 `optimizer.step()`，用 **时间换空间**。  
- **有效 batch 粗算**：  
  \[
  B_{\text{eff}} \approx \text{per\_device} \times \text{GPU数} \times \text{accumulation}
  \]  
- **`run_pt.sh` 示例**：`per_device_train_batch_size=4`，双卡，`accumulation=8` → 等效很大，**更稳的梯度估计**。

### 7.4 `max_seq_len` vs `block_size`（本脚本里主要叫 `block_size`）

- 在 `pretraining.py` 里，训练长度主参数是 **`--block_size`**（默认数据类里 1024，脚本可覆盖）。  
- **含义**：**每个训练样本** 包含的 token 数（配合 packing 时是 **切块长度**）。  
- **增大 block**：上下文变长，**显存暴涨**（注意力 \(\propto L^2\) 量级压力）。  
- **过小 block**：长程依赖学弱；医学长段落可能被 **切断**。

### 7.5 `num_train_epochs`（训练轮数）

- **含义**：把整个训练集扫多少遍。  
- **`run_pt.sh` 示例**：`0.5`（半轮）——配合 `max_train_samples` 常用于 **演示/调试**；真训练要拿掉样本上限并重新估算。  
- **过拟合信号**：训练 loss 仍降，但验证 **perplexity** 变差、续写 **复读** 或 **胡编医学细节**。

### 7.6 与 `max_train_samples` / `max_eval_samples`（极易踩坑）

官方 `run_pt.sh` 里 **`--max_train_samples 10000`** 是 **调试友好** 设置：只训 1 万条。  
**上线训练务必去掉或设为 None**，否则你以为训了「全量语料」，实际只喂了 **一小勺**。

---

## 8. 数据处理流程：`tokenize` → `group_texts` → batches

### 8.1 流程 ASCII 图（PT 阶段数据流）

```
  原始语料目录 (txt/json/jsonl)
           |
           v
    load_dataset 读入 Dataset
    每行/每条 -> 字段 "text"
           |
           +------------------+------------------+
           | packing=True      | packing=False
           v                   v
   tokenize_wo_pad        tokenize_function
   (不强制 pad 到满长)     (truncation + pad 到 block_size)
           |                   |
           v                   |
   group_text_function         |
   多文拼接 EOS 再切块         |
           |                   |
           +---------+---------+
                     v
              input_ids, attention_mask
              labels = input_ids.copy()
                     |
                     v
           fault_tolerance_data_collator
                     |
                     v
              Trainer 一步 forward
                     |
                     v
                 CLM Loss
```

### 8.2 `packing=True` 时发生了什么（高频考点）

`group_text_function` 逻辑（语义描述）：

1. 把 batch 内各篇文本的 `input_ids` **首尾相接**；  
2. 篇与篇之间用 **`eos_token_id`** 分隔（若原文末尾无 EOS 会补一个）；  
3. 接成长流后，按 **`block_size`** **切成多块**；  
4. 丢弃 **不足一块的尾巴**（避免形状不齐）；  
5. **`labels` 复制 `input_ids`**，用于 CLM。

**好处**：减少 **padding 浪费**，GPU 算力用在 **真实 token** 上，吞吐高。  
**注意**：同一块里可能 **跨文档**，模型会学到「EOS 后开启新篇」的边界；一般可接受。

### 8.3 `packing=False` 时

`tokenize_function` 对每条 `text` **截断/填充** 到 `block_size`，更像「一篇一条」。**短文本会 pad**，padding 位置在 loss 里通常由 attention mask 处理；实现细节以 HuggingFace 模型 forward 为准。

### 8.4 `create batches` 是谁做的

**`Trainer` + DataLoader** 自动组 batch。`pretraining.py` 使用自定义 **`fault_tolerance_data_collator`** 做 **容错堆叠**（避免某些字段导致默认 collator 崩）。

---

## 9. Causal Language Modeling（CLM）的 loss 计算

### 9.1 对齐方式（时间步错位）

对序列位置 \(t\)，模型在位置 \(t\) 的输出 logits 用来预测 **\(t+1\)** 的 token。实现上常用 **shift logits / shift labels**：

- 去掉 logits **最后一个时间步**  
- 去掉 labels **第一个时间步**  

二者对齐后算 **交叉熵**。

### 9.2 交叉熵直觉

每个位置猜 **词表大小** 的分类问题；**越自信地猜对**，loss 越低。  
**Perplexity（困惑度）**：\(\mathrm{PPL} = \exp(\text{平均 loss})\)，**越低越好**（直观：模型「平均在多少个词里犹豫」）。

### 9.3 padding 与 ignore_index

具体哪些位置参与 loss，由模型内部的 **`labels` 与 `attention_mask` 处理** 决定；读源码时关注 **`CausalLM` 的 loss 计算** 与 HF 版本行为。

---

## 10. 训练命令示例：`run_pt.sh` 内容解析

官方 [`run_pt.sh`](https://github.com/shibing624/MedicalGPT/blob/main/run_pt.sh) 典型内容如下（引用随仓库更新可能微调）：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 pretraining.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --train_file_dir ./data/pretrain \
  --validation_file_dir ./data/pretrain \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --use_peft True \
  --seed 42 \
  --max_train_samples 10000 \
  --max_eval_samples 10 \
  --num_train_epochs 0.5 \
  --learning_rate 2e-4 \
  --warmup_steps 5 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 50 \
  --eval_strategy steps \
  --save_steps 500 \
  --save_strategy steps \
  --save_total_limit 13 \
  --gradient_accumulation_steps 8 \
  --preprocessing_num_workers 10 \
  --block_size 512 \
  --packing True \
  --output_dir outputs-pt-qwen-v1 \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing True \
  --cache_dir ./cache
```

### 10.1 逐段解读（建议你对着终端默读一遍）

| 片段 | 含义 |
|------|------|
| `CUDA_VISIBLE_DEVICES=0,1` | 只用第 0、1 号 GPU。 |
| `torchrun --nproc_per_node 2` | **单机双进程** DDP，每卡一进程。 |
| `Qwen/Qwen2.5-0.5B` | **Base** 向模型（非 Instruct 也可做 PT；后续 SFT 再对齐对话）。 |
| `train_file_dir` / `validation_file_dir` | 语料目录；脚本递归搜 `txt/json/jsonl`。 |
| `use_peft True` + LoRA 参数 | **增量 PT 常用 LoRA** 降本；不是唯一选择。 |
| `max_train_samples 10000` | ⚠️ **调试截断**；全量训练应去掉。 |
| `max_eval_samples 10` | ⚠️ 验证极少条；正式训练应增大 eval 才有意义。 |
| `block_size 512` | 训练块长；显存紧张时常选 512/1024。 |
| `packing True` | 拼接切块，**吞吐高**。 |
| `gradient_checkpointing True` | **换时间省显存**（重算激活）。 |
| `bf16` / `torch_dtype bfloat16` | 混合精度训练；需硬件支持。 |

### 10.2 为什么 PT 用 Base 模型也合理

PT 学的是 **文本续写分布**；**Instruct 模型** 也能做 PT，但要小心 **破坏对话格式先验** 的风险与 **学习率** 设置。工程上常见：**Base 上 PT → 再 SFT 成助手**，路径清晰。

---

## 11. 什么时候可以跳过 PT 阶段

### 11.1 可跳过或弱化的信号

- 任务以 **短问答** 为主，且 **SFT 数据极高质量**、**基座够强**。  
- 事实性主要靠 **RAG**，模型只负责 **组织语言**。  
- **算力/时间紧**，优先保证 **SFT + 对齐**。  
- 领域语料 **获取成本高且质量差**，PT 可能 **得不偿失**。

### 11.2 仍建议做 PT 的信号

- 长文本 **医学生成**（小结、摘要、叙述）且评测显示 **域内语言明显不对**。  
- 专有术语、缩写、规范表述 **频繁出错**（在排除数据与解码问题后）。  
- 有大量 **干净域内语料** 不用可惜。

### 11.3 面试标准答法

「PT 不是必选项，是 **数据与目标驱动** 的工程决策：有 **大规模干净域内语料** 且下游对 **语言分布** 敏感，PT 性价比高；否则可 **优先 SFT / RAG / 对齐**。」

---

## 12. 与 L06 的衔接：PT 和 SFT 到底差在哪

### 12.1 一张表（面试背这个）

| 维度 | PT | SFT |
|------|----|----|
| 数据 | 连续文本 / 非结构化 | 指令、对话、问答对 |
| 标签来源 | 自监督（下一词） | 人类示范（output） |
| 主要提升 | 域内 **语言与统计** | **指令遵循**、格式、角色 |
| loss 覆盖 | 通常 **整段**（除 pad） | 常 **只算回答段** |

### 12.2 为什么不能「只做 SFT」？（见下一节面试题）

---

## 13. 面试高频题

### Q1：PT 和 SFT 的区别？

**答法骨架**：  
**数据形态不同**（无标注连续文本 vs 有标注指令数据）；**优化目标形式常同为 CLM**，但 **SFT 常 mask 提示**；**能力侧重不同**（域内语言统计 vs 对话与任务格式）。  

### Q2：为什么不能只做 SFT？

**答法骨架**：  
SFT 容量与时间有限，主要靠 **示范模仿**；若基座对 **医学文本分布** 不熟，要 **硬背更多表层模式**，数据效率差，长文本与术语续写易弱。**PT 在大规模无标注文本上铺开统计优势**，常与 SFT **互补**。  
**补充**：若任务极简单且基座够强，**可以** 不做 PT，这是 **工程权衡** 不是教条。

### Q3：PT 会不会让模型「变笨」？

**可能**。若语料 **脏、偏激、错误多**，或 LR 过大、训练过久，可能出现 **灾难性遗忘** 或 **胡编自信**。需要 **学习率、早停、混合通用语料、LoRA 限制更新子空间** 等手段。

### Q4：PT 的评估指标看什么？

**主**：验证集 **loss / perplexity**；**辅**：token accuracy、人工续写抽检、下游任务（若已有）。  
**不建议**：只看 train loss 自我感动。

### Q5：`packing` 会不会泄露「跨样本边界」？

模型可见 **EOS 边界**；这是 **有意设计** 让模型知道篇界。**隐私合规**仍要在 **数据源** 保证脱敏与授权。

---

## 14. 实操清单（训练前念一遍）

```
□ 训练/验证文件类型一致（全 txt 或全 jsonl）？
□ 字段是否有 text（或脚本认可的第一列）？
□ 是否误留 max_train_samples 导致只训子集？
□ block_size 与显存、预处理 cache 是否可接受？
□ packing 开关是否符合预期？
□ LoRA target_modules 日志是否包含不该训的层？
□ output_dir / tensorboard 路径是否可写？
□ 验证集规模是否足以看 ppl 趋势？
```

---

## 15. 常见踩坑与排障

1. **JSON 字段不对**：没有 `text`，map 阶段异常或训出空数据。  
2. **train/val 混用类型**：脚本直接 `ValueError`。  
3. **DDP + 量化 + ZeRO3**：QLoRA 路径下官方有 **不兼容警告**，需查当前矩阵。  
4. **tokenizer 与 model 词表不一致**：需 `resize_token_embeddings`（脚本在特定条件下处理）。  
5. **把演示脚本当生产**：`max_train_samples=10000` 是 **经典踩坑**。

---

## 16. 小实验：建议你本地做的 2 个 sanity check

1. **`packing=True/False` 各训 200 step**：看 **吞吐（samples/s）** 与 **验证 ppl**。  
2. **decode 第一条 `train_dataset[0]`**：肉眼确认 **不是乱码、不是意外截断**。

---

## 17. 术语表

| 中文 | 英文 | 说明 |
|------|------|------|
| 预训练 | Pre-training | 大规模自监督练基座 |
| 增量预训练 | Continue PT | 在域内语料上继续 CLM |
| 下一词预测 | Next Token Prediction | CLM 的自监督标签 |
| 因果语言建模 | CLM | 只看过去预测未来 |
| 困惑度 | Perplexity | \(\exp(loss)\)，越低越好 |
| 拼接打包 | Packing | 多文拼接后切块训练 |

---

## 18. 推荐阅读

1. [MedicalGPT 仓库](https://github.com/shibing624/MedicalGPT)：`pretraining.py`、`run_pt.sh`、`docs/datasets.md`。  
2. HuggingFace 示例演进：[`run_clm.py` 思路](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling)（理解数据流）。  
3. 本仓库 **L06**：有监督微调与 **prompt mask**。  
4. 本仓库 **L07**：LoRA/QLoRA 如何降低 PT 成本。  
5. 本仓库 **L18**：源码精读对照表（PT vs SFT）。

---

## 19. 课后自检

- [ ] 我能口述 **`packing` 前后** 数据形态变化。  
- [ ] 我能解释 **为什么 `labels` 常等于 `input_ids`**。  
- [ ] 我能说明 **何时跳过 PT** 是合理决策。  
- [ ] 我能 **逐项解释** `run_pt.sh` 里至少 8 个参数。  
- [ ] 我能回答 **PT vs SFT** 与 **只做 SFT 行不行**。

---

## 20. 附录：CLM 对齐示意图（ASCII）

```
  token:  [t0] [t1] [t2] [t3] [EOS]
              \   \   \   \
               v   v   v   v
  模型在位置 0..2 的输出 logits 预测下一 token: t1,t2,t3,EOS
```

---

**下一课（L06）预告**：读完论文还不够——我们用 **有监督微调（SFT）** 教模型 **按指令答题**，并学会 **只对回答算 loss** 的「做题法」。

---

[← 上一课](../L04-MedicalGPT项目全景/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L06-有监督微调SFT/README.md)
