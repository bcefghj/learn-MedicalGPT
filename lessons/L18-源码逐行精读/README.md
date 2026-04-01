[← 上一课](../L17-RAG检索增强生成/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L19-简历包装与项目描述/README.md)

---

# L18 源码逐行精读（MedicalGPT）

> **一句话精髓：**「读源码是高手和新手的分水岭。」

本课以 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 官方仓库为唯一事实来源，带你把三条主训练脚本读成「可调参、可排错、可讲面试」的模块地图：**`pretraining.py`**（PT）、**`supervised_finetuning.py`**（SFT）、**`dpo_training.py`**（DPO）。

> 说明：下列行号以仓库 `main` 分支为参考；若你本地克隆版本不同，请以 **函数名与类名** 为准在 IDE 中跳转。

---

## 一、为什么要读源码

```
   只跑脚本的人                读过源码的人
   ============                ============

   会改参数                    知道参数落到哪段逻辑
       |                            |
       v                            v
   报错靠搜索                  先看 stack trace 对应模块
       |                            |
       v                            v
   面试背概念                  能讲「数据怎么变 tensor」
```

**读源码的最小目标：** 能回答三个问题——**数据从哪来、loss 怎么算、模型怎么保存**。

---

## 二、MedicalGPT 代码架构总览

```
   MedicalGPT（训练主链路）
   =======================

   pretraining.py          -->  PT：因果语言建模 CLM
        |
        v
   merge_peft_adapter.py   -->  合并 LoRA（工程步骤，非训练核心）
        |
        v
   supervised_finetuning.py --> SFT：对话指令监督
        |
        v
   merge_peft_adapter.py
        |
        +--> reward_modeling.py / ppo_training.py（RLHF 线）
        |
        +--> dpo_training.py（DPO 线）

   应用侧示例：chatpdf.py（RAG）、inference.py（推理）
```

**与 HuggingFace 生态关系：**

- **模型与分词**：`transformers.AutoModelForCausalLM`、`AutoTokenizer`
- **数据**：`datasets.load_dataset`
- **训练器**：`transformers.Trainer`（PT/SFT）与 `trl.DPOTrainer`（DPO）
- **参数高效**：`peft`（LoRA / QLoRA）

---

## 三、`supervised_finetuning.py` 精读（SFT）

> **GitHub 行号锚点（`main` 分支，便于你 IDE 跳转；若仓库更新请以关键字搜索为准）：**  
> - `ModelArguments`：约第 63–137 行  
> - `DataArguments`：约第 140–192 行  
> - `ScriptArguments`：约第 195 行起（含 `use_peft`、`train_on_inputs`、`template_name` 等）  
> - `main()`：约第 325 行起（tokenizer → 数据 → `preprocess_function` → 模型 → `Trainer`）

### 3.1 文件头部：它到底在训练什么？

文件注释说明：针对 **因果语言建模** 类模型（GPT、LLaMA、BLOOM 等），在 json/jsonl 或 Hub 数据集上做微调；并注明部分代码源自 `textgen` 与 FastChat 思路。

**面试句：** 「SFT 脚本本质是 **把多轮对话拼成单条序列**，再用 Trainer 做 **next-token 监督**。」

---

### 3.2 参数解析：四类 `dataclass` 分工

脚本使用：

```python
HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))
```

**为什么有 `Seq2SeqTrainingArguments`？**  
历史与工程复用原因：HF 示例里常用它承载 `generation_max_length` 等字段；在本脚本中 **训练主循环仍是 Causal LM**，但参数解析沿用该类型以兼容更多训练开关。

#### （1）`ModelArguments`：模型、精度、设备

核心字段（语义级精读）：

| 字段 | 作用 |
|------|------|
| `model_name_or_path` | 基座 checkpoint |
| `tokenizer_name_or_path` | 可独立于模型指定 tokenizer |
| `load_in_4bit` / `load_in_8bit` | 量化加载，配合 QLoRA |
| `torch_dtype` | `float16` / `bfloat16` / `float32` / `auto` |
| `device_map` | `auto` 或分布式下置空 |
| `trust_remote_code` | 部分国产模型需要 |
| `flash_attn` / `shift_attn` | 注意力加速与 LongLoRA 相关开关 |
| `neft_alpha` | NEFTune 噪声注入强度 |

`__post_init__`：**强制**要求 `model_name_or_path`，防止空路径静默失败。

#### （2）`DataArguments`：数据从哪来、怎么切分

| 字段 | 作用 |
|------|------|
| `dataset_name` | 直接从 HF Hub 拉数据集 |
| `train_file_dir` / `validation_file_dir` | 本地 `json`/`jsonl` 目录（递归 glob） |
| `max_train_samples` / `max_eval_samples` | 调试截断 |
| `validation_split_percentage` | 无验证集时从 train 切分 |
| `ignore_pad_token_for_loss` | label 中 pad 是否参与 loss |
| `preprocessing_num_workers` | `dataset.map` 并行 |

#### （3）`Seq2SeqTrainingArguments`：训练调度

包含 **学习率、batch、日志步长、DeepSpeed、FSDP** 等标准 HF 训练参数。  
你要改「多久存一次」「多久 eval 一次」，主要在这里。

#### （4）`ScriptArguments`：SFT 特有行为

| 字段 | 作用 |
|------|------|
| `use_peft` | 是否 LoRA |
| `train_on_inputs` | **是否对 prompt 部分也算 loss**（医疗数据要谨慎打开） |
| `target_modules` | LoRA 作用模块，`all` 时脚本会扫描线性层 |
| `lora_rank` / `lora_alpha` / `lora_dropout` | LoRA 超参 |
| `qlora` | 是否走 4bit QLoRA 路径 |
| `model_max_length` | 截断上限，和显存强相关 |
| `template_name` | 对话模板（`template.py`） |

---

### 3.3 `SavePeftModelTrainer`：为什么自定义 Trainer？

```python
class SavePeftModelTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)
```

**目的：** LoRA 训练后 **只存 adapter**，避免把全量权重误存成巨大文件。  
**面试点：** 「全参保存 vs `save_pretrained` 在 PEFT 下的语义不同。」

---

### 3.4 Tokenizer 加载与特殊符号修补

在 `main()` 中，`AutoTokenizer.from_pretrained` 之后立刻：

- 若缺 `eos`：用模板 `stop_str` 补上。
- 若缺 `bos`：与 `eos` 对齐（工程折中）。
- 若缺 `pad`：优先 `unk`，否则退回 `eos`。

**为什么重要？**  
SFT 的 `labels` 里大量位置会被置为 `IGNORE_INDEX`，**pad/eos 不一致**会导致训练 silently wrong。

---

### 3.5 数据加载：Hub 与本地 json/jsonl 两路

逻辑分支：

1. **`dataset_name` 非空**：`load_dataset`；若无 `validation`，用 shuffle + `train_test_split` 切。
2. **否则**：递归收集 `train_file_dir` 下所有 `.json` / `.jsonl`，`load_dataset('json', data_files=...)`。

**面试句：** 「MedicalGPT 的 SFT 数据入口是 **目录级** 的，适合大规模多文件。」

---

### 3.6 数据预处理核心：`preprocess_function`

**目标：** 把样本里的 `conversations`（human/gpt 交替）转成：

- `input_ids`
- `attention_mask`
- `labels`（其中 prompt 部分常为 `IGNORE_INDEX`，除非 `train_on_inputs=True`）

**关键直觉（必须能说清）：**

```
   [---- prompt ----|---- answer ----| eos]

   labels:
   若 train_on_inputs=False:
      [-100, -100, ...,  a1,  a2, ..., eos]
   只对 answer 段计算 loss
```

**`get_dialog` 生成器：**

- 支持 `system` 角色行；
- 用 `template.get_dialog(history_messages, system_prompt=...)` 把多轮对话渲染成 **字符串对列表** `[q,a,q,a,...]`。

**长度控制：**

- 按 source/target 比例动态分配 `max_source_len` / `max_target_len`，避免极端长 prompt 吃光上下文。

**`filter_empty_labels`：** 去掉全 `-100` 的样本，防止无效 step。

---

### 3.7 模型加载流程（与量化、DeepSpeed 的交互）

要点：

- 分布式（`WORLD_SIZE>1`）时 **`device_map` 置空**，交给 DDP/FSDP/DeepSpeed。
- `load_in_4bit/8bit` 时走 `BitsAndBytesConfig`，并可能 `prepare_model_for_kbit_training`。
- `use_peft`：`LoraConfig` + `get_peft_model`；`target_modules=='all'` 时调用 `find_all_linear_names` 自动推断。

**面试高频：** 「LoRA 为什么不训 `lm_head`？」（脚本里对 `lm_head` / `output_layer` 跳过，避免不稳定与无意义开销。）

---

### 3.8 Trainer 初始化与训练

- `DataCollatorForSeq2Seq`：把 batch pad 对齐。
- `Trainer` 或 `SavePeftModelTrainer`：`model=model`，`train_dataset=...`，`eval_dataset=...`。
- 训练结束：`trainer.save_model()` + `tokenizer.save_pretrained()`。

---

## 四、`pretraining.py` 精读（PT）与 SFT 的差异

> **行号锚点：** `ModelArguments` / `DataArguments` 同样在前部 dataclass 区域（约第 47 行起）；PT 特有字段如 `block_size`、`packing` 等在 `DataArguments` 中定义，读时与 SFT 对照差异最直接。

### 4.1 任务定义差异

| 维度 | PT (`pretraining.py`) | SFT (`supervised_finetuning.py`) |
|------|------------------------|----------------------------------|
| 数据形态 | 连续文本 / 语料块 | 对话指令 |
| 目标 | CLM：预测下一 token | 仍 CLM，但 **mask 掉 prompt** |
| 模板 | 一般不需要对话模板 | 强依赖 `template_name` |

### 4.2 `DataArguments` 在 PT 里的特点

- `block_size`：训练序列长度（Notebook 演示可到 128）。
- `packing`：**把多篇文本用 EOS 拼接再切块**，提升吞吐（非常 PT 友好）。

### 4.3 `compute_metrics` 与 accuracy

PT 脚本里常见 **token 级 accuracy** 作为辅助指标（具体实现见 `compute_metrics` 与 `preprocess`）。  
**面试句：** 「PT 更看 ppl/loss 曲线；accuracy 是辅助观测。」

### 4.4 为什么 PT 也叫「增量预训练」

在领域语料上继续 CLM，让 **词分布与实体共现** 更接近医疗文本；但是否必要取决于数据规模与下游任务（官方 Notebook 也提示可跳过）。

---

## 五、`dpo_training.py` 精读（DPO）

> **行号锚点：** `ScriptArguments` 约第 34–175 行；`return_prompt_and_responses` 约第 286–305 行；`DPOConfig` + `DPOTrainer` 构建约第 384–418 行（随版本可能略有偏移）。

### 5.1 文件职责

用 **偏好对**（chosen vs rejected）在 SFT 模型上继续对齐；核心训练器为 **`trl.DPOTrainer`**。

### 5.2 `ScriptArguments`：DPO 脚本的「大一统」参数类

与 SFT 里拆成 `ModelArguments` / `DataArguments` / `Seq2SeqTrainingArguments` / `ScriptArguments` 不同，`dpo_training.py` 当前版本把 **模型、数据、训练、LoRA、模板** 等全部收进 **单个 `ScriptArguments` dataclass**，再用 `HfArgumentParser(ScriptArguments)` 一次解析。字段语义上仍可对号入座：

- **模型侧**：`model_name_or_path`、`tokenizer_name_or_path`、`load_in_4bit/8bit`、`torch_dtype`、`device_map`、`trust_remote_code` 等；
- **数据侧**：`dataset_name`、`train_file_dir`、`validation_file_dir`、`template_name`、`max_source_length` / `max_target_length` 等；
- **训练侧**：`do_train`、`do_eval`、`learning_rate`、`max_steps`、`fp16`/`bf16`、`gradient_checkpointing`、`output_dir`、`report_to` 等；
- **PEFT 侧**：`use_peft`、`qlora`、`target_modules`、`lora_rank` / `lora_alpha` / `lora_dropout`。

**面试句：** 「拆开还是合并是组织问题，关键是我知道 **每个 flag 落到加载、map、DPOConfig、DPOTrainer 哪一步**。」

### 5.3 偏好数据如何进 `DPOTrainer`：`return_prompt_and_responses`

**输入（原始样本字段，语义级）：**

- `system`、`history`、`question`、`response_chosen`、`response_rejected`

**输出（训练所需）：**

```python
{
  "prompt": [...],
  "chosen": [...],
  "rejected": [...],
}
```

**prompt 构造：**

- 把 `history + [[question, '']]` 交给 `prompt_template.get_prompt(...)`，保证与 SFT 同一对话格式。

**过滤条件：**

- `len(prompt+chosen)` 与 `len(prompt+rejected)` 均在 `(0, max_source_length+max_target_length]` 内。

**面试考点：** 「DPO 数据不是普通 QA，而是 **同一 prompt 下两条 completion 的优劣对比**。」

### 5.4 `DPOTrainer` 初始化关键点

```python
trainer = DPOTrainer(
    model,
    ref_model=None if args.use_peft else deepcopy(model),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=peft_config if args.use_peft else None,
)
```

**`ref_model` 语义：**

- **全参 DPO**：需要 reference model（常见做法是拷贝一份冻结参数）。
- **LoRA DPO**：`trl` 路径下可用 `ref_model=None` 由库侧处理（与版本相关，面试时说明「以你使用的 `trl` 版本文档为准」）。

### 5.5 `DPOConfig` 与长度

`max_length=full_max_length`，其中 `full_max_length = max_source_length + max_target_length`，确保 **偏好对拼接后** 不被截断得莫名其妙。

---

## 六、项目如何利用 HuggingFace 生态

```
   HuggingFace 生态在 MedicalGPT 中的落点
   =====================================

   AutoModelForCausalLM / AutoTokenizer / AutoConfig
          |
          v
   datasets.load_dataset（Hub 或 json/jsonl）
          |
          v
   Trainer（PT/SFT） 或  trl.DPOTrainer（DPO）
          |
          v
   peft.LoraConfig + get_peft_model / PeftModel
          |
          v
   tensorboard / wandb 报告（training_args.report_to）
```

**面试亮点：** 「我不是调 API，我是 **HF Trainer 范式 + PEFT + TRL** 的组合拳。」

---

## 七、代码风格：为什么像 Transformers 官方示例

- **`HfArgumentParser` + dataclass**：命令行可复现、参数可序列化。
- **`datasets.map` 预处理**：可缓存、可多进程。
- **日志用 `loguru`**：比 print 更工程化。

**你读其他 LLM 仓库时：** 优先找 `HfArgumentParser`、`Trainer`、`training_args`。

---

## 八、面试高频题（带答题骨架）

### Q1：讲讲 MedicalGPT 代码中的关键模块？

**答：** 三条脚本对应三阶段；`template.py` 统一对话格式；`merge_peft_adapter.py` 负责合并；应用侧 `inference.py` / `chatpdf.py`。

### Q2：模型是怎么加载的？

**答：** `AutoConfig.from_pretrained` →（可选量化 `BitsAndBytesConfig`）→ `AutoModelForCausalLM.from_pretrained` →（可选）`get_peft_model`。

### Q3：LoRA 是怎么集成的？

**答：** `LoraConfig` 指定 `task_type=CAUSAL_LM` 与 `target_modules`；`get_peft_model` 注入可训练适配器；`SavePeftModelTrainer.save_model` 存 adapter。

### Q4：SFT 的 loss 算在哪些 token 上？

**答：** 默认不算 prompt，只算 assistant 回复段（`IGNORE_INDEX` 掩码）。

### Q5：DPO 数据长什么样？

**答：** 同一 `prompt` 下 `chosen` 优于 `rejected`；脚本把多轮上下文与 question 渲染成统一 prompt。

---

## 九、建议你本地怎么「真逐行」读

```
   阅读顺序（90 分钟版）
   ====================

   1) supervised_finetuning.py：从 main() 开始，顺着 print 日志往下
   2) template.py：对照你用的 template_name
   3) dpo_training.py：只读数据 map + DPOTrainer 两块
   4) pretraining.py：读 packing 与 block_size
```

---

## 十、本课自测

1. 说出 SFT 中 `ModelArguments` 与 `DataArguments` 各管什么。
2. 解释 `train_on_inputs=False` 的训练语义。
3. 说明 DPO 的 `chosen/rejected` 与 SFT 单答案数据的区别。
4. 为什么分布式训练时要处理 `device_map`？

---

## 十一、附录 A：`find_all_linear_names` 在面什么试

面试官想听：

- 你知道 LoRA 要挂在 **Linear** 上；
- 你会排除 `lm_head`；
- 量化模型里 Linear 可能是 `bnb.nn.Linear4bit`，需要分支判断。

---

## 十二、附录 B：与 `trl` 版本相关的提醒

`DPOTrainer` 参数名曾随版本调整（如 `tokenizer` → `processing_class`）。  
**实操建议：** 以你环境 `pip show trl` 版本为准阅读对应文档。

---

## 十三、附录 C：数据格式速记（SFT）

典型 `json` 样本（语义）：

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

---

## 十四、附录 D：数据格式速记（DPO）

字段级语义（与 `return_prompt_and_responses` 对齐）：

```json
{
  "system": "...",
  "history": [],
  "question": "...",
  "response_chosen": "...",
  "response_rejected": "..."
}
```

---

## 十五、附录 E：DeepSpeed / ZeRO 与 QLoRA

`dpo_training.py` 里提示：**ZeRO3 与 QLoRA 可能不兼容**。  
面试答法：「分布式分片与量化权重管理冲突，需要按官方矩阵选型。」

---

## 十六、附录 F：为什么 MedicalGPT 爱用 `loguru`

- 结构化日志等级；
- 比大规模 `print` 更易收集。

---

## 十七、附录 G：读源码时的调试技巧

1. 先把 `max_train_samples` 调到很小；
2. 在 `preprocess_function` 后打印 `decode(input_ids)`；
3. 确认 `labels` 非全 `-100`。

---

## 十八、附录 H：`merge_peft_adapter.py` 在链路中的必然性

LoRA 训练产出 adapter；合并后得到 **单文件推理友好的完整权重**（或便于继续下一阶段训练）。

---

## 十九、附录 I：医疗场景读代码要特别看的点

- **模板是否泄露隐私占位**（不要把真实病历写进模板示例）；
- **`train_on_inputs`** 是否误开；
- **DPO 偏好数据是否标注一致**（标准不统一会教坏模型）。

---

## 二十、附录 J：你如何在面试中「展开到代码级」

示范：

「SFT 的 `preprocess_function` 会把多轮对话用 `get_conv_template` 渲染成序列；`labels` 在 prompt 段用 `IGNORE_INDEX` 屏蔽；训练用 HF `Trainer` 和 `DataCollatorForSeq2Seq` 做 padding。DPO 则先把样本映射成 `prompt/chosen/rejected`，再交给 `trl.DPOTrainer`，并设置 `max_length` 为 prompt+answer 上限。」

---

## 二十一、渐进复习（重复记忆用）

### R1：`HfArgumentParser` 解析了哪四个类？（SFT）

`ModelArguments`、`DataArguments`、`Seq2SeqTrainingArguments`、`ScriptArguments`。

### R2：PT 脚本的 `packing=True` 意味着什么？

多文本拼接再按 `block_size` 切块，提升训练效率。

### R3：DPO 损失函数你需要手推吗？

面试初中级通常问到 **直觉**：让模型对 chosen 的概率相对 rejected 更高；具体公式复习 `DPO` 论文。

### R4：为什么 tokenizer 可能要 `add_special_tokens`？

基座不一致时，缺 eos/pad 会导致生成与训练对齐失败。

### R5：`IGNORE_INDEX` 从哪来？

常来自 `LabelSmoother.ignore_index`（与 HF 版本相关），用于在 loss 计算时跳过 prompt token。

### R6：`PeftModel` 在推理时怎么用？

`inference.py` 或你自己脚本：`base + adapter` 或合并后单模型。

### R7：你如何快速定位 OOM？

从 `per_device_train_batch_size`、`model_max_length`、`gradient_checkpointing` 三角开始。

### R8：`template.py` 为什么值得读？

它决定 **对话渲染格式**，是 SFT/DPO 共同的「隐形协议」。

### R9：为什么说 MedicalGPT 接近 Transformers 示例风格？

参数解析、Trainer、dataset.map 三板斧一致。

### R10：读源码的最终产物是什么？

一张 **数据流图** + 一张 **参数表**（你本课应该能画出来）。

---

## 二十二、附录 K：Suggested reading path in GitHub UI

1. 打开 `supervised_finetuning.py` → 搜索 `def main`  
2. 打开 `dpo_training.py` → 搜索 `return_prompt_and_responses`  
3. 打开 `pretraining.py` → 搜索 `block_size`

---

## 二十三、附录 L：和 L16 Colab 的对应关系

你在 Notebook 里运行的命令行，就是本课读的 `main()` 参数入口：**读懂脚本 = Notebook 不再神秘**。

---

## 二十四、附录 M：常见 follow-up

**Q：** 为什么 SFT 用 CLM 而不是 seq2seq？  
**A：** 底座是 decoder-only；用 CLM + label mask 实现「只学习回答段」。

---

## 二十五、附录 N：代码阅读笔记模板

| 模块 | 输入 | 输出 | 依赖 HF API |
|------|------|------|-------------|
| preprocess_function | raw conversations | input_ids/labels | tokenizer |
| DPO map | raw preference | prompt/chosen/rejected | template |

---

## 二十六、附录 O：全参微调 vs LoRA（从脚本角度）

- `use_peft=False`：全参；保存大；显存高。
- `use_peft=True`：LoRA；`SavePeftModelTrainer`。

---

## 二十七、附录 P：为什么有 `check_and_optimize_memory`（若你版本包含）

部分分支会加入显存诊断与 SDP 后端选择；属于工程增强，理解即可。

---

## 二十八、附录 Q：读完本课，你应该能画出的图

```
 raw jsonl --> load_dataset --> map(tokenize) --> Trainer --> adapter/full ckpt
```

---

## 二十九、附录 R：与 RAG 的代码交界

`chatpdf.py` 用 `AutoModelForCausalLM` 与 `PeftModel`，与训练脚本共享 **同一套 HF 加载逻辑认知**。

---

## 三十、附录 S：最后的叮嘱

读源码不要从第一行逐字背，**跟着数据流走**：`load -> map -> train -> save`。
