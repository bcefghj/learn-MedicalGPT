[← 上一课](../L15-模型评估与推理部署/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L17-RAG检索增强生成/README.md)

---

# L16 动手实战Colab全流程

> **一句话精髓：**「纸上得来终觉浅，动手才是真功夫。」

本课把 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 官方提供的 Colab Notebook 当成「可执行的地图」：你不仅知道 PT → SFT → DPO 在讲什么，还能在浏览器里把整条链路跑通，并把日志、报错、曲线读成「工程语言」。

---

## 本课你将学会什么

```
        学习目标（Checklist）
        ====================

  [ ] 在 Colab 上稳定拿到 GPU，并理解「为什么有时只有 CPU」
  [ ] 打开官方 DPO / PPO 两个 Pipeline，知道各自差在哪
  [ ] 按 Cell 理解：环境 → 数据 → PT → 合并 → SFT → 合并 → DPO → 推理
  [ ] 看到 loss / tensorboard 曲线时，能说出「正常 vs 异常」
  [ ] OOM、loss 不降、乱码三类问题能按清单自查
  [ ] 面试时能讲：我跑通过全流程，我改过哪些超参、遇到过什么坑
```

---

## 一、Colab 使用指南：如何获取 GPU

### 1.1 Colab 在整条学习链路中的位置

```
   你的笔记本浏览器
   ----------------
          |
          v
   +-------------+     可选：Colab Pro
   | Google Colab | --> 更高优先级 GPU、更长运行时间
   +-------------+
          |
          |  分配「运行时」
          v
   +------------------+
   | Python3 + GPU    |  <-- 本课主战场
   +------------------+
          |
          v
   git clone MedicalGPT --> pip install --> python *.py
```

### 1.2 打开 GPU 的标准路径（必做）

1. 在 Colab 菜单选择：**代码执行程序 → 更改运行时类型**。
2. **运行时类型**：Python 3。
3. **硬件加速器**：**GPU**。
4. **GPU 类型**（若界面提供）：官方 DPO Notebook 说明里建议 **T4**；若只能选「GPU」而无细分，直接保存即可。
5. 点击**保存**。

**免费版 vs Colab Pro（心里有数即可）：**

| 项 | 免费 GPU | Colab Pro / Pro+（以 Google 当前政策为准） |
|----|----------|---------------------------------------------|
| 排队与断连 | 高峰更明显 | 相对更好 |
| 可用 GPU 档次 | 不保证 | 更易拿到更长时、更好卡 |
| 适合本课 | 能跑通演示 | 想重复多组对比实验时更省心 |

**面试怎么说：** 「我在 Colab **免费配额**下跑通过 MedicalGPT 官方 DPO Pipeline，OOM 时通过 **batch / 序列长度 / gradient checkpointing** 三角排查」—— 这是真实工程经验，不是空话。

### 1.3 验证是否真的在用 GPU

在任意代码 Cell 中运行：

```python
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

**预期输出（正常）：**

- `cuda available: True`
- `device: Tesla T4`（或 A100、L4 等，视分配而定）

**常见情况（非正常但不是你的代码错）：**

- `cuda available: False`：当前会话没分到 GPU，或运行时类型仍是 CPU。回到 1.2 重选。
- 长时间排队：免费配额高峰时段常见，可换时段、换账号（合规前提下）、或考虑 Pro。

### 1.4 磁盘与 Hugging Face 缓存（面试常问「Colab 数据放哪」）

```
   Colab 会话内
   ------------

   /content/               <-- 一般 clone 项目到这里
       MedicalGPT/
           data/
           outputs-pt-v1/
           ...

   ~/.cache/huggingface/   <-- 模型权重默认缓存（体积大）
```

**提示：** 会话断开后，**非 Google Drive 挂载**下的文件可能丢失。重要 checkpoint 应 `zip` 下载或挂载 Drive 再保存。

---

## 二、MedicalGPT 提供的两个 Notebook

| Notebook | 一键 Colab | 大致耗时（官方体验向） | 阶段 |
|----------|------------|------------------------|------|
| `run_training_dpo_pipeline.ipynb` | [Open in Colab](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_dpo_pipeline.ipynb) | 约 **15 分钟**量级 | **PT + SFT + DPO** |
| `run_training_ppo_pipeline.ipynb` | [Open in Colab](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_ppo_pipeline.ipynb) | 约 **20 分钟**量级 | **PT + SFT + RLHF(PPO)** |

### 2.1 两条链路怎么选（直觉版）

```
                    训练对齐路线选择
                    --------------

              想先掌握「更简单、更稳」的对齐？
                        |
                        v
                 +-------------+
                 | DPO Pipeline |  无需训练 Reward Model + PPO 的复杂闭环
                 +-------------+

              想理解经典 RLHF 论文链路？
                        |
                        v
                 +-------------+
                 | PPO Pipeline|  RM + PPO，工程与调试成本更高
                 +-------------+
```

**本课主线条：** 以 **DPO Notebook** 为「逐步拆解」对象；你在读懂 DPO 全流程后，再看 PPO Notebook 会轻松很多。

### 2.3 PPO Pipeline 专讲：`run_training_ppo_pipeline.ipynb`（约 20 分钟）

官方 PPO Notebook 走的是经典 **RLHF 三件套**里「后半段」：**在已有 SFT 模型上，先训奖励模型（RM），再用 PPO 微调策略网络**。整体比 DPO 多一个 **RM 训练** 与一个 **环境式 rollout + KL 约束** 的闭环，因此 Colab 上官方体验时间常标 **~20 分钟**（视 GPU 与数据截断而定）。

```
  run_training_ppo_pipeline.ipynb（逻辑流，与仓库脚本名对齐）
  ==========================================================

  Stage 1: PT（可选，与 DPO 线相同思路）
       |
       v
  merge_peft_adapter.py  --> merged-pt/
       |
       v
  Stage 2: SFT
       |
       v
  merge_peft_adapter.py  --> merged-sft/
       |
       +------------------+------------------+
       v                  v                  v
  reward_modeling.py   ppo_training.py    （推理）
  产出 RM checkpoint     用 RM 打分 + PPO      inference.py
       |                  更新策略
       +------------------+
```

**与 DPO Notebook 的对比（面试常考）：**

| 维度 | DPO Pipeline | PPO Pipeline |
|------|--------------|--------------|
| 偏好怎么用 | 成对数据里直接学「chosen vs rejected」 | 先训 RM 给标量分，再 PPO |
| 训练稳定性 | 通常更省事、调参面更小 | KL、advantage、clip 等多旋钮 |
| 算力与脚本 | 少一个 RM 全量训练阶段 | RM + 策略，显存与时间往往更高 |
| 适合 Colab 入门 | 优先跑通 | 适合「我要讲清 RLHF 论文图」 |

**PPO 侧常见 Cell 语义（与 DPO 重叠部分略）：**

1. **环境与数据**：除 `pretrain` / `finetune` 外，需准备 **偏好或打分相关数据** 供 RM（具体文件名以 Notebook 内 `ls` 与命令行为准）。
2. **RM 训练**：`reward_modeling.py` 日志里会出现与「评分/回归」相关的 loss，不要和 CLM 的 loss 数值对表。
3. **PPO 训练**：`ppo_training.py` 可能多轮 **生成 → 打分 → 更新**；若 `trl` 版本升级，参数名以报错提示与官方文档为准。
4. **推理**：同样回到 `inference.py` 或 Notebook 内嵌生成；若策略更新过猛，易出现 **模式崩塌**（只会说短句、重复），与 DPO 的「偏好过拟合」不同类，排查时优先看 **KL 系数、学习率、RM 是否校准**。

**预期输出（正常）：**

- RM：`eval_loss` 或相关指标随训练呈合理变化（不必与 SFT 同量级）。
- PPO：`reward` 或 `objective` 类指标有信号；全程一条直线要怀疑 RM、数据或超参。

**常见错误（加练）：**

| 现象 | 方向 |
|------|------|
| PPO 极慢 | 减小每步生成 token 数、batch、或用更小基座做通流程 |
| RM loss 不降 | 检查标签是否为连续分数/排序一致、数据是否足够 |
| CUDA OOM | PPO 常同时驻留多份模型；优先降 batch、开 checkpointing、缩短生成长度 |

---

## 三、DPO Pipeline 全流程鸟瞰（对照 Notebook 结构）

```
  run_training_dpo_pipeline.ipynb（逻辑流）
  ========================================

  Stage 1: PT (pretraining.py)
       |
       |-- 小数据 + 小模型 + LoRA
       v
  merge_peft_adapter.py  --> merged-pt/
       |
       v
  Stage 2: SFT (supervised_finetuning.py)
       |
       v
  merge_peft_adapter.py  --> merged-sft/
       |
       v
  Stage 3: DPO (dpo_training.py)
       |
       v
  merge_peft_adapter.py  --> merged-dpo/
       |
       v
  inference.py --base_model merged-dpo
```

---

## 四、逐步讲解 DPO Pipeline 的每个阶段（对应 Notebook Cell 语义）

> 说明：Notebook 的 **Cell 序号**会随仓库更新略有变化，下面按 **「阶段 + 典型 Cell 内容」** 讲解，你打开仓库里的 ipynb 对照即可一眼定位。

### 4.0 DPO Notebook Cell 渐进索引表（建议打印对照）

下表是「从打开 ipynb 到跑完推理」的 **心智地图**：你在 Colab 里从上到下执行时，可扫一眼确认自己卡在哪一类 Cell。

```
  Colab 执行方向
  --------------
  Cell 1,2,3 ...  ----->  环境就绪  ----->  PT  ----->  merge  ----->  SFT  ----->  merge  ----->  DPO  ----->  merge  ----->  inference
```

| 阶段 | 典型 Cell 在做什么 | 你应看到的「正常信号」 | 高频翻车点 |
|------|-------------------|------------------------|------------|
| 说明/Markdown | 告诉你全程阶段与预计耗时 | 读懂「先 merged 再下一阶段」 | 跳过导致后面路径错误 |
| `git clone` + `pip` | 拉代码与依赖 | 目录里有 `pretraining.py` 等 | 网络超时、未 `cd` 到项目根 |
| 检查 GPU | `torch.cuda.is_available()` | `True` + 设备名 | 忘开 GPU 运行时 |
| PT 命令 | `pretraining.py` + LoRA | `outputs-pt-v1` 下有 adapter | OOM、数据目录空 |
| merge PT | `merge_peft_adapter.py` | `merged-pt/` 体积明显变大 | `base_model` 与 PT 不一致 |
| SFT 命令 | `supervised_finetuning.py` | `outputs-sft-v1` | `template_name` 与数据不对 |
| merge SFT | `merge_peft_adapter.py` | `merged-sft/` | 上一步 LoRA 未存好 |
| DPO 命令 | `dpo_training.py` | `outputs-dpo-v1` | 偏好字段缺失、`template_name` 错 |
| merge DPO | `merge_peft_adapter.py` | `merged-dpo/` | `base_model` 链错 |
| 推理 | `inference.py` 或交互 Cell | 能生成连贯中文 | tokenizer/合并路径错误导致乱码 |

**面试考点提示：** 能说清「**每个 `output_dir` 里是什么**（adapter 还是全量）」「**merge 在工程上接上了哪一阶段的 `model_name_or_path`**」，比背脚本名更加分。

### 4.1 环境安装（克隆 + 依赖）

**典型代码（Notebook 内）：**

```bash
!git clone --depth 1 https://github.com/shibing624/MedicalGPT.git
%cd MedicalGPT
%ls
!pip install -r requirements.txt
```

**这一阶段在做什么：**

- 固定一份可复现的代码版本（`--depth 1` 浅克隆，省时间）。
- 安装 `transformers`、`peft`、`trl`、`datasets` 等训练依赖。

**预期输出：**

- 目录列表中出现 `pretraining.py`、`supervised_finetuning.py`、`dpo_training.py`、`data/` 等。
- `pip` 末尾显示 `Successfully installed ...` 或无致命 Error。

**常见错误与处理：**

| 现象 | 可能原因 | 处理方向 |
|------|----------|----------|
| `git clone` 超时 | 网络波动 | 重试；或本地下载 zip 上传 Colab |
| `pip` 依赖冲突 | 版本过旧/过新 | 优先按仓库 `requirements.txt`；必要时新建 Colab 运行时 |
| CUDA 相关 wheel 安装失败 | 环境与包不匹配 | 确认用的是 Colab 官方镜像；少用手动混搭 CUDA wheel |

---

### 4.2 模型加载与小模型实验：官方默认 vs `bloomz-560m`

**官方 Notebook 为 Colab 默认选用：** `Qwen/Qwen2.5-0.5B`（小参数、中文友好、社区活跃）。

**你想改用 `bigscience/bloomz-560m` 时：** 本质是替换 **`--model_name_or_path`**（以及后续 `merge_peft_adapter.py` 的 `--base_model`）为 Bloom 系列 checkpoint，并注意 **tokenizer / 模板** 与对话格式是否匹配。

**示例（PT 阶段思路，注意与仓库脚本参数一致）：**

```bash
!python pretraining.py \
  --model_name_or_path bigscience/bloomz-560m \
  --train_file_dir ./data/pretrain \
  ...
```

**面试考点提示：**

- 「换小模型」主要影响：**显存占用**、**收敛速度**、**词表与特殊符号**（`eos`/`pad`）是否齐全。
- MedicalGPT 脚本里普遍会对 **缺失的 eos/pad** 做补齐逻辑；若你换到极冷门权重，要优先检查 **tokenizer 配置是否完整**。

**常见错误：**

- `OSError: ... does not appear to have a file named config.json`：模型名写错或需要登录 Hugging Face（私有模型）。
- `CUDA out of memory`：560M 也可能因 **batch、序列长度、梯度检查点关闭** 而 OOM，见第八节。

---

### 4.3 数据加载（PT / SFT / DPO 各自读什么）

官方 DPO Notebook 的演示数据（以仓库当前说明为准）：

```
   data/
     pretrain/    <-- PT：领域/通用连续文本（演示用小样本）
     finetune/    <-- SFT：对话指令数据（演示用 Belle 子集）
     reward/      <-- DPO：偏好对（chosen vs rejected）
```

**在 Notebook 里常见 Cell：**

```bash
%ls ./data/pretrain/
%ls ./data/finetune
%ls ./data/reward/
```

**预期输出：**

- 能看到 `json` / `jsonl` 文件列表；文件名因版本略有差异属正常。

**常见错误：**

- `FileNotFoundError`：你没有 `cd` 到 `MedicalGPT` 根目录，或数据未随仓库拉下来。
- 数据格式不匹配：SFT 需要对话结构字段；DPO 需要偏好字段（见 L18 / 官方 `dpo_training.py` 中的 `return_prompt_and_responses` 逻辑）。

---

### 4.4 PT 训练（Stage 1）

**典型训练命令（与官方 Notebook 一致，参数可能随仓库微调）：**

```bash
!python pretraining.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --train_file_dir ./data/pretrain \
  --validation_file_dir ./data/pretrain \
  --per_device_train_batch_size 3 \
  --do_train \
  --do_eval \
  --use_peft True \
  --bf16 \
  --max_train_samples 20000 \
  --num_train_epochs 1 \
  --block_size 128 \
  --output_dir outputs-pt-v1 \
  --gradient_checkpointing True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16
```

**预期现象：**

- 日志里出现 `loss` 随 step 波动下降（不必单调）。
- 输出目录 `outputs-pt-v1/` 下出现 `adapter_model.safetensors`、`adapter_config.json` 等。

**常见错误：**

- **OOM**：减小 `per_device_train_batch_size`、`block_size`，或开启 `gradient_checkpointing`。
- **loss = nan**：学习率过大、bf16/fp16 数值不稳定、数据有异常长文本；尝试降 `learning_rate`、改 `fp32` 验证。

---

### 4.5 合并 LoRA 到基座（PT → merged-pt）

```bash
!python merge_peft_adapter.py \
  --base_model Qwen/Qwen2.5-0.5B \
  --lora_model outputs-pt-v1 \
  --output_dir merged-pt/
```

**预期输出：**

- `merged-pt/` 下出现完整权重结构，后续 SFT 的 `--model_name_or_path merged-pt` 才能「接上」。

**常见错误：**

- `base_model` 与训练 PT 时不一致：合并后分布错乱，后续 SFT loss 异常。
- 磁盘满：合并会写出完整模型，注意 Colab 空间。

---

### 4.6 SFT 训练（Stage 2）

```bash
!python supervised_finetuning.py \
  --model_name_or_path merged-pt \
  --train_file_dir ./data/finetune \
  --validation_file_dir ./data/finetune \
  --do_train \
  --do_eval \
  --use_peft True \
  --bf16 \
  --max_train_samples 1000 \
  --output_dir outputs-sft-v1 \
  --gradient_checkpointing True
```

**预期现象：**

- `eval_loss` 往往比 PT 的「语言建模 loss」更「贴近任务」；不同模板下数值不可横向硬比。

**常见错误：**

- 模板名不匹配：`template_name` 与数据对话格式不一致会导致「模型在学错格式」。
- `max_train_samples` 太小：过拟合演示数据，推理时泛化差。

---

### 4.7 合并 SFT（→ merged-sft）

```bash
!python merge_peft_adapter.py \
  --base_model merged-pt \
  --lora_model outputs-sft-v1 \
  --output_dir ./merged-sft
```

---

### 4.8 DPO 训练（Stage 3）

```bash
!python dpo_training.py \
  --model_name_or_path ./merged-sft \
  --template_name qwen \
  --train_file_dir ./data/reward \
  --validation_file_dir ./data/reward \
  --do_train \
  --do_eval \
  --use_peft True \
  --max_steps 100 \
  --max_source_length 256 \
  --max_target_length 256 \
  --output_dir outputs-dpo-v1 \
  --bf16 True \
  --gradient_checkpointing True
```

**预期现象：**

- 日志中出现 DPO 相关指标（具体字段因 `trl` 版本略有差异）；总体应看到优化在进行而非全程常数。

**常见错误：**

- **偏好数据字段不对**：`chosen`/`rejected` 或脚本期望的 `system`/`history`/`question` 结构缺失。
- **模板名错误**：`qwen` 模板与基座不匹配时，偏好对拼接会错位。

---

### 4.9 合并 DPO（→ merged-dpo）与推理测试

```bash
!python merge_peft_adapter.py \
  --base_model merged-sft \
  --lora_model outputs-dpo-v1 \
  --output_dir merged-dpo/

!python inference.py --base_model merged-dpo
```

**预期输出（Notebook 末尾示例语义）：**

- 给定简单中文问题，模型输出连贯、与提示词语言一致。
- 若演示数据极小，**不要期待**「医学专业能力」；本课目标是 **跑通工程链路**。

**常见错误：**

- 推理乱码：见第八节「乱码」清单。
- 仍然像基座：DPO steps 太少、偏好数据噪声大、学习率不合适。

---

## 五、训练 loss 曲线怎么读（TensorBoard）

### 5.1 日志在哪

训练脚本通常将事件文件写在：

```
outputs-*/runs/
```

本地或 Colab 均可启动：

```bash
tensorboard --logdir outputs-pt-v1/runs --host 0.0.0.0 --port 8009
```

Colab 中常配合端口转发或 `%load_ext tensorboard`（按你环境选用）。

**Colab 内嵌 TensorBoard（常用写法）：**

```python
%load_ext tensorboard
%tensorboard --logdir outputs-pt-v1/runs
```

执行后会在 Notebook 下方嵌入面板；若 `logdir` 写错，面板为空。SFT、DPO 阶段把路径改成对应 `outputs-sft-v1`、`outputs-dpo-v1` 即可并排对比（可用父目录 `--logdir outputs` 一次挂载多子目录，视 TF 版本而定）。

### 5.2 正常曲线长什么样（直觉）

```
  loss
   ^
   |  *
   |    * *
   |       *  *  *
   |---------------> step
        前期下降快，后期抖动变窄
```

**要点：**

- **单调下降不是硬性要求**；小 batch、数据噪声会带来锯齿。
- **长期持平或上升**：要怀疑学习率、数据、模板、是否训到了「空标签」样本。

### 5.3 PT vs SFT vs DPO 的「数值」不要硬比

| 阶段 | loss 含义（简化） | 读图重点 |
|------|-------------------|----------|
| PT | 下一词预测 | ppl 趋势、是否 nan |
| SFT | 指令对话上的监督信号 | eval_loss、是否过拟合 |
| DPO | 偏好对齐目标（与 CE 不同） | 是否稳定、是否崩溃 |

---

## 六、如何修改参数做自己的实验（安全改法）

### 6.1 显存不够时优先改什么

```
  调参优先级（OOM 场景）
  ====================

  1) per_device_train_batch_size  ↓
  2) max_source_length / max_target_length / block_size  ↓
  3) gradient_accumulation_steps  ↑（保持有效 batch）
  4) gradient_checkpointing  = True
  5) LoRA rank  ↓（最后才动，影响表达能力）
```

### 6.2 想训得「更充分」时改什么

- **epochs / max_steps** ↑（注意过拟合）
- **learning_rate**：SFT 常比 PT 小一个量级（Notebook 已给参考）
- **数据量**：演示用 `max_train_samples` 截断只为快；真实实验改为全量或更大采样

### 6.3 记录实验（面试官喜欢听这个）

建议你用一个表格记录每次实验：

| id | 基座 | 阶段 | 关键参数 | 指标 | 结论 |
|----|------|------|----------|------|------|
| exp01 | Qwen0.5B | SFT | lr=2e-5 | eval_loss=... | 基线 |
| exp02 | Qwen0.5B | SFT | lr=5e-5 | nan | 过大 |

---

## 七、面试亮点：「我实际跑过全流程」怎么说

**STAR 法则示例（可自行替换数字）：**

- **S（情境）：** 需要在有限算力下验证 MedicalGPT 的 PT-SFT-DPO 链路可复现。
- **T（任务）：** 在 Colab T4 上跑通官方 Notebook，并记录日志与超参。
- **A（行动）：** 按三阶段执行；OOM 时通过调 batch 与序列长度解决；用 TensorBoard 对比曲线。
- **R（结果）：** 成功合并出 `merged-dpo` 并完成 `inference.py` 推理；总结 3 条可复用排错经验。

**加分表述：**

- 你能解释 **为什么 DPO 比 RLHF+PPO 更适合快速实验**。
- 你能说明 **LoRA 权重与合并后全量权重的区别与使用场景**。

---

## 八、常见问题排查：OOM、loss 不下降、乱码

### 8.1 OOM（Out Of Memory）

```
  OOM 排查脑图
  ===========

  是否 batch 太大？ ----是----> 减小 batch / 累积梯度
       |
       否
       v
  是否序列太长？ ----是----> 降 block_size / max_*_length
       |
       否
       v
  是否忘了 checkpointing？ ----是----> 打开 gradient_checkpointing
       |
       否
       v
  是否同时开了全参训练？ ----是----> 改 LoRA / 量化（进阶）
```

### 8.2 loss 不下降

- **学习率过小或过大**：过小「几乎不动」，过大「震荡或 nan」。
- **标签全被 mask**：SFT 里 `train_on_inputs=False` 时，若模板拼接错误，可能导致有效监督极弱。
- **eval 集太大**：每一步 eval 很慢且日志噪声大；演示可调小 `max_eval_samples`。
- **数据重复率过高**：模型快速记住小样本，训练集 loss 很低但推理差。

### 8.3 模型输出乱码

- **tokenizer 与基座不匹配**：合并时用错 `base_model`。
- **生成参数极端**：`temperature` 过高、`top_p` 过大。
- **没训够 / 数据太脏**：演示链路常见；换更干净指令数据再训。
- **特殊符号未对齐**：`eos`/`pad` 缺失导致生成停不下来或拼接异常。

---

## 九、本课自测（建议手写答案）

1. 为什么 PT 在 MedicalGPT 里是「可选」阶段？（提示：官方 Notebook 文案）
2. `merge_peft_adapter.py` 解决的是什么工程问题？
3. DPO 相对 PPO+RM 的主要优势是什么？
4. 你会如何向面试官证明「我真的跑过」？

---

## 十、延伸资源

- 官方仓库：[MedicalGPT](https://github.com/shibing624/MedicalGPT)
- DPO Notebook：[run_training_dpo_pipeline.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/run_training_dpo_pipeline.ipynb)
- PPO Notebook：[run_training_ppo_pipeline.ipynb](https://github.com/shibing624/MedicalGPT/blob/main/run_training_ppo_pipeline.ipynb)
