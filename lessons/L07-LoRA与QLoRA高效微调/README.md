[← 上一课](../L06-有监督微调SFT/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L08-奖励模型RM/README.md)

---

# L07 LoRA 与 QLoRA 高效微调（PEFT）

> **一句话精髓**：*「不用改整本书，只需贴几张便签纸。」*

**L05** 做 PT、**L06** 做 SFT，本质上都是在 **更新神经网络参数**。但全模型的参数量可能是 **数十亿**，全参数微调对显存与时间都不友好。**LoRA** 与 **QLoRA** 是当下最主流的 **参数高效微调（PEFT）** 方案：用极少的 **可训练增量** 去适配新领域与新任务。

### 0.1 把三件事绑在一起记（PT / SFT / LoRA）

```
  数据阶段          训练对象（常见）          像什么
  --------          ----------------          -----
  L05 PT            基座 +（可选）LoRA        厚书上做「行间批注」
  L06 SFT           基座 +（可选）LoRA        答题模板上「贴便利贴」
  L07 本课          讲清「便利贴」数学与工程  批注贴哪、贴多厚
```

**LoRA 不是第三种任务**，而是一种 **参数更新方式**：你可以 **在 PT 用 LoRA**，也可以 **在 SFT 用 LoRA**；MedicalGPT 的 `run_pt.sh` / `run_sft.sh` 默认 `--use_peft True` 就是在说：**先把路跑通**。

### 0.2 读完本课你应该能画出的「一张图」

能在白板上画出：**\(W'\)、\(W_0\)、\(B\)、\(A\)、\(\alpha/r\)** 的关系，并指出 **哪部分冻结、哪部分反传**。面试官让你「推一下 LoRA 参数量」时，你能用 \(r(d_{\text{in}}+d_{\text{out}})\) **口算数量级**。

---

## 本课你将学会什么

1. 解释 **为什么需要 PEFT**（显存、存储、多版本迭代）。  
2. 掌握 LoRA 核心式子 **\(W' = W_0 + BA\)** 与 **低秩** 含义。  
3. 用 **类比** 建立直觉，并能口述 **简化数学推导**。  
4. 理解超参 **`r` / `alpha` / `target_modules` / `lora_dropout`**。  
5. 知道 **`rank` 与 `alpha` 的经验取值** 与调节方向。  
6. 会选择 **`q_proj` / `k_proj` / `v_proj` / `o_proj`** 等待注入模块。  
7. 理解 **QLoRA = LoRA + 4bit 权重**，以及 **NF4、双重量化、分页优化器** 在解决什么问题。  
8. 对照 MedicalGPT 的 `pretraining.py` / `supervised_finetuning.py` 理解 **`LoraConfig` 与 `get_peft_model`**。  
9. 掌握 **LoRA 权重合并与加载** 的常见工作流。  
10. 记住 **显存对比** 的量级关系与 **LoRA 优缺点**。  
11. 回答面试题：**LoRA 为什么有效？rank 多大？与全参 trade-off？**

---

## 1. 为什么需要参数高效微调（PEFT）

### 1.1 全参数微调的「三座大山」

1. **显存**：不仅要存参数，还要存 **优化器状态**、**梯度**、**激活**（视实现而定）。  
2. **存储**：每次实验一个完整权重副本，**磁盘与版本管理**压力大。  
3. **迭代速度**：团队并行试 **数据配方 / 模板 / 超参** 时，全参 **太慢**。

### 1.2 PEFT 的承诺

**只训练一小撮参数**，其余权重 **冻结（frozen）**。训练结束主要产出 **小适配器（adapter）**，可 **热插拔**。

### 1.3 类比（本课主旨）

把基座模型当成一本 **印刷好的厚教科书**（\(W_0\)）：

- **全参数微调**：把每一页 **重印** ——贵、慢、容易 **改坏** 原书结构。  
- **LoRA**：在关键页面 **贴便签**（\(BA\)）——读者看到的内容是「原页 + 便签补充」，**成本低、可撕掉换一套**。

---

## 2. LoRA 核心原理：\(W' = W_0 + BA\)

### 2.1 记号

- \(W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\)：预训练得到的 **冻结权重**。  
- \(B \in \mathbb{R}^{d_{\text{out}} \times r}\)，\(A \in \mathbb{R}^{r \times d_{\text{in}}}\)：可训练的低秩矩阵，通常 **\(r \ll \min(d_{\text{out}}, d_{\text{in}})\)**。  
- 前向（简化写法）：

\[
W' = W_0 + BA
\]

\[
y = W' x = W_0 x + B A x
\]

其中 \(W_0 x\) **不算梯度**（冻结），\(BAx\) **对 A、B 反传**。

### 2.2 「低秩」是什么意思

**秩（rank）** 可以理解为矩阵 **真正独立的信息维度**。LoRA 假设：**权重更新 \(\Delta W\) 可以用低秩矩阵近似**：

\[
\Delta W \approx BA,\quad \text{rank}(\Delta W) \le r
\]

直觉：领域适配往往不需要 **扭转整个高维空间**，只需要在 **少数几个方向** 上推拉。

---

## 3. 直觉理解：用类比讲清楚 LoRA

### 3.1 音响均衡器类比

原曲 = 基座能力；LoRA = **只调几个频段**（低音/中音/高音），而不是 **重录整首歌**。

### 3.2 办公装修类比

- 全参：拆墙重建。  
- LoRA：换窗帘、加台灯、贴海报 —— **观感变很多**，但 **主体结构未动**。

### 3.3 医学场景类比

基座已懂 **中文与基础常识**；医学适配主要是：

- 术语偏好（「心肌梗死」vs 口语）  
- 文书风格（更像指南摘要）  
- 任务格式（问诊追问）

这些变化往往 **不需要重写所有神经元连接**，LoRA 足够实用。

---

## 4. LoRA 的数学推导（简化版，面试够用）

### 4.1 从全量更新到约束更新

全参微调优化整个 \(W\)。LoRA 令：

\[
W = W_0 + \Delta W,\quad \Delta W = BA
\]

参数量从 \(d_{\text{out}} d_{\text{in}}\) 降到 **\(r(d_{\text{out}} + d_{\text{in}})\)**。

### 4.2 参数量对比（数量级直觉）

设 \(d_{\text{out}} = d_{\text{in}} = 4096\)，\(r = 8\)：

- 全量 \(\Delta W\)：约 \(4096^2 \approx 1.6 \times 10^7\) 参数。  
- LoRA：\(8 \times (4096 + 4096) \approx 6.5 \times 10^4\) 参数。

**缩小两个数量级** 很常见。

### 4.3 缩放因子 \(\alpha / r\)（工程实现）

HuggingFace PEFT 常引入缩放：

\[
y = W_0 x + \frac{\alpha}{r} B A x
\]

**直觉**：当 \(r\) 变化时，用 \(\alpha\) 维持 **更新强度可比**。

---

## 5. 关键超参数详解

### 5.1 `rank`（`r`）

- **含义**：低秩适配的 **容量旋钮**。  
- **更大**：表达力强，显存与训练成本上升，可能 **过拟合** 小数据。  
- **更小**：更省，但 **欠拟合** 复杂适配。

### 5.2 `lora_alpha`（\(\alpha\)）

- **含义**：对 LoRA 分支的 **增益**。  
- **经验**：常与 `r` 成对调；MedicalGPT 示例里常见 **`r=8, alpha=16`**（比例 2:1）。

### 5.3 `target_modules`

- **含义**：在 **哪些线性层** 上插入 LoRA。  
- **`all`（MedicalGPT）**：脚本会 `find_all_linear_names` 自动找 **合适的 Linear**（排除 `lm_head` 等）。

### 5.4 `lora_dropout`

- **含义**：LoRA 分支上的 dropout，**正则化** 防过拟合。  
- **典型**：`0.05`～`0.1`。

---

## 6. `rank` 和 `alpha` 的设置经验

### 6.1 起步建议（工程向）

| 场景 | 起点（经验） | 调参方向 |
|------|----------------|----------|
| 小数据 SFT | r=8, α=16 | 过拟合→加 dropout / 减 r；欠拟合→增 r |
| 大数据域适配 | r=16 或更高 | 观察验证集与生成质量 |
| PT（继续语言建模） | r=8～64 | 视语料规模与基座差距 |

### 6.2 不要迷信大 rank

rank 很大时，LoRA 接近 **小范围全参**，**失去 PEFT 优势**。要先 **洗数据**，再考虑加 rank。

### 6.3 与 learning rate 的耦合

LoRA 往往用 **比全参更大的 LR**（相对意义上），但仍需 **warmup** 与 **监控 loss**。

---

## 7. `target_modules` 怎么选（q/k/v/o）

### 7.1 Transformer 注意力里的线性层

标准多头注意力里常见投影：

- **`q_proj`**：Query  
- **`k_proj`**：Key  
- **`v_proj`**：Value  
- **`o_proj`**：输出投影（把多头合并回残差维度）

### 7.2 常见策略

1. **只加 q、v**：更省显存，有时效果也够。  
2. **q、k、v、o 全加**：更常见、更稳（MedicalGPT `all` 会覆盖相关线性层，具体以 `find_all_linear_names` 结果为准）。  
3. **再加 FFN（gate/up/down）**：更强表达，成本上升。

### 7.3 面试一句话

**注意力投影控制「信息怎么读与怎么写」，是适配的甜点区；FFN 更像「知识存储与非线性变换」，有时也要动，但成本更高。**

---

## 8. QLoRA：LoRA + 4bit 量化

### 8.1 动机

即使只训 LoRA，**前向仍要承载全精度基座权重** 的某种表示。QLoRA 把 **基座权重用 4bit 存**，并用 **量化-反量化** 在计算时配合 **LoRA 低秩分支**。

### 8.2 训练时的大致结构（概念）

```
 4bit 基座 W0（省显存）
        +
 LoRA 分支 BA（fp16/bf16 等计算 dtype）
```

**梯度主要更新 A、B**；基座量化权重通过 **特殊内核与配置** 支持训练回路。

---

## 9. QLoRA 的核心创新（记关键词即可）

### 9.1 NF4 量化（NormalFloat4）

**思想**：权重分布常近似 **零中心正态**；NF4 把量化格子 **按正态分位数划分**，同样 4bit 下 **信息保留更好**。

**面试关键词**：**适配正态分布的分位数量化**。

#### 9.1.1 直觉图：为什么「分位数」比「均匀格子」更省位宽

```
  权重值大致集中在 0 附近（示意）
       |
       |     * * * * *
       |   * * * * * * *
       | * * * * * * * * *
       +--------------------> 数值
       均匀切 16 档：两端格子很空 → 浪费
       按分位数切 16 档：中间更密 → 同样 4bit 更划算
```

### 9.2 双重量化（Double Quantization）

**思想**：对 **量化常数（scale）** 再做一次量化，进一步 **压存储与搬运开销**。

**面试关键词**：**连 scale 也量化**。

#### 9.2.1 一句话类比

第一次量化是「把书压缩成简写版」；双重量化是「连简写版的目录说明也再压缩一次」——**元数据也要省空间**。

### 9.3 分页优化器（Paged Optimizer）

**思想**：优化器状态在 GPU 显存紧张时可 **与 CPU 分页交换**，减少 OOM。

**面试关键词**：**CPU offload 分页，缓峰值显存**。

#### 9.3.1 什么时候你会在日志里感谢它

- batch 略大、序列略长，训练 **偶发 OOM**；  
- 多任务并行，GPU 上 **碎片显存** 不够用；  

它不是「免费午餐」：**CPU-GPU 搬运** 可能让 step 变慢，但 **能训完** 往往比 **训到一半崩** 更值。

> 细节以 QLoRA 原论文与 bitsandbytes 实现为准；面试答到 **目的 + 三个名字** 通常足够。

---

## 10. MedicalGPT 中的 LoRA / QLoRA 使用方式

### 10.1 共同模式（PT 与 SFT）

在 `pretraining.py` 与 `supervised_finetuning.py` 中，流程高度相似：

1. `AutoModelForCausalLM.from_pretrained` 加载基座（可选 `BitsAndBytesConfig` 4bit/8bit）。  
2. 若量化 + 训练，调用 `prepare_model_for_kbit_training`。  
3. 解析 `target_modules`；若含 `all`，`find_all_linear_names` 自动推断。  
4. 构造 `LoraConfig(task_type=TaskType.CAUSAL_LM, r=..., lora_alpha=..., ...)`。  
5. `get_peft_model(model, peft_config)`。  
6. `Trainer` 训练；保存时 `PeftModel.save_pretrained` 只存 **adapter**。

### 10.2 QLoRA 开关

脚本参数里有 `--qlora` 布尔项；与 `load_in_4bit` 及 `BitsAndBytesConfig` 的 **`bnb_4bit_use_double_quant`、`bnb_4bit_quant_type="nf4"`** 等联动（以源码为准）。

### 10.3 训练后 dtype 小处理

源码中对部分可训练参数有 `.float()` 或保持训练稳定性的处理；**读源码时关注「量化 + LoRA 的 dtype 一致性」**。

---

## 11. 代码示例：`LoraConfig` 配置

下面是与 HuggingFace PEFT **风格一致** 的示例（与 MedicalGPT 参数对应）：

```python
from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 或 MedicalGPT 的 "all"
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

**QLoRA（概念示例）** 还需 `BitsAndBytesConfig` 与 `load_in_4bit=True` 等加载选项，具体拼法以你环境的 `transformers` / `bitsandbytes` 版本为准。

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config, ...)
```

---

## 12. LoRA 权重合并与加载

### 12.1 两套权重

- **Base**：原始预训练模型。  
- **Adapter**：LoRA 的 A、B（以及配置）。

### 12.2 推理时常见两种做法

1. **不合并**：`PeftModel.from_pretrained(base, adapter_dir)` 动态应用 LoRA。  
2. **合并（merge）**：把 LoRA 融进线性层得到 **单一大权重**，推理更快、部署更简单。

### 12.3 `merge_and_unload` 概念

在 PEFT 中常见 API（名称以库版本为准）：

```python
# 伪代码：合并 LoRA 到基座并卸载适配器包装
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged-ckpt")
```

### 12.4 注意点

- **合并后再继续训** 与 **只训 adapter** 是不同工作流。  
- **多适配器切换** 更适合 **不合并**。  
- 量化模型上的合并/导出 **更敏感**，需查当前版本文档。

---

## 13. 显存对比表（全参数 vs LoRA vs QLoRA）

> 以下为 **教学量级表**，真实数字随模型大小、序列长度、batch、框架版本波动极大。

| 方案 | 可训练参数 | 基座权重精度（典型） | 相对显存（同任务粗比） | 备注 |
|------|------------|----------------------|-------------------------|------|
| 全参数 FP16/BF16 | 100% | FP16/BF16 | 高 | 最高上限 |
| LoRA + BF16 基座 | 很少 | BF16 | 中～高 | 主要省「可训状态」，基座仍占大头 |
| QLoRA 4bit + LoRA | 很少 | 4bit 基座 | 低～中 | 小卡友好 |

**记忆口诀**：**QLoRA 砍基座存储，LoRA 砍训练参数；两者可叠加。**

### 13.1 显存都花在哪了（面试加分：不只说「模型很大」）

粗略拆分一块 GPU 上训练时的「大头」：

```
  参数权重（W0，是否量化）
        +
  梯度（对可训练部分）
        +
  优化器状态（Adam 一类：动量等，常是「隐形富豪」）
        +
  激活（与 batch、序列长度、是否 checkpoint 强相关）
```

- **LoRA** 主要砍 **可训练参数 → 梯度与优化器状态**。  
- **QLoRA** 再砍 **基座权重的存储与带宽压力**（配合 NF4 等）。  
- **Gradient Checkpointing** 砍 **激活**，换 **重算时间**。

### 13.2 一张「怎么选」的决策草图（ASCII）

```
  显存充裕 + 追极限效果？ ──> 评估全参数（或更大模型）
        |
        否
        v
  还能 BF16 装下基座？ ──> LoRA / 部分层解冻
        |
        否
        v
  尝试 QLoRA（4bit + LoRA）+ 合理 batch/长度
```

---

## 14. LoRA 的优缺点

### 14.1 优点

- **省显存 / 省存储**（尤其只存 adapter）。  
- **多版本并行**：同一基座 + N 套便签。  
- **快速迭代数据配方**。  
- **降低灾难性遗忘风险**（相对全参，仍可能发生）。

### 14.2 缺点

- **容量上限**：极难任务或巨大分布偏移时不如全参。  
- **推理开销**：不合并时 **额外计算 BA 分支**（通常可接受）。  
- **排查更难**：问题可能来自 **模板、数据、rank、target_modules** 的组合。

---

## 15. ASCII 图：LoRA 矩阵分解示意图

```
  输入向量 x (d_in)
        |
        +--------------------+
        |                    |
        v                    v
   冻结 W0 · x           A · x  ---> (r 维)
   (原基座输出)                |
                               v
                           B · (A·x)
        |                    |
        v                    v
        +---------+----------+
                  |
                  v
            y = W0·x + (α/r)·B·A·x
```

**r 很小** 时，中间 **瓶颈维度** 细，像 **细管子**，迫使更新 **结构化、低自由度**。

---

## 16. 面试高频题

### Q1：LoRA 为什么有效？

**答法框架**：

- 大模型适配往往位于 **低维子空间**；低秩分解 **参数效率高**。  
- 冻结 \(W_0\) **保留通用能力**，\(\Delta W\) **专注领域/任务残差**。  
- 实践中 **SFT/PT** 的目标可用小更新近似。

### Q2：`rank` 设多大合适？

**答法**：

- **无固定值**；从 **8/16** 起步。  
- **指标**：验证 loss、生成质量、是否过拟合。  
- **趋势**：数据越大、任务越难，**倾向更大 rank**（但警惕性价比）。

### Q3：LoRA vs 全参数微调的 trade-off？

| 维度 | LoRA | 全参 |
|------|------|------|
| 成本 | 低 | 高 |
| 上限 | 中～高（视任务） | 最高 |
| 遗忘 | 相对可控 | 更易伤基座 |
| 部署 | 可合并可分离 | 大文件 |

### Q4：为什么常对 q、v 加 LoRA？

**答法**：Query/Value 控制 **注意力的检索与取值**；对 **风格与任务路由** 敏感，性价比高（经典经验，不是绝对真理）。

### Q5：QLoRA 会不会让精度崩？

**答法**：**可能**，但 NF4+双重量化+合适 compute dtype 在多数场景 **可用**；最终以 **任务评测** 为准。

---

## 17. 与 L06 的衔接：为什么 SFT 脚本默认 LoRA

MedicalGPT `run_sft.sh` / `run_pt.sh` 默认 `--use_peft True`：

- **学习者设备友好**  
- **快速试错**  
- **多实验并存**

当你要追 **极限效果** 且资源充足时，再评估 **全参** 或 **更大 rank / 更广 target_modules**。

---

## 18. 实操检查清单（建议你训练前念一遍）

```
□ target_modules 是否与模型架构匹配（层名是否存在）？
□ template（SFT）是否与 Instruct 基座一致？
□ r/α/lr/warmup 是否同一组实验只改一个变量？
□ 是否保存了 adapter_config.json 与权重？
□ 推理用的是 PeftModel 还是 merged 模型？（路径别混）
□ QLoRA 环境是否安装 bitsandbytes 且 GPU 兼容？
```

---

## 19. 常见错误与排障

1. **OOM**：先降 `per_device_train_batch_size` / `model_max_length` / `r`，或上 QLoRA。  
2. **loss 不降**：检查 **模板 mask**、学习率、数据是否真的进模型。  
3. **推理无效**：加载错基座、adapter 版本不匹配、没 `from_pretrained` adapter。  
4. **把 lm_head 也 LoRA 了**：部分框架/脚本会排除；乱改源码要小心 **词表与输出维**。

---

## 20. 延伸阅读

1. LoRA 论文：*LoRA: Low-Rank Adaptation of Large Language Models*。  
2. QLoRA 论文：*QLoRA: Efficient Finetuning of Quantized LLMs*。  
3. HuggingFace PEFT 文档：[https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)  
4. MedicalGPT：`pretraining.py`、`supervised_finetuning.py` 中的 `LoraConfig` 段落。

---

## 21. 术语表

| 中文 | 英文 | 说明 |
|------|------|------|
| 参数高效微调 | PEFT | 只训少量增量参数 |
| 低秩适配 | LoRA | \(\Delta W \approx BA\) |
| 秩 | rank (r) | 低秩矩阵的瓶颈维度 |
| 缩放系数 | alpha (α) | 与 r 共同调节 LoRA 强度 |
| 双重量化 | Double Quant | 对 scale 再量化 |
| 分页优化器 | Paged Optimizer | 优化器状态分页缓 OOM |

---

## 22. 小计算练习（加深 rank 直觉）

若某层 \(d_{\text{in}}=d_{\text{out}}=4096\)：

- 全量更新参数：\(4096^2 = 16{,}777{,}216\)。  
- LoRA（r=8）：\(8 \times 4096 + 8 \times 4096 = 65{,}536\)。

比例约 **256:1**。面试时口算 **两个数量级差距** 很加分。

---

## 23. LoRA 与「Adapter / Prefix Tuning」族谱（简版）

PEFT 家族很大，面试常问「LoRA 和 Adapter 区别」：

- **Adapter**：在模块间插 **小 MLP 瓶颈层**，前向走旁路。  
- **LoRA**：直接在 **权重矩阵旁加低秩乘积**，更贴近线性层结构。

**记忆**：Adapter 像 **外挂小模块**；LoRA 像 **给原矩阵加残差低秩修正**。

---

## 24. MedicalGPT `run_pt.sh` / `run_sft.sh` 里的 LoRA 参数对照

两份脚本都常见：

```
--use_peft True
--target_modules all
--lora_rank 8
--lora_alpha 16
--lora_dropout 0.05
```

**含义**：自动找线性层 + 经典小秩配置 + 轻 dropout。作为 **默认基线** 很合理。

---

## 25. 合并权重后的部署场景

- **服务端高 QPS**：倾向 **合并** 降低延迟与实现复杂度。  
- **多租户不同适配器**：倾向 **不合并**，动态加载。  
- **边缘设备**：QLoRA 训练后 **导出格式** 需专门验证（生态演进快）。

---

## 26. FAQ

**问：LoRA 能代替 PT 吗？**  
不能简单代替。PT 解决 **语料分布**；LoRA只是 **训练方式**。你可以在 PT 阶段用 LoRA（MedicalGPT 支持），但 **数据与目标** 仍是 PT。

**问：只训 LoRA，知识能更新吗？**  
**部分能**（通过调整表示与输出偏好），但 **强事实更新** 往往还需 **RAG、继续预训练、更大容量更新**。

**问：r=1 是不是笑话？**  
不一定；有时 **极强正则** 或 **极小任务** 会试极端小 rank，但多数工程从 8 起步。

---

## 27. 与下一课（L08 RM）的桥梁

当你有了 **SFT 模型**，你可能会发现：它 **会答**，但 **好坏难分**。下一阶段 **奖励模型（RM）** 学习 **打分函数**，为 RLHF 提供信号 —— **LoRA 也常用于训 RM**，因为同样是 **大模型上的高效适配**。

---

## 28. 课后自检清单

- [ ] 我能写出 \(y = W_0 x + (\alpha/r) B A x\) 并解释每项。  
- [ ] 我能说明 **QLoRA 三个创新词** 各自解决什么。  
- [ ] 我能描述 **合并 vs 不合并** 的适用场景。  
- [ ] 我能口头对比 **LoRA 与全参** 的成本与上限。  
- [ ] 我知道 MedicalGPT 里 **`target_modules all`** 大致会触发什么逻辑。

---

## 29. 进阶：什么时候该加大 `target_modules`

**信号**：

- 验证集/人工评测显示 **推理结构错误**（非语言表面问题）。  
- 仅 q/v LoRA **怎么调都不涨**。

**动作**：

- 扩到 **o_proj / FFN**，或评估 **全参**。

---

## 30. 版本提醒

`transformers`、`peft`、`bitsandbytes` 版本组合对 QLoRA **极敏感**。生产环境请 **锁版本** 并记录：

```
Python / torch / CUDA / transformers / peft / bnb 版本号
```

---

## 31. 代码阅读任务：在 MedicalGPT 里自己「圈出」三行

克隆仓库后，用编辑器搜索以下关键字（**比背概念更牢**）：

1. `LoraConfig`：看 **`r` / `lora_alpha` / `target_modules`** 如何从命令行进来。  
2. `find_all_linear_names`：看为什么 **跳过 `lm_head`**（面试高频）。  
3. `prepare_model_for_kbit_training`：看 **QLoRA 初始化** 在模型加载后哪一步调用。

---

## 32. 合并权重后的「推理路径」对照表

| 场景 | 推荐做法 | 备注 |
|------|----------|------|
| 快速 A/B 切换多个科室适配器 | `PeftModel` 动态加载 | 服务层要有统一基座版本管理 |
| 线上低延迟、部署简单 | `merge_and_unload` 后单文件 | 合并前备份 adapter |
| 边缘设备 | 谨慎：量化+合并路径依赖工具链 | 以目标硬件测试为准 |

---

## 33. 本课与论文的映射（写进简历可用）

- **LoRA**：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* —— **低秩残差假设 + 注入点选择**。  
- **QLoRA**：Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* —— **NF4 + 双重量化 + 分页优化器** 的工程组合。  

简历一句（示例）：「熟悉 **PEFT/LoRA/QLoRA**，能在 **MedicalGPT 的 PT/SFT 脚本** 中配置量化与适配器训练，并理解 **合并推理与 adapter 热切换** 的工程差异。」

---

## 34. 「秩 r」的几何直觉（再补一刀）

把线性层看成 **高维空间里的一个变换**。全参微调允许 **任意方向任意扭**；LoRA 只允许在 **最多 r 个方向** 上扭。  
**r 小**：扭得「克制」，更像 **口音纠正**；**r 大**：扭得「用力」，更像 **换一套说话习惯** —— 也更贵、更易过拟合。

---

**你已完成 Pipeline 核心三连：PT → SFT → PEFT。** 下一步（L08）我们将进入 **偏好对齐的前置：奖励模型 RM**。

---

[← 上一课](../L06-有监督微调SFT/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L08-奖励模型RM/README.md)
