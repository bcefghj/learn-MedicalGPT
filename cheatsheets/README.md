# 核心概念速查表（一页纸）

> **用途**：训练/面试前 **5 分钟** 扫一遍；详细讲解见各课 README 与 **[L20](../lessons/L20-面试通关高频考点/README.md)**。

---

## 导航

| 链接 | 说明 |
|------|------|
| [← 返回课程总览](../README.md) | Learn MedicalGPT 主页 |
| [L20 面试通关高频考点](../lessons/L20-面试通关高频考点/README.md) | 完整问答与追问 |
| [面试速查手册（关键词版）](../interview/README.md) | 80 题极简答案 |

---

## 训练阶段速查（各一行）

| 阶段 | 数据形态 | 目标（直觉） | 典型产物 |
|------|----------|--------------|----------|
| **PT / CPT** | 连续纯文本 | 让模型「读会」领域语言与知识分布 | 领域 LM checkpoint |
| **SFT** | 指令 / 多轮对话 | 让模型「会按格式答」、跟任务对齐 | Chat/SFT 模型 |
| **RM** | 偏好对 / 排序 | 学「人更喜欢哪种回答」 | Reward Model |
| **PPO（RLHF）** | 在线 rollout + RM 分 | 在奖励下改进策略，KL 防跑飞 | 对齐后 policy |
| **DPO** | 离线偏好对 | 直接优化相对 reference 的偏好似然 | 对齐后模型 |
| **ORPO** | 指令+偏好联合（视实现） | 减少分阶段、联合偏好目标（以论文为准） | 对齐后模型 |
| **GRPO** | 组内多样回答 | 组内相对优化、弱化 critic（以论文为准） | 对齐后模型 |

---

## 关键超参数速查表

| 超参 | 常见范围 / 口诀 | 调大/调小直觉 |
|------|-----------------|---------------|
| **学习率（LoRA）** | `1e-4`～`1e-3` 网格 | 大→快但不稳；小→稳但慢 |
| **warmup ratio** | 约 `0.03`～`0.1`（视总步数） | 防一开始破坏预训练特征 |
| **max length** | 受显存与任务约束 | 过长→OOM；过短→丢病史 |
| **LoRA r** | `8 / 16 / 32 / 64` | r↑容量↑显存↑ |
| **LoRA α** | `r` 或 `2r` | 与 r、LR 强耦合，需一起扫 |
| **DPO β** | 常从小开始试 | 大→贴 reference；小→更敢改但易不稳 |
| **grad clip** | 如 `1.0`（依框架默认值再调） | 防爆 NaN / 震荡 |
| **有效 batch** | 单卡小 batch + **梯度累积** | 影响收敛与泛化，不单是显存 |

---

## 显存需求速查（量级直觉，非保证）

> 实际与 **模型规模、序列长、是否 ZeRO、框架实现** 强相关；下列帮助 **选型口述**。

| 设定 | 7B 量级直觉 | 备注 |
|------|-------------|------|
| **推理 fp16/bf16** | ~14GB+ 量级起（+ KV 随长文本涨） | 视实现与 batch |
| **全参微调 7B** | 往往 **多卡高显存** | Adam 状态占大头 |
| **LoRA 7B** | 常 **单卡 24G 可训**（视 seq/batch） | 比全参省很多 |
| **QLoRA 7B** | 更低；适合 **单卡预算紧** | 注意 bnb/CUDA 兼容 |
| **ZeRO-2** | 减 **优化器+梯度** 显存 | 通信适中 |
| **ZeRO-3** | 再减 **参数** 显存 | **通信更重**，带宽要够 |

**口诀**：先 **QLoRA/LoRA + 梯度检查点 + 累积**，不够再上 **ZeRO**。

---

## 算法对比速查：PPO vs DPO vs GRPO vs ORPO

| 维度 | **PPO（RLHF）** | **DPO** | **ORPO** | **GRPO** |
|------|-----------------|---------|----------|----------|
| **是否需要 RM** | 需要显式 RM | 不需要（隐式） | 依方法与实现 | 依方法与实现 |
| **在线采样** | 需要 rollout | 离线偏好对 | 多为离线/联合（视论文） | 常依赖组内采样（视论文） |
| **工程复杂度** | 高（四模型回路） | 相对较低 | 中（看实现） | 中（看实现） |
| **稳定性** | 敏感，需调参 | 相对友好，怕噪声数据 | 视目标设计 | 视组内设计与任务 |
| **面试一句话** | RM+PPO+KL 经典链路 | 直接偏好似然+β 约束 | 联合偏好优化少分阶段 | 组内相对、弱化 critic |

---

## 数据格式速查

| 格式 | 典型字段 | 备注 |
|------|----------|------|
| **Alpaca** | `instruction`, `input`, `output` | 单轮常见；`input` 可空 |
| **ShareGPT** | `conversations[]` → `from`/`value` 或 role/content | 多轮；注意与模板映射 |
| **偏好（DPO）** | `prompt` + `chosen` + `rejected` | 去平局、去隐私、防长度偏见 |
| **PT** | 纯 `text` 或拼接段落 | 注意 EOS 与文档边界 |

**铁律**：训练与推理 **同一 `chat_template`**；SFT **常 mask 非 assistant**。

---

## 常用命令速查（MedicalGPT / HF 生态）

> 路径与脚本名 **以你克隆的仓库为准**；下列为 **典型形态**，复制前请 `--help` 核对参数。

```bash
# 环境（示例）
pip install torch transformers datasets peft accelerate bitsandbytes deepspeed

# 增量预训练（示例：按仓库脚本名调整）
# python pretraining.py --train_file pt.jsonl --model_name_or_path <BASE> ...

# 有监督微调（示例）
# torchrun --nproc_per_node N supervised_finetuning.py \
#   --model_name_or_path <BASE_OR_SFT> \
#   --train_file sft.jsonl --template_name qwen --use_peft True ...

# DeepSpeed（示例）
# deepspeed --num_gpus N supervised_finetuning.py --deepspeed ds_config.json ...

# DPO（示例）
# python dpo_training.py --model_name_or_path <SFT> --ref_model <SFT> ...

# 合并 LoRA（概念命令，依仓库工具）
# python merge_peft_adapter.py --base_model <BASE> --adapter_dir <ADAPTER> --out_dir <MERGED>
```

**自查三件套**：`python -c "import torch; print(torch.cuda.is_available())"`、`transformers.__version__`、`git rev-parse HEAD`。

---

## 一页纸自检（进场前 1 分钟）

1. **三阶段**：我实际做了 **PT? SFT? DPO/PPO?**  
2. **三数字**：**数据量、训练时长、指标提升**（内部集定义）。  
3. **三缩写**：**LoRA、DPO、ZeRO** 各一句人话。  
4. **一合规**：**免责 / 拒答 / 脱敏** 我能说清一条。  

---

## 相关课程索引

| 概念 | 建议回查课程 |
|------|----------------|
| Transformer / 注意力 | [L02](../lessons/L02-Transformer架构核心原理/README.md) |
| PT / SFT | [L05](../lessons/L05-增量预训练PT/README.md)、[L06](../lessons/L06-有监督微调SFT/README.md) |
| LoRA / QLoRA | [L07](../lessons/L07-LoRA与QLoRA高效微调/README.md) |
| RM / PPO / RLHF | [L08](../lessons/L08-奖励模型RM/README.md)、[L09](../lessons/L09-强化学习PPO与RLHF/README.md) |
| DPO / ORPO / GRPO | [L10](../lessons/L10-直接偏好优化DPO/README.md)、[L11](../lessons/L11-ORPO与GRPO前沿方法/README.md) |
| 数据 / 分布式 / 部署 | [L13](../lessons/L13-数据处理与质量工程/README.md)、[L14](../lessons/L14-分布式训练与DeepSpeed/README.md)、[L15](../lessons/L15-模型评估与推理部署/README.md) |
