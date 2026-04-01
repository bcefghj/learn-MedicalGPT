# MedicalGPT 学习资源汇总

> 本页面收集整理了 MedicalGPT 及医疗大模型领域的高质量学习资源，帮助你系统性掌握整个技术栈。

---

## 一、MedicalGPT 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| MedicalGPT 源码仓库 | [GitHub](https://github.com/shibing624/MedicalGPT) | 5.1K+ Star，完整训练 Pipeline |
| 官方 Wiki | [Wiki](https://github.com/shibing624/MedicalGPT/wiki) | FAQ、数据集说明、训练参数文档 |
| 数据集 Wiki | [数据集](https://github.com/shibing624/MedicalGPT/wiki/%E6%95%B0%E6%8D%AE%E9%9B%86) | PT/SFT/RM/DPO 各阶段数据说明 |
| Colab DPO Pipeline | [Notebook](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_dpo_pipeline.ipynb) | 15 分钟跑通 PT+SFT+DPO |
| Colab PPO Pipeline | [Notebook](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_ppo_pipeline.ipynb) | 20 分钟跑通 PT+SFT+RLHF |
| HuggingFace 模型 | [Models](https://huggingface.co/shibing624) | 项目发布的预训练模型权重 |
| 医疗数据集 | [Dataset](https://huggingface.co/datasets/shibing624/medical) | 240 万条中文医疗问答数据 |

---

## 二、GitHub 优秀学习项目

### 2.1 大模型训练与微调

| 项目 | Star | 说明 | 推荐理由 |
|------|------|------|----------|
| [MiniMind](https://github.com/jingyaogong/minimind) | 45K+ | 2 小时从零训练 64M 参数 GPT | 纯 PyTorch，极低成本，面试首选实战项目 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 40K+ | 一站式大模型微调框架 | 支持 100+ 模型，工业级工具 |
| [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) | 3K+ | ChatGLM 高效微调 | LoRA/P-Tuning 实践参考 |
| [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | 18K+ | 中文 LLaMA & Alpaca | 中文增量预训练经典案例 |

### 2.2 医疗 AI 专项

| 项目 | Star | 说明 | 推荐理由 |
|------|------|------|----------|
| [HuatuoGPT](https://github.com/FreedomIntelligence/HuatuoGPT) | 1K+ | 华佗 GPT 医疗大模型 | RLHF 医疗场景落地 |
| [ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor) | 3K+ | 医疗对话模型 | LLaMA 医疗微调典型案例 |
| [DoctorGLM](https://github.com/xionghonglin/DoctorGLM) | 1K+ | 基于 ChatGLM 的医疗模型 | 中文医疗 SFT 参考 |
| [BenTsao (本草)](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) | 2K+ | 中文医疗 LLaMA | 医疗知识图谱 + LLM |
| [Med-PaLM](https://arxiv.org/abs/2305.09617) | — | Google 医疗大模型论文 | 医疗评测标杆 |

### 2.3 大模型知识体系

| 项目 | Star | 说明 | 推荐理由 |
|------|------|------|----------|
| [LLMSurvey](https://github.com/RUCAIBox/LLMSurvey) | 12K+ | 大模型综述论文配套 | 系统性理解 LLM 全貌 |
| [LLM-Action](https://github.com/liguodongiot/llm-action) | 10K+ | 大模型实战笔记 | 训练、推理、评测实操 |
| [awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) | 18K+ | LLM 资源大全 | 论文、框架、数据集一站式 |
| [LLM101n](https://github.com/karpathy/LLM101n) | 30K+ | Karpathy 从零造 LLM | 大神手把手教学 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | 36K+ | 微软分布式训练框架 | MedicalGPT 底层依赖 |

---

## 三、技术博客与教程文章

### 3.1 MedicalGPT 专项

| 文章标题 | 平台 | 链接 | 要点 |
|---------|------|------|------|
| 中文医疗大模型训练全流程源码剖析 | 微信公众号 | [阅读](https://mp.weixin.qq.com/s/DTHIxyDb9vG793hAKGLt2g) | PT/SFT/RM/RL 四阶段源码解析 |
| MedicalGPT 入门指南 | 懂AI | [阅读](https://www.dongaigc.com/a/medicalgpt-introduction-guide-chatgpt-training) | 项目架构与快速上手 |
| MedicalGPT 全参微调技术解析 | GitCode | [阅读](https://blog.gitcode.com/c7b1a162cb201a90682ecb5c65da9822.html) | 全参数微调参数与实践 |
| MedicalGPT 大规模模型训练与长文本处理 | GitCode | [阅读](https://blog.gitcode.com/5092aa02b07d46484fc2d3688d3aea65.html) | 67B 模型分布式训练 |
| 医疗大模型实战 MedicalGPT 项目记录 | CSDN | [阅读](https://blog.csdn.net/Wzxdecsdn/article/details/135489341) | 实战踩坑记录 |
| MedicalGPT 模型训练教程 | 魔乐社区 | [阅读](https://modelers.csdn.net/69a6881c7bbde9200b9c59f8.html) | 配置参数详解 |

### 3.2 大模型面试八股文

| 文章标题 | 平台 | 链接 | 要点 |
|---------|------|------|------|
| 大模型面试 100 问：训练与优化篇 | 80AJ | [阅读](https://www.80aj.com/2026/01/04/llm-interview-training-optimization/) | 100 题系统覆盖 |
| RLHF 夺命连环 17 问 | CSDN/DeepSeek 社区 | [阅读](https://deepseek.csdn.net/67b53c5c4d0686499adf4e1d.html) | RLHF 深度追问 |
| RLHF 八股总结 | 牛客网 | [阅读](https://www.nowcoder.com/feed/main/detail/20e8f456d0c5418cad2b46b39c0d0f61) | 面经实录 |
| DPO vs RLHF 对齐之争 | CSDN | [阅读](https://blog.csdn.net/sinat_37574187/article/details/145964594) | 对齐方法深度对比 |

### 3.3 大模型通用教程

| 文章标题 | 平台 | 链接 | 要点 |
|---------|------|------|------|
| State of GPT | Karpathy | [PDF](https://karpathy.ai/stateofgpt.pdf) | RLHF Pipeline 经典演讲 |
| LoRA 原论文 | arXiv | [论文](https://arxiv.org/abs/2106.09685) | LoRA 核心原理 |
| DPO 原论文 | arXiv | [论文](https://arxiv.org/abs/2305.18290) | 直接偏好优化 |
| DeepSeek R1 技术报告 | DeepSeek | [论文](https://arxiv.org/abs/2501.12948) | 2025 前沿：GRPO + 推理 |
| ORPO 原论文 | arXiv | [论文](https://arxiv.org/abs/2403.07691) | 无参考模型偏好优化 |

---

## 四、视频教程

| 视频 | 平台 | 时长 | 说明 |
|------|------|------|------|
| Andrej Karpathy: Let's build GPT | YouTube | 2h | 从零搭建 GPT，理解 Transformer |
| 李沐 Transformer 论文精读 | B 站 | 1h | 中文最佳 Transformer 讲解 |
| RLHF 从入门到精通 | B 站 | 3h | PPO/DPO 系列讲解 |
| MiniMind 从零训练 LLM 全流程 | B 站 | 2h | 配合 MiniMind 项目实操 |
| DeepSpeed 实战教程 | B 站 | 1.5h | 分布式训练配置详解 |
| HuggingFace TRL 微调教程 | YouTube | 1h | SFT/DPO/PPO 一站式微调 |

---

## 五、核心论文清单（面试必读）

按优先级排序，面试重点标注 **[必读]**：

### 5.1 基础架构
1. **[必读]** Attention Is All You Need (2017) — Transformer 架构
2. **[必读]** BERT / GPT / GPT-2 / GPT-3 系列 — 预训练范式演进
3. Language Models are Few-Shot Learners (GPT-3) — In-context Learning

### 5.2 训练方法
4. **[必读]** LoRA: Low-Rank Adaptation (2021) — 高效微调
5. **[必读]** QLoRA: Efficient Finetuning (2023) — 4-bit 量化微调
6. Training Language Models to Follow Instructions (InstructGPT) — RLHF 流程

### 5.3 对齐技术
7. **[必读]** DPO: Direct Preference Optimization (2023) — 简化对齐
8. **[必读]** RLHF: Learning from Human Feedback — 强化学习对齐
9. ORPO (2024) — 无参考模型优化
10. GRPO (DeepSeek, 2025) — 分组相对策略优化

### 5.4 推理与部署
11. **[必读]** vLLM: PagedAttention (2023) — 高效推理
12. FlashAttention (2022/2023) — 显存优化注意力
13. Speculative Decoding — 投机解码加速

### 5.5 医疗 AI
14. Med-PaLM 2 (Google, 2023) — 医疗问答 SOTA
15. PMC-LLaMA (2023) — 医学预训练
16. ChatDoctor (2023) — 医疗对话微调

---

## 六、学习路径推荐

### 路径 A：零基础入门（4 周）

```
第 1 周：基础概念
├── 看 Karpathy Let's build GPT 视频
├── 读 L01-L04 课程
└── 克隆 MedicalGPT 仓库，跑通 Gradio Demo

第 2 周：训练核心
├── 精读 L05-L07（PT/SFT/LoRA）
├── Colab 跑通 DPO Pipeline
└── 对照源码理解 pretraining.py / supervised_finetuning.py

第 3 周：对齐与工程
├── 精读 L08-L11（RM/PPO/DPO/ORPO/GRPO）
├── 精读 L12-L15（数据/分布式/部署）
└── 跑通 MiniMind 全流程（预训练+SFT+DPO）

第 4 周：面试冲刺
├── 精读 L19-L20 + interview/ 速查
├── 背熟 cheatsheets/ 速查表
└── 模拟面试 3-5 轮
```

### 路径 B：有基础速通（2 周）

```
第 1 周：核心内容
├── L04 全景 → L05-L11 训练Pipeline
├── Colab 实战 → MiniMind 实战
└── 精读 MedicalGPT 源码（L18）

第 2 周：面试突击
├── L19 简历包装 + L20 面试通关
├── interview/ 150+ 题速查
└── 模拟面试 + 查漏补缺
```

### 路径 C：面试突击（5 天）

```
Day 1: L04 全景 + L19 简历包装
Day 2: interview/ 速查 + cheatsheets/
Day 3: L20 面试通关 80 题详解（上）
Day 4: L20 面试通关 80 题详解（下）+ 前沿技术
Day 5: 模拟面试 + 薄弱点查漏
```

---

## 七、工具与环境

| 工具 | 用途 | 链接 |
|------|------|------|
| PyTorch | 深度学习框架 | [pytorch.org](https://pytorch.org) |
| HuggingFace Transformers | 模型加载与微调 | [huggingface.co](https://huggingface.co/docs/transformers) |
| HuggingFace TRL | SFT/DPO/PPO 训练 | [TRL](https://huggingface.co/docs/trl) |
| PEFT | LoRA/QLoRA 微调 | [PEFT](https://huggingface.co/docs/peft) |
| DeepSpeed | 分布式训练 | [deepspeed.ai](https://www.deepspeed.ai/) |
| vLLM | 高效推理引擎 | [vllm.ai](https://docs.vllm.ai/) |
| Gradio | Web Demo 搭建 | [gradio.app](https://www.gradio.app/) |
| Weights & Biases | 实验追踪 | [wandb.ai](https://wandb.ai) |

---

> 提示：资源在精不在多。建议先完成本仓库 20 课体系，遇到不懂再查阅上述资源。面试前重点刷 interview/ 和 cheatsheets/。
