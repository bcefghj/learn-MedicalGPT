[← 上一课](../L03-环境搭建与工具链/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L05-增量预训练PT/README.md)

---

# L04 MedicalGPT 项目全景

> **一句话精髓**：**先看地图，再走路，不会迷路**——MedicalGPT 把「医疗领域大模型」训练拆成清晰阶段与脚本，你脑子里先有流水线，再打开 `.py` 才不会懵。

---

## 本课你将学到什么

```
  · MedicalGPT 是什么、解决什么问题
  · 为什么要做医疗大模型（动机）
  · PT → SFT → RLHF/DPO 在仓库里如何对应
  · 关键 Python 脚本与 run_*.sh 分工
  · 支持的底座模型家族（Qwen / LLaMA / Baichuan …）
  · 核心依赖与社区影响力
  · 简历/面试 30 秒 & 2 分钟版本介绍
  · 面试高频考点
```

---

## 一、MedicalGPT 是什么？（一段话讲清楚）

**MedicalGPT** 是开源项目 [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT) 的实现与文档集合：它在通用大模型（如 LLaMA、Qwen、Baichuan 等）之上，提供一套接近 **ChatGPT 训练流水线** 的脚本——包括可选的**增量预训练（PT）**、**有监督微调（SFT）**、**奖励模型（RM）**、**PPO 强化学习（RLHF）**、**DPO/ORPO 等偏好对齐**，以及 **Gradio 演示、推理脚本、基于文件的 RAG（ChatPDF）**。你可以把它理解为：**把「领域数据 + 对齐方法」灌进开源底座模型的一整套工程脚手架**，医疗是其代表性应用场景与示例数据方向之一。

---

## 二、项目背景与动机：为什么要做医疗大模型？

### 2.1 现实痛点（产品与研究视角）

```
  通用大模型
      |
      +-- 医学术语、指南表述、院内文书风格 → 分布不一致
      |
      +-- 问诊、病历、科研场景 → 格式与安全要求更高
      |
      +-- 幻觉在医疗上代价极大 → 需要数据与流程双约束
```

### 2.2 开源 MedicalGPT 的价值

| 点 | 说明 |
|----|------|
| **可复现流水线** | PT/SFT/RM/PPO/DPO/ORPO 有脚本与示例 |
| **与 HF 生态对齐** | Transformers + PEFT + TRL 等 |
| **可替换底座** | 支持多系列模型（见后文列表） |
| **持续更新** | 版本日志覆盖 Qwen2.5、Llama3、GRPO 等 |

### 2.3 类比

> 通用模型像**通识教材**；MedicalGPT 这套流程像**带实习的专科培养方案**——先补领域语料（PT），再练答题格式（SFT），再纠正「说话方式」（对齐）。

### 2.4 合规声明（必读心态）

本课程与上游项目均**不**构成医疗建议。任何真实临床应用须遵循法规、伦理与院内流程；技术学习聚焦**方法与工程**。

---

## 三、核心训练流程：PT → SFT → RLHF / DPO

### 3.1 ASCII 总览（建议默画）

```
   ┌──────────────────┐
   │  可选：增量预训练   │  PT：继续「读」医疗/领域文本
   │   Continue PT    │      让分布更贴近领域
   └────────┬─────────┘
            |
            v
   ┌──────────────────┐
   │   有监督微调 SFT   │  指令数据：教模型「怎么答」
   │  Supervised FT   │
   └────────┬─────────┘
            |
     +------+------+
     |             |
     v             v
┌─────────┐   ┌──────────┐
│  RLHF   │   │   DPO    │  二选一或按项目组合
│ RM+PPO  │   │ 直接偏好  │
└─────────┘   └──────────┘
     |             |
     v             v
   更「听话/安全/有用」的对话模型
```

### 3.2 RLHF 子阶段（经典三件套）

```
  SFT 模型
      |
      v
  训练 Reward Model（人类偏好排序数据）
      |
      v
  PPO 用奖励信号更新策略（policy）
```

### 3.3 DPO 路线（工程上常更轻）

```
  偏好对数据（chosen vs rejected）
      |
      v
  直接优化策略模型相对参考模型的偏好目标
      |
      v
  无需在线 PPO 采样环（实现成本常更低）
```

### 3.4 ORPO（仓库亦支持）

**ORPO** 试图在**更少阶段**内把 SFT 与偏好对齐结合（详见上游论文与 `orpo_training.py`）。面试可说：**单阶段整合、无 ref 模型等 trade-off**，细节见 L11。

### 3.5 与 Karpathy「State of GPT」的关系

上游 README 写明：RLHF pipeline 思想参考 **Andrej Karpathy** 的演讲材料 [State of GPT PDF](https://karpathy.ai/stateofgpt.pdf) 与对应视频——这是理解「产品级 ChatGPT 如何炼成」的**绝佳课外读物**。

---

## 四、项目代码结构：关键文件与作用

下列文件均指 **MedicalGPT 上游仓库**根目录下常见入口（以 [GitHub 主分支](https://github.com/shibing624/MedicalGPT) 为准，版本迭代可能新增脚本）。

### 4.1 `pretraining.py`（增量预训练 / Continue PT）

```
  作用：在领域语料上继续做「语言建模」式训练
  典型输入：大规模纯文本 / 领域文档 token 化后的数据集
  你关心：学习率、max length、是否 Deepspeed、词表扩充等
```

**类比**：学生考完通识后，又刷了一整套**专业课本**——不一定立刻会答题，但**术语与句式**更熟。

### 4.2 `supervised_finetuning.py`（SFT）

```
  作用：用「指令-回答」或对话格式数据微调
  典型：ShareGPT / Alpaca / 自建问诊单轮或多轮
  你关心：template_name、多轮模板、LoRA 目标模块、max_length
```

**类比**：开始刷**题库**，学会按考试格式写答案。

### 4.3 `reward_modeling.py`（RM）

```
  作用：输入一对回答，输出「哪个更好」的分数/排序信号
  数据：人类偏好、规则构造的偏好对
  你关心：是否与 PEFT 兼容、分类头、loss 形式
```

**类比**：训练一位**阅卷老师**，后面 RL 用老师的打分来改学生。

### 4.4 `ppo_training.py`（PPO / RLHF）

```
  作用：策略模型生成 → RM 打分 → PPO 更新
  特点：工程组件多（policy、ref、RM、KL 惩罚等）
  你关心：batch、生成长度、KL、学习率、稳定性
```

**类比**：学生交卷 → 老师打分 → 根据分数调整写作策略；**循环多次**。

### 4.5 `dpo_training.py`（DPO）

```
  作用：用偏好数据直接优化，无需 PPO 内环采样
  你关心：beta、ref_model、序列 mask、双路 logprob
```

**类比**：不用每天模拟考，直接告诉学生「这篇作文比那篇好」，从对比里学。

### 4.6 `orpo_training.py`（ORPO）

```
  作用：ORPO 偏好优化相关训练入口（见官方 run_orpo.sh）
  你关心：与 SFT 一体化训练叙事、是否需要 ref 等差异
```

### 4.7 `run_*.sh` 脚本

```
  run_pt.sh      -> 调用 pretraining.py 的参数预设
  run_sft.sh     -> SFT 典型超参与路径
  run_rm.sh      -> 奖励模型训练
  run_ppo.sh     -> PPO 训练
  run_dpo.sh     -> DPO 训练
  run_orpo.sh    -> ORPO 训练
```

**学习法**：先读 `.sh` 里**模型路径、数据路径、关键 flag**，再进 `.py` 找对应 `ArgumentParser`。

### 4.8 `gradio_demo.py`

```
  作用：浏览器里与模型聊天，验证微调效果
  典型：--base_model / --lora_model / --template_name
```

**类比**：**试驾车**，给客户/导师演示生成质量。

### 4.9 `inference.py`

```
  作用：命令行交互或非交互生成，加载基座 + 可选 LoRA
  用途：批量试 prompt、写评测脚本前的手工检查
```

### 4.10 `chatpdf.py`

```
  作用：结合知识库文件的问答（RAG 向），提升领域问答准确率
  关联：上游说明链接 ChatPDF 思路
```

**类比**：允许**开卷考试**——先检索相关页再回答，降低胡编概率。

### 4.11 其他可能见到的脚本（版本迭代）

官方更新日志提到 **GRPO** 等训练方法；若你本地仓库存在 `grpo_training.py` 或类似入口，应以 **Release 说明**为准。面试可说：**「仓库持续跟进 TRL/社区新对齐算法」**。

### 4.12 文档与 Notebook（强烈建议扫一眼）

| 资源 | 作用 |
|------|------|
| `docs/training_params.md` | 训练参数说明 |
| `docs/datasets.md` | 数据集格式与来源线索 |
| `run_training_dpo_pipeline.ipynb` | Colab 约 15 分钟串 PT+SFT+DPO |
| `run_training_ppo_pipeline.ipynb` | Colab 约 20 分钟串 PT+SFT+RLHF |

---

## 五、支持的模型列表（节选与归类）

上游 README 维护 **Supported Models** 大表（模板名、LoRA 目标模块各不相同）。下面按**家族**归类，便于记忆（细节以官方表格为准）：

| 家族 | 示例规模 / 备注 |
|------|-----------------|
| **LLaMA / LLaMA2 / LLaMA3** | 7B、13B、70B 等 |
| **Qwen / Qwen1.5 / Qwen2 / Qwen2.5** | 0.5B～110B 等多档 |
| **Baichuan / Baichuan2** | 7B / 13B |
| **ChatGLM / ChatGLM2 / ChatGLM3** | 6B 档 |
| **BLOOMZ** | 多规格至极大 |
| **Mistral / Mixtral** | 7B、MoE 8x7B 等 |
| **DeepSeek / DeepSeek3** | 多种规模 |
| **InternLM2、Yi、XVERSE、Orion、Cohere** | 视许可与场景选用 |

**面试提示**：能说清 **「模板 template 与模型家族绑定」** ——写错 chat 模板会导致**训练/推理格式不一致**，效果诡异。

---

## 六、核心依赖库：Transformers / PEFT / DeepSpeed / TRL

```
  Transformers  —— 模型、Tokenizer、Trainer
  PEFT          —— LoRA / 适配器，降显存微调
  DeepSpeed     —— 大模型训练优化（ZeRO 等），按需启用
  TRL           —— DPO、PPO 等对齐训练组件
  datasets      —— 数据管道
  accelerate    —— 分布式与混合精度辅助
```

**类比**：**Transformers** 是车架；**PEFT** 是轻量化改装件；**DeepSpeed** 是挂车与副油箱；**TRL** 是驾校高级课程。

---

## 七、项目的 Star 数与影响力

GitHub 仓库首页的 **Stars** 徽章会随时间变化（README 中带有 `shields.io/github/stars` 动态图标）。**本讲义不硬编码具体数字**，以免过时；请直接查看：

- 仓库：[github.com/shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)  
- Star 历史：[star-history.com 上的 MedicalGPT](https://star-history.com/#shibing624/MedicalGPT&Timeline)

**定性描述**：MedicalGPT 是中文社区**较早系统化公开医疗大模型训练流水线**的项目之一，**Star 与 Issue/PR 活跃度**在「领域 LLM 训练模板」类仓库中具备**代表性**，常被学习者用于简历项目与复现基线。

---

## 八、你需要重点关注的文件（学习路径）

### 8.1 第一周（建立全局）

```
  README.md（上游）        -> 硬件表、支持模型、脚本映射
  run_sft.sh              -> 最常用入门路径
  supervised_finetuning.py-> 读参数与数据流
  inference.py / gradio_demo.py -> 验效果
```

### 8.2 第二周（对齐与进阶）

```
  dpo_training.py + run_dpo.sh
  reward_modeling.py + run_rm.sh
  ppo_training.py + run_ppo.sh
```

### 8.3 数据与文档

```
  docs/datasets.md
  docs/training_params.md
```

### 8.4 ASCII：学习顺序

```
  README -> run_sft.sh -> supervised_finetuning.py -> inference
              |
              +--> run_dpo.sh -> dpo_training.py
              |
              +--> run_ppo.sh -> ppo_training.py（先理解 RM）
```

---

## 九、面试时如何介绍这个项目

### 9.1 30 秒版本（背熟）

> 我主要学习并复现了开源 **MedicalGPT**：它在通用大模型底座上，走 **可选 PT + SFT + 偏好对齐（DPO 或经典 RLHF）** 的流水线，让模型更适配医疗文本与对话。我熟悉关键入口如 `supervised_finetuning.py`、`dpo_training.py`，能用 `inference.py` 或 `gradio_demo.py` 验证生成；显存有限时会优先 **LoRA/QLoRA**，并对照官方 **显存表**做配置。

### 9.2 2 分钟版本（STAR 骨架）

- **背景**：通用 LLM 在医疗场景存在**术语分布差异、格式与安全要求高**等问题，需要领域适配与对齐。  
- **项目**：MedicalGPT 基于 HuggingFace 生态，提供 **PT/SFT/RM/PPO/DPO/ORPO** 等脚本与 `run_*.sh` 示例，支持 **Qwen、LLaMA、Baichuan** 等多底座。  
- **我的工作**（按真实填写）：例如「复现 SFT + DPO」「清洗 x 条指令数据」「记录 loss 与样例对比」「OOM 时改 LoRA rank 与 max_length」。  
- **结果**：用**定性样例 + 简单指标**说明提升（若无权威榜，强调**可复现与消融**）。  
- **延伸**：提一句 **RAG（chatpdf.py）** 缓解幻觉、以及 **合规与人工审核** 意识。

---

## 十、面试考点

```
  1. PT / SFT / RLHF / DPO 各自解决什么问题？
  2. 为什么 SFT 之后还要 DPO 或 PPO？
  3. RM 在 RLHF 里的角色？没有 RM 能否训练？
  4. DPO 相对 PPO 的工程优缺点？
  5. LoRA 目标模块为何因模型而异（q_proj vs W_pack）？
  6. template 写错会怎样？
  7. OOM 时你如何调参（结合官方显存表）？
  8. chatpdf.py 与纯微调的边界（RAG vs parametric memory）？
  9. 医疗场景如何降低幻觉风险（数据、检索、拒答、评测）？
 10. 你如何验证一次训练「真的生效」？
```

---

## 十一、推荐阅读

| 资源 | 链接 |
|------|------|
| MedicalGPT 源码 | [github.com/shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT) |
| Wiki | [MedicalGPT Wiki](https://github.com/shibing624/MedicalGPT/wiki) |
| DPO 论文 | [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290.pdf) |
| ORPO 论文 | [ORPO](https://arxiv.org/abs/2403.07691) |
| State of GPT | [PDF](https://karpathy.ai/stateofgpt.pdf) |
| HF 作者页模型 | [huggingface.co/shibing624](https://huggingface.co/shibing624) |

---

## 附录 A：Pipeline 与脚本对照速查表

| 阶段 | Python 脚本 | Shell 示例 |
|------|-------------|------------|
| 增量预训练 | `pretraining.py` | `run_pt.sh` |
| 有监督微调 | `supervised_finetuning.py` | `run_sft.sh` |
| 奖励模型 | `reward_modeling.py` | `run_rm.sh` |
| PPO | `ppo_training.py` | `run_ppo.sh` |
| DPO | `dpo_training.py` | `run_dpo.sh` |
| ORPO | `orpo_training.py` | `run_orpo.sh` |

---

## 附录 B：你可能混淆的概念

| 混淆 | 澄清 |
|------|------|
| PT = 指令学习？ | PT 主要是**续写式**建模；指令遵循主要靠 SFT |
| SFT 能解决幻觉？ | 能缓解风格与格式；**事实性**还要 RAG、数据、评测 |
| DPO 不要奖励模型？ | **不显式训练 RM**；偏好仍来自数据与隐式目标 |

---

## 附录 C：本课与后续课程映射

```
  L04 全景
    -> L05 PT
    -> L06 SFT
    -> L07 LoRA/QLoRA
    -> L08 RM
    -> L09 PPO
    -> L10 DPO
    -> ...
```

---

## 附录 D：自检清单

- [ ] 能不看资料说出 PT、SFT、DPO 各一件事。  
- [ ] 能解释 `run_sft.sh` 与 `supervised_finetuning.py` 关系。  
- [ ] 知道 `gradio_demo.py` 与 `inference.py` 的使用场景差异。  
- [ ] 能口述 30 秒项目介绍。  

---

## 附录 E：数据在仓库里「流」起来是什么样？

```
  原始语料 / 指令 JSON / 偏好对
            |
            v
      datasets 加载 + 预处理（脚本内或外部）
            |
            v
      DataCollator / template 格式化为模型输入
            |
            v
      Trainer / TRL Trainer 训练循环
            |
            v
      checkpoint 目录（config + 权重 + tokenizer）
            |
            v
   inference.py / gradio_demo.py 加载验证
```

**小白提示**：第一次不要纠结每一行代码，先记住：**数据 → 模板 → 训练器 → 权重 → 推理** 这条龙骨。

---

## 附录 F：`run_*.sh` 里通常藏了哪些信息？

| 信息类型 | 为什么重要 |
|----------|------------|
| `CUDA_VISIBLE_DEVICES` | 指定用哪张卡 |
| `--base_model` / 数据路径 | 复现实验的「锚点」 |
| `per_device_train_batch_size` | OOM 第一反应参数 |
| `gradient_accumulation_steps` | 小显存凑大有效 batch |
| `use_peft` / LoRA rank | 是否走参数高效微调 |
| `model_max_length` | 显存与任务上下文折中 |

**类比**：`.sh` 是**外卖订单备注**——真正做菜在 `.py`，但备注写错会送错地址。

---

## 附录 G：Colab Notebook 与本地训练怎么选？

| 场景 | Colab | 本地 / 云 GPU |
|------|-------|----------------|
| 体验流水线 | 极适合（官方 Notebook） | 也可 |
| 长时大规模 PT | 易中断 | 更合适 |
| 数据敏感 | 需谨慎（上传策略） | 更易合规隔离 |

---

## 附录 H：零基础常见追问（对话体）

**问：MedicalGPT 是一个新模型吗？**  
答：它首先是**训练框架与流程**；发布页上也有作者训练的**示例权重**，但你可以换任意兼容底座。

**问：我只跑 SFT 不跑 PT 可以吗？**  
答：可以。PT 常是**可选**，看领域分布偏移是否大到值得花算力。

**问：DPO 一定比 RLHF 好吗？**  
答：**不**。DPO 工程更简单是常见优势；最终看**数据质量、任务、稳定性**。

**问：`chatpdf.py` 能替代微调吗？**  
答：**不能简单替代**。RAG 解决「可查的外部知识」；微调解决「风格、格式、领域习惯与指令遵循」。真实产品常 **RAG + 微调** 组合。

**问：我要背下所有超参吗？**  
答：不用。背**量级与方向**（如更大 `max_length` 更吃显存），细节查 `docs/training_params.md`。

---

## 附录 I：术语迷你表

| 缩写 | 含义 |
|------|------|
| PT | Pre-Training，此处多指 Continue PT |
| SFT | Supervised Fine-Tuning |
| RM | Reward Model |
| PPO | Proximal Policy Optimization |
| DPO | Direct Preference Optimization |
| ORPO | Odds Ratio Preference Optimization |
| RLHF | Reinforcement Learning from Human Feedback |
| RAG | Retrieval-Augmented Generation |
| PEFT | Parameter-Efficient Fine-Tuning |
| LoRA | Low-Rank Adaptation |

---

## 附录 J：「能讲清楚」vs「跑过一遍」

```
  只跑过：能启动命令，但说不清每阶段输入输出
            |
            v
  及格线：能画 PT-SFT-对齐 图 + 指到具体脚本
            |
            v
  加分项：能结合显存表谈 LoRA/QLoRA 选型 + 举幻觉与 RAG 例子
```

---

[← 上一课](../L03-环境搭建与工具链/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L05-增量预训练PT/README.md)
