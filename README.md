# Learn MedicalGPT — 从零基础到写进简历

**用 20 节课 + MiniMind 实战，彻底搞懂医疗大模型训练全流程，面试对答如流。**

> 本项目参考 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 的完整训练 Pipeline，
> 从"什么是大模型"讲起，一路带你走到"面试官随便问"的水平。
> 同时配套 [MiniMind](https://github.com/jingyaogong/minimind) 从零训练实战，让你既懂「怎么用」也懂「怎么做」。
> 
> 最终目标：**把 MedicalGPT + MiniMind 写进简历，面试时自信讲解每一个技术细节。**

![开篇漫画](comics/comic-01-opening.png)

### 交互式学习平台（带动画！）

```sh
cd web && npm install && npm run dev   # 访问 http://localhost:3000
```

包含训练 Pipeline 动画演示、LoRA 原理动画、RLHF vs DPO 对比动画、算法对比面板等交互式可视化。

---

```
                    MedicalGPT 训练全流程
                    =====================

    海量医疗文本          指令微调数据          人类偏好数据
         |                      |                      |
         v                      v                      v
    +---------+          +---------+          +--------------+
    |  阶段一  |          |  阶段二  |          |    阶段三     |
    |   PT    | -------> |   SFT   | -------> | RLHF/DPO/..  |
    | 增量预训练|          |有监督微调 |          |  偏好对齐     |
    +---------+          +---------+          +--------------+
         |                      |                      |
         v                      v                      v
     模型学会                模型学会               模型学会
    "读"医疗文本           "答"医疗问题           "好"的回答方式
```

---

## 项目全景导航

### 核心学习内容

| 板块 | 内容 | 适合人群 |
|------|------|---------|
| [20 节课程体系](lessons/) | LLM 基础 → Transformer → PT/SFT/LoRA → RLHF/DPO → 数据/分布式/部署 | 系统学习 |
| [MiniMind 实战指南](minimind/) | 从零训练 64M 参数 GPT，纯 PyTorch 全流程 | 动手实践 |
| [面试速查 150+ 题](interview/) | 关键词版面试题，面试前 30 分钟速查 | 面试突击 |
| [速查表](cheatsheets/) | 核心概念、命令、超参一页纸 | 随时参考 |

### 求职导向内容

| 板块 | 内容 | 适合人群 |
|------|------|---------|
| [岗位市场与面经](job-market/) | 牛客/小红书面经 + Boss 直聘 JD 分析 + 薪资参考 | 求职准备 |
| [学习资源汇总](resources/) | GitHub 项目、博客教程、论文清单、视频课程 | 拓展学习 |
| [简历包装 (L19)](lessons/L19-简历包装与项目描述/) | STAR 法则写项目经验 | 写简历 |
| [MiniMind 简历包装](minimind/04-简历包装.md) | 三个版本的简历模板 | 写简历 |
| [MiniMind STAR 面试稿](minimind/05-STAR面试稿.md) | 30 秒/1 分钟/3 分钟项目介绍 | 面试练习 |
| [MiniMind 面试 100 题](minimind/06-面试问答100题.md) | 项目可能被问到的所有问题 | 深度准备 |

---

## 学习路线图

```
Phase 1: 基础入门                    Phase 2: 训练 Pipeline 核心
========================             ================================
L01  什么是大语言模型         [零基础]   L05  增量预训练 PT             [核心]
     从 ChatGPT 聊起                        让模型学会"读"医疗文本
     |                                      |
     +-> L02  Transformer 架构   [核心]     L06  有监督微调 SFT          [核心]
              注意力机制的直觉理解               教模型"回答"医疗问题
              |                                 |
         L03  环境搭建与工具链   [实操]     L07  LoRA 与 QLoRA          [核心]
              Python/PyTorch/HF                  用小显存做大事
              |                                 |
         L04  MedicalGPT 全景    [地图]     L08  奖励模型 RM            [进阶]
              鸟瞰整个项目                       让模型学会"好坏"
                                                |
                                           L09  PPO 与 RLHF           [进阶]
                                                让模型越来越好
                                                |
                                           L10  DPO 直接偏好优化        [进阶]
                                                更简单的对齐方式
                                                |
                                           L11  ORPO 与 GRPO          [前沿]
                                                最新优化方法

Phase 3: 数据与工程                    Phase 4: 实战进阶
========================             ========================
L12  医疗数据集详解          [数据]     L16  Colab 动手全流程      [实战]
     数据是一切的基石                        PT + SFT + DPO 跑通
     |                                      |
L13  数据处理与质量工程      [数据]     L17  RAG 检索增强生成      [应用]
     打造高质量训练数据                      让医疗问答更精准
     |                                      |
L14  分布式训练 DeepSpeed    [工程]     L18  源码逐行精读          [深入]
     多卡训练实战                            理解每一行代码
     |
L15  模型评估与推理部署      [工程]
     从训练到上线


Phase 5: 面试冲刺                     Phase 6: MiniMind 实战
========================             ========================
L19  简历包装与项目描述      [求职]     M01  MiniMind 项目全景    [实战]
     如何让面试官眼前一亮                    64M 参数 GPT 全貌
     |                                      |
L20  面试通关高频考点        [求职]     M02  从零搭建指南         [实战]
     150+ 道真题全覆盖                       手把手跑通全流程
                                             |
                                        M03  源码精读             [深入]
                                             逐行理解核心代码
                                             |
                                        M04  简历包装             [求职]
                                             STAR 法写项目
                                             |
                                        M05  STAR 面试稿          [求职]
                                             三种版本口述模板
                                             |
                                        M06  面试 100 题          [求职]
                                             全部可能的问题
```

---

## 每节课一句话

| # | 课程 | 一句话精髓 | 难度 |
|---|------|-----------|------|
| **L01** | 什么是大语言模型 | *"大模型就是一个超级学霸，读了互联网上所有的书"* | * |
| **L02** | Transformer 架构 | *"注意力机制 = 阅读时知道该重点看哪个词"* | ** |
| **L03** | 环境搭建与工具链 | *"磨刀不误砍柴工，环境搭好事半功倍"* | * |
| **L04** | MedicalGPT 全景 | *"先看地图，再走路，不会迷路"* | * |
| **L05** | 增量预训练 PT | *"先让模型读一千万篇医学论文"* | *** |
| **L06** | 有监督微调 SFT | *"读完书还不够，得做题才会答题"* | *** |
| **L07** | LoRA 与 QLoRA | *"不用改整本书，只需贴几张便签纸"* | *** |
| **L08** | 奖励模型 RM | *"训练一个'老师'来打分"* | **** |
| **L09** | PPO 与 RLHF | *"用'老师'的评分不断改进'学生'的回答"* | **** |
| **L10** | DPO 直接偏好优化 | *"不要老师了，直接从好坏对比中学"* | *** |
| **L11** | ORPO 与 GRPO | *"更高效：一步到位学会对齐"* | **** |
| **L12** | 医疗数据集 | *"数据决定上限，算法决定下限"* | ** |
| **L13** | 数据处理与质量 | *"Garbage in, garbage out"* | *** |
| **L14** | 分布式训练 | *"一张卡不够？那就八张一起上"* | **** |
| **L15** | 评估与部署 | *"训练好了不上线，等于白训练"* | *** |
| **L16** | Colab 全流程 | *"纸上得来终觉浅，动手才是真功夫"* | ** |
| **L17** | RAG 检索增强 | *"让模型先查资料再回答，准确率翻倍"* | *** |
| **L18** | 源码精读 | *"读源码是高手和新手的分水岭"* | **** |
| **L19** | 简历包装 | *"好的项目描述 = STAR法则 + 量化数据"* | ** |
| **L20** | 面试通关 | *"面试官问的，这里全有答案"* | *** |

---

## 如何使用本仓库

### 零基础学习者

```
建议顺序：L01 → L02 → L03 → L04 → L05 → L06 → L07 → ... → L20 → MiniMind 全套
预计时间：3-4 周（每天 2-3 小时）
```

### 有一定基础的学习者

```
快速通道：L04(全景) → L05-L11(核心Pipeline) → MiniMind实战 → L19-L20(面试)
预计时间：1-2 周
```

### 面试突击

```
速成路线：L04(全景) → interview/(150+速查) → L19(简历) → L20(面试题)
         → MiniMind/05(STAR面试稿) → MiniMind/06(100题)
预计时间：3-5 天
```

---

## 项目结构

```
learn-MedicalGPT/
│
├── README.md                          <- 你在这里
│
├── comics/                            <- 哆啦A梦风格漫画插图
│   ├── comic-01-opening.png           <- 开篇：面对面试 BOSS
│   ├── comic-02-training-pipeline.png <- 训练流程：PT→SFT→DPO
│   ├── comic-03-lora.png             <- LoRA：只贴便签纸
│   ├── comic-04-minimind.png         <- MiniMind：3 元训练大模型
│   └── comic-05-interview.png        <- 面试：STAR 法从容应对
│
├── web/                               <- 交互式学习平台（Next.js + 动画）
│   ├── src/components/visualizations/ <- 训练Pipeline/LoRA/RLHF动画组件
│   └── src/components/timeline/       <- 学习路线时间轴动画
│
├── lessons/                           <- 20 节课程（核心内容）
│   ├── L01-什么是大语言模型/
│   ├── L02-Transformer架构核心原理/
│   ├── ...
│   ├── L19-简历包装与项目描述/
│   └── L20-面试通关高频考点/
│
├── minimind/                          <- MiniMind 实战专区（6 篇）
│   ├── 01-项目全景.md                  <- 项目架构与定位
│   ├── 02-从零搭建.md                  <- 手把手跑通全流程
│   ├── 03-源码精读.md                  <- 逐行理解核心代码
│   ├── 04-简历包装.md                  <- STAR 法写简历
│   ├── 05-STAR面试稿.md               <- 30s/1min/3min 口述模板
│   └── 06-面试问答100题.md             <- 100 题 STAR 格式详解
│
├── interview/                         <- 面试速查（150+ 题）
│   └── README.md
│
├── job-market/                        <- 岗位市场与面经
│   └── README.md                      <- JD 分析 + 面经 + 薪资
│
├── resources/                         <- 学习资源汇总
│   └── README.md                      <- GitHub/博客/论文/视频
│
├── cheatsheets/                       <- 速查表
│   └── README.md                      <- 核心概念一页纸速查
│
└── docs/                              <- 多格式输出
    └── ...
```

---

## 核心参考资源

| 资源 | 链接 | 说明 |
|------|------|------|
| MedicalGPT 源码 | [GitHub](https://github.com/shibing624/MedicalGPT) | 本课程学习的核心项目 |
| MiniMind 源码 | [GitHub](https://github.com/jingyaogong/minimind) | 从零训练 LLM 实战项目 |
| MedicalGPT Wiki | [Wiki](https://github.com/shibing624/MedicalGPT/wiki) | 官方文档、FAQ、数据集说明 |
| Colab DPO Pipeline | [Notebook](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_dpo_pipeline.ipynb) | 15 分钟跑通 PT+SFT+DPO |
| Colab PPO Pipeline | [Notebook](https://colab.research.google.com/github/shibing624/MedicalGPT/blob/main/run_training_ppo_pipeline.ipynb) | 20 分钟跑通 PT+SFT+RLHF |
| State of GPT (Karpathy) | [PDF](https://karpathy.ai/stateofgpt.pdf) | RLHF Pipeline 的来源演讲 |
| HuggingFace 模型库 | [Models](https://huggingface.co/shibing624) | 项目发布的预训练模型 |
| 医疗数据集 | [Dataset](https://huggingface.co/datasets/shibing624/medical) | 240 万条中文医疗数据 |

---

## Star History

如果这个项目对你有帮助，请点个 Star 支持一下！

---

## License

MIT License - 自由使用、学习、分享。

---

> **记住：学完这个仓库，你不只是"了解" MedicalGPT 和 MiniMind，你是能"讲清楚"它们的人。**
>
> **面试官问什么，你都能从原理到代码、从数据到部署，给出完整的回答。**
