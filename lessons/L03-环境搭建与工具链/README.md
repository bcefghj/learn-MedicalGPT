[← 上一课](../L02-Transformer架构核心原理/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L04-MedicalGPT项目全景/README.md)

---

# L03 环境搭建与工具链

> **一句话精髓**：**磨刀不误砍柴工，环境搭好事半功倍**——同样一份 MedicalGPT 代码，环境对齐的人「复制粘贴就能跑」，环境不对的人会在 `ImportError` 和 `CUDA OOM` 里卡很久。

---

## 本课你将学到什么

```
  ┌──────────────────────────────────────────────┐
  │ · Python 要会到什么程度才算够                │
  │ · PyTorch CPU/GPU 安装与自检                 │
  │ · CUDA / 驱动 / 显卡选型的心智模型          │
  │ · HuggingFace 全家桶各包干什么               │
  │ · conda / venv 环境与依赖锁定                │
  │ · Jupyter / Colab 的典型工作流               │
  │ · Git 够用的最小命令集                       │
  │ · 克隆 MedicalGPT、装依赖、跑通 import       │
  │ · 常见问题排障思路                           │
  │ · 官方显存表 + 面试环境题                    │
  └──────────────────────────────────────────────┘
```

---

## 一、Python 基础：需要会到什么程度？

### 1.1 必会（否则读训练脚本很痛苦）

| 技能 | 说明 |
|------|------|
| **语法** | 变量、分支、循环、函数、类的基础用法 |
| **数据结构** | list / dict / tuple，列表推导式 |
| **文件与路径** | `open`、`pathlib`、`os.environ` |
| **调试** | 会看 traceback，会 `print` / 断点 |
| **虚拟环境** | 知道「全局 pip」与「项目环境」的区别 |

### 1.2 建议会（做微调很快用到）

| 技能 | 说明 |
|------|------|
| **简单 numpy** | 形状、广播的直觉即可 |
| **读文档** | 会查 PyTorch / HF 官方文档与论坛 |
| **命令行** | `cd`、`ls`、`pip`、`python script.py` |

### 1.3 暂时不必深挖（遇到再学）

- C++ 扩展编译细节  
- CUDA kernel 手写  
- 分布式文件系统

**类比**：你要开车去外地，**会开、会看导航**就行；不必先学会造发动机。

---

## 二、PyTorch 安装与验证（CPU / GPU）

### 2.1 为什么训练大模型几乎总提 PyTorch？

生态成熟、研究代码多、与 **HuggingFace** 集成好。MedicalGPT 依赖 PyTorch。

### 2.2 CPU 版安装（适合先跑通 import 与小实验）

到官网按平台选择：[PyTorch Get Started](https://pytorch.org/get-started/locally/)

```bash
# 示例：仅 CPU（具体以官网生成命令为准）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 GPU 版安装（需要匹配 CUDA）

**心智模型**：

```
  显卡驱动 (Driver)
        |
        v
  CUDA 运行时（与 PyTorch wheel 匹配）
        |
        v
  PyTorch（带 CUDA 的构建）
```

到官网选择：**你的 CUDA 版本**与 **操作系统**，复制官方命令，**不要混用多个教程的旧命令**。

### 2.4 验证脚本（必跑）

```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device="cuda")
    print("tensor on cuda ok:", x.device)
```

**期望**：

- CPU 环境：`cuda available: False` 正常。  
- GPU 环境：`True` 且能创建 CUDA 张量。

### 2.5 面试考点

> **如何确认 GPU 可用？** —— `torch.cuda.is_available()` + 小张量 CUDA 运算 + 实际跑一步 forward/backward。

---

## 三、CUDA 和 GPU 选型指南

### 3.1 名词扫盲

| 名词 | 通俗理解 |
|------|----------|
| **GPU** | 显卡里的计算芯片，适合大规模矩阵运算 |
| **显存 VRAM** | GPU 上的「内存」，OOM 就是它不够 |
| **CUDA** | NVIDIA 的并行计算平台 |
| **cuDNN** | 深度卷积等算子的加速库（常随 PyTorch 打包） |
| **驱动** | 操作系统与 GPU 对话的程序 |

### 3.2 为什么显存比「显卡型号」更关键？

```
  训练 7B / 13B 时：
    显存不够 = 直接崩（OOM）
    型号很新但显存小 = 仍要 QLoRA / 降 batch
```

### 3.3 经验向选型（非绝对，以实测为准）

| 目标 | 经验建议 |
|------|----------|
| 本地玩 **QLoRA 7B** | 常从 **≥8GB～12GB** 显存谈起（还看序列长度与实现） |
| **LoRA 13B** | 更稳妥 **24GB** 档 |
| **全参数** 大模型 | 多卡 / 企业集群，见官方显存表 |

### 3.4 没有 GPU 怎么办？

```
  ① Google Colab（按配额与政策）
  ② 云 GPU 按小时租用
  ③ 先学 L01/L02 + 读脚本，再上机
```

### 3.5 ASCII：一次训练时显存大致装什么？

```
  ┌──────────────── GPU VRAM ────────────────┐
  │ 模型参数 | 梯度 | 优化器状态 | 激活缓存  │
  └──────────────────────────────────────────┘
         ^           ^            ^           ^
         |           |            |           |
      全参最贵    反向传播要    Adam 要存动量  长序列更贵
```

---

## 四、HuggingFace 生态：Transformers / PEFT / Datasets / Accelerate / TRL

### 4.1 总览图

```
                    Hugging Face 生态
                          |
     +--------+--------+--------+--------+--------+
     |        |        |        |        |        |
 Transformers PEFT  Datasets Accelerate   TRL    Hub
  模型+分词   高效微调  数据管道  分布式训练  RLHF等  权重托管
```

### 4.2 各包一句话

| 包 | 作用 | 在 MedicalGPT 里 |
|----|------|------------------|
| **transformers** | 预训练模型、Trainer、配置 | 核心 |
| **peft** | LoRA / 适配器等 | 省显存微调 |
| **datasets** | 大规模数据加载、缓存 | 读 JSON/CSV 语料 |
| **accelerate** | 多 GPU、混合精度统一入口 | 常与 Trainer 配合 |
| **trl** | DPO、PPO 等对齐训练工具 | RLHF/DPO 脚本 |

### 4.3 类比

- **Transformers** = **整车**（发动机、变速箱都打包）  
- **PEFT** = **改装套件**（只换排气管级别的可训练模块）  
- **Datasets** = **加油站与货运系统**（持续供数据）  
- **Accelerate** = **四驱分动箱**（多卡协同）  
- **TRL** = **驾校高级课程**（对齐阶段专用教练）

### 4.4 安装示例

```bash
pip install "transformers>=4.3x" datasets accelerate peft trl
# 版本请以 MedicalGPT requirements.txt 为准
```

---

## 五、pip / conda 环境管理

### 5.1 conda 适合什么？

- 需要 **Python 版本切换**、部分 **非 Python 依赖** 时较省心。  
- 团队若统一 conda，可照团队规范。

### 5.2 venv 适合什么？

- 轻量、只管理 Python 包。  
- CI / 服务器上常见。

### 5.3 推荐习惯（减少「我电脑上能跑」）

```
  ① 一项目一环境
  ② requirements.txt 锁主要版本
  ③ 重大升级前复制环境或记录旧版本号
```

### 5.4 最小 conda 工作流

```bash
conda create -n medgpt python=3.10 -y
conda activate medgpt
pip install -r requirements.txt
```

**MedicalGPT** 官方写明 Python 3.8+；本系列常用 **3.10** 较稳（以仓库 `requirements.txt` 为准）。

### 5.5 常见坑

- **混用 pip 与 conda** 导致重复包 → 尽量「创建环境后只用一种主力」。  
- **在 base 环境乱装** → 难以复现。

---

## 六、Jupyter Notebook / Google Colab 使用

### 6.1 Jupyter 本地

```bash
pip install notebook ipykernel
jupyter notebook
```

**用途**：交互式试 `tokenizer.encode`、看 tensor 形状、画 loss 曲线。

### 6.2 Colab 典型流程

```
  打开 Notebook
      |
  运行时 -> 更改运行时类型 -> GPU（如 T4）
      |
  验证 torch.cuda.is_available()
      |
  克隆仓库 / 挂载 Drive / pip install
```

**注意**：Colab **会话会回收**，长训练要配合 checkpoint 与外部存储。

### 6.3 类比

- **脚本 `.py`** = **正式生产线**（可重复、可调度）  
- **Notebook** = **实验室台**（快速试错）

---

## 七、Git 基础操作（够用版）

### 7.1 为什么要会 Git？

```
  拉官方更新、开分支做实验、对比你改了哪、回滚误改
```

### 7.2 最小命令集

| 命令 | 作用 |
|------|------|
| `git clone <url>` | 拷贝远程仓库 |
| `git status` | 看工作区变更 |
| `git diff` | 看具体改动 |
| `git add` / `git commit` | 暂存与提交 |
| `git pull` | 拉远程更新 |
| `git checkout -b feat/x` | 新建分支 |

### 7.3 大文件注意

模型权重 **不要** 无脑提交进 Git；用 **Git LFS** 或网盘 / Hub。

---

## 八、MedicalGPT 项目克隆与依赖安装

### 8.1 克隆

```bash
git clone https://github.com/shibing624/MedicalGPT.git
cd MedicalGPT
```

### 8.2 安装依赖（官方推荐）

```bash
pip install -r requirements.txt --upgrade
```

**说明**：`requirements.txt` 会随版本更新；遇到问题先 **对照仓库版本** 与 **本课排障节**。

### 8.3 快速自检（示例）

```bash
python -c "import torch; import transformers; import peft; print('ok')"
```

### 8.4 与课程仓库的关系

你现在阅读的 `learn-MedicalGPT` 是**讲义仓库**；**MedicalGPT** 是**上游训练代码仓库**。学习时两个都要在本地或云端各就各位。

```
  learn-MedicalGPT/lessons/...  ← 教程与笔记
  MedicalGPT/                   ← 实际训练脚本
```

---

## 九、常见环境问题排查

### 9.1 `CUDA out of memory`

```
  优先序（从便宜到贵）：
    减小 batch size / 梯度累积步数调整
    缩短 max_length
    开 gradient checkpointing
    用 LoRA / QLoRA
    ZeRO / 多卡
    换更大显存
```

### 9.2 版本不兼容（Transformers / PEFT / Torch）

- 读报错栈最上面 **缺什么符号 / 哪个参数不存在**。  
- **对齐** `requirements.txt` 版本。  
- 避免同一环境 **反复覆盖安装** 不留记录。

### 9.3 HuggingFace 下载慢或失败

按合规与网络环境选择：

- 镜像（各地政策不同，自行查证最新方案）  
- `huggingface-cli download` 预下载后 **本地路径** 加载

### 9.4 `libcudnn` / 驱动相关

- 升级 / 重装 **与 CUDA 匹配的驱动**  
- 重装 **对应 CUDA 的 PyTorch wheel**

### 9.5 Windows vs Linux

```
  训练大模型：更推荐 Linux 或 WSL2
  原生 Windows：部分库支持弱、路径坑多
```

### 9.6 阅读超长 traceback

```
  从下往上：
    1) CUDA error / killed / segfault
    2) Python 哪一行
    3) ImportError 哪个包版本
```

---

## 十、硬件需求表（直接引用 MedicalGPT 官方 README）

以下表格**直接摘自** [MedicalGPT 仓库 README](https://github.com/shibing624/MedicalGPT) 的 **Hardware Requirement (显存/VRAM)** 小节（标注为**估算值**，实际随实现、序列长、batch、框架版本波动）：

| 训练方法 | 精度 | 7B | 13B | 30B | 70B | 110B | 8x7B | 8x22B |
|---------|------|-----|-----|-----|-----|------|------|-------|
| 全参数 | AMP(自动混合精度) | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| 全参数 | 16 | 60GB | 120GB | 300GB | 600GB | 900GB | 400GB | 1200GB |
| LoRA | 16 | 16GB | 32GB | 64GB | 160GB | 240GB | 120GB | 320GB |
| QLoRA | 8 | 10GB | 20GB | 40GB | 80GB | 140GB | 60GB | 160GB |
| QLoRA | 4 | 6GB | 12GB | 24GB | 48GB | 72GB | 30GB | 96GB |
| QLoRA | 2 | 4GB | 8GB | 16GB | 24GB | 48GB | 18GB | 48GB |

### 10.1 怎么读这张表？

```
  全参数：「土豪/集群」路线，个人工作站很难扛大模型
  LoRA：  只训适配器，显存显著下降，入门微调主路径
  QLoRA： 量化 + LoRA，单卡友好，但要注意稳定性与超参
```

### 10.2 和「推理」区别

本表偏 **训练** 估算；**推理** 显存与 **KV Cache、并发、量化** 强相关，不能简单等同。

---

## 十一、面试可能问到的环境相关问题

1. **你如何验证 GPU 训练环境 OK？**  
   `torch.cuda.is_available()`、CUDA 张量、极小 batch 的 forward/backward。

2. **OOM 你怎么排查？**  
   batch、seq length、精度、checkpoint、ZeRO、是否误全参、梯度累积等。

3. **conda 和 venv 区别？**  
   conda 管得更宽；venv 轻；团队规范优先。

4. **依赖如何可复现？**  
   锁版本、`requirements.txt`、Docker、记录驱动与 CUDA。

5. **为什么 HuggingFace Transformers 流行？**  
   统一 API、模型卡生态、与 PEFT/Accelerate 集成。

6. **PEFT 解决什么痛点？**  
   消费级 GPU 上微调大模型的可行性。

7. **CUDA 版本与 PyTorch 不一致会怎样？**  
   可能导入成功但运行报错或极慢；以 **官方组合矩阵**为准。

8. **混合精度（fp16/bf16）好处？**  
   省显存、提速；注意数值稳定与 loss scaling（视实现）。

---

## 十二、推荐阅读

| 资源 | 链接 |
|------|------|
| PyTorch 安装 | [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) |
| HuggingFace 文档 | [huggingface.co/docs](https://huggingface.co/docs) |
| MedicalGPT README | [github.com/shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT) |
| MedicalGPT Wiki | [Wiki](https://github.com/shibing624/MedicalGPT/wiki) |

---

## 附录 A：一条「从 0 到 import ok」命令流（模板）

```bash
conda create -n medgpt python=3.10 -y
conda activate medgpt
# 下面一行请换成你机器匹配的 CUDA 版 PyTorch（见官网）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate peft trl
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 附录 B：磁盘与网络清单

- 7B 权重常见 **十几 GB** 量级（视格式与精度）。  
- 13B、70B 成倍上升。  
- 预留：**模型 + 数据集 + checkpoint + 日志**。

---

## 附录 C：WSL2 提示（Windows 用户）

```
  宿主机安装/更新 NVIDIA 驱动（支持 WSL）
           |
           v
  WSL2 内安装与驱动匹配的 PyTorch CUDA 版
           |
           v
  nvidia-smi 在 WSL 内可见
```

---

## 附录 D：环境检查清单（打印勾选用）

```
[ ] python --version 符合项目要求
[ ] pip 指向当前环境（which pip / pip -V）
[ ] torch 可 import
[ ] GPU 机器上 cuda True
[ ] transformers / peft 版本与 requirements 一致
[ ] 能打开 Jupyter 或能跑通一个最小 generate
[ ] git clone MedicalGPT 成功
```

---

## 附录 E：显存估算心智公式（面试吹牛用，勿当精确）

```
  粗浅直觉：
    参数相关占用 ∝ 模型规模 × 精度 ×（是否训练：梯度+优化器）
    激活相关占用 ∝ batch × 序列长 × 层数 × 隐藏维（常很敏感）
```

---

## 附录 F：与下一课衔接

L04 会带你看 **MedicalGPT 仓库地图**：哪些脚本对应 PT/SFT/DPO——**本课把环境铺好**，下一课就能「指哪打哪」。

---

[← 上一课](../L02-Transformer架构核心原理/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L04-MedicalGPT项目全景/README.md)
