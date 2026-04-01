[← 上一课](../L13-数据处理与质量工程/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L15-模型评估与推理部署/README.md)

# L14 分布式训练与 DeepSpeed

> **一句话精髓**：「一张卡不够？那就八张一起上」——大模型训练本质是**内存墙 + 算力墙**的工程问题；分布式与 ZeRO 等技术的目标，是在可接受的成本下把墙「拆掉」或「绕开」。

---

## 0. 本课你将带走什么

1. 能解释 **DP / MP / PP** 各自切什么、适合什么模型规模。  
2. 能对比 **ZeRO-1 / ZeRO-2 / ZeRO-3** 分区了什么张量，以及通信与显存 trade-off。  
3. 能读懂一份 **DeepSpeed config JSON** 的关键字段，并关联到 MedicalGPT 训练脚本。  
4. 会使用 **`torchrun`** 启动多进程训练，理解 **`local_rank` / `world_size`**。  
5. 掌握 **梯度累积、混合精度、Flash Attention** 的直觉与典型配置。  
6. 能背一张 **7B–110B 量级显存需求表（经验值）** 应对面试数量级问题。

---

## 1. 为什么需要分布式训练

### 1.1 显存三板斧：参数、优化器、激活

训练时显存大致要装：

```
┌─────────────────────────────────────────────┐
│  模型参数（weights）                         │
│  + 梯度（gradients）                         │
│  + 优化器状态（如 Adam 的 m/v）               │
│  + 激活（activations，随 batch/长度变化）     │
└─────────────────────────────────────────────┘
```

**类比**：你要同时放下「课本（参数）」「错题本（梯度）」「老师的两份笔记（优化器）」以及「课堂草稿（激活）」。桌子（显存）不够就只能：

- 换大桌子（更大 GPU）  
- 把书分开放（分布式）  
- 用更小的字（量化 / checkpointing）  
- 少同时翻太多页（减小 batch / 梯度检查点）

### 1.2 算力与时间的线性幻想

理想中：8 卡 ≈ 单卡 1/8 时间。现实中：

- 通信开销  
- 负载不均衡  
- IO 瓶颈  
- 小 batch 下 GPU 吃不饱

所以分布式是**必要但不魔法**。

---

## 2. 数据并行（Data Parallel, DP）

### 2.1 核心思想

**每张卡一份完整模型**，各自吃不同 mini-batch，**梯度做平均**再更新。

```
         batch 切分
    ┌────┬────┬────┬────┐
    │ G0 │ G1 │ G2 │ G3 │
    └────┴────┴────┴────┘
        \   |   |   /
         \  |   |  /
          AllReduce(梯度)
```

### 2.2 优缺点

| 优点 | 缺点 |
|------|------|
| 实现相对简单 | 每卡都存全量参数+优化器 → 大模型卡死 |
| 吞吐随卡数提升（理想情况） | batch 很大时激活显存仍爆 |

**面试一句话**：DP 解决「算得慢」，**不一定**解决「装不下」。

---

## 3. 模型并行（Model Parallel, MP / Tensor Parallel）

### 3.1 核心思想

**把一层里的矩阵乘切开**到多张卡，每张只算一部分。

**类比**：四个人接力算一个大乘法：每人只算几位数，最后再拼起来（概念上）。

### 3.2 何时考虑

- 单层都塞不进单卡  
- 超大宽度（hidden size）模型

### 3.3 代价

- **通信频繁**（每层可能都要同步）  
- 需要好的 NVLink / 机内带宽

---

## 4. 流水线并行（Pipeline Parallel, PP）

### 4.1 核心思想

按**层**切：GPU0 管前若干层，GPU1 管中间，GPU2 管后面。

```
  micro-batch 像工厂流水线

  时间 →
  GPU0: [fwd0][fwd1][fwd2]...
  GPU1:      [fwd0][fwd1]...
  GPU2:           [fwd0]...
```

### 4.2 气泡（bubble）

流水线会有**空闲等待**（bubble），需要用 **micro-batch** 填满流水线提高利用率。

**面试考点**：PP 减少单卡参数，但引入调度复杂度与气泡损失。

---

## 5. ZeRO 优化：ZeRO-1 / ZeRO-2 / ZeRO-3

ZeRO（Zero Redundancy Optimizer）的核心：**去掉多卡之间的冗余存储**。

### 5.1 类比：四人合租

- **不分 ZeRO**：每人家里都买一套完整百科全书（参数、梯度、优化器各一份）。  
- **ZeRO**：书拆册，每人只保管几册，要看时去借（通信）。

### 5.2 三者分区对象（必背）

| 阶段 | 分区内容 | 直觉 |
|------|----------|------|
| ZeRO-1 | **优化器状态**分片 | 最常见起步 |
| ZeRO-2 | ZeRO-1 + **梯度**分片 | 显存进一步降 |
| ZeRO-3 | ZeRO-2 + **参数**分片 | 最省显存，通信最重 |

### 5.3 ZeRO-2 vs ZeRO-3（面试高频）

```
ZeRO-2:
  - 前向时参数仍在各卡「凑齐」或按实现广播/收集（实现细节因框架而异）
  - 梯度分片 → AllReduce 变分片 reduce-scatter 等模式

ZeRO-3:
  - 参数也分片 → 前向/反向常需要按层「按需取回」参数
  - 显存最省，但通信路径更长，对网络要求更高
```

**标准答法**：

> ZeRO-2 主要分片**优化器状态与梯度**，参数通常仍可按 DP 思路使用；ZeRO-3 进一步分片**模型参数**，把单卡参数显存压到约 `1/N`，但引入更频繁的参数聚合与通信，需要更好的集群带宽与更细致的性能调优。

---

## 6. DeepSpeed 配置详解（JSON 字段怎么读）

下面是一份**教学合成**的 `ds_config.json` 片段（具体项目以 MedicalGPT 仓库为准）：

```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

### 6.1 字段解释（面试逐个点名）

- **`train_micro_batch_size_per_gpu`**：单卡单次前向的 batch。  
- **`gradient_accumulation_steps`**：累积多少步再同步/更新（后面专节）。  
- **`train_batch_size`**：DeepSpeed 里通常要求满足大致关系：  
  `train_batch_size ≈ micro_batch * world_size * grad_accum`（具体以官方校验为准）。  
- **`zero_optimization.stage`**：1/2/3。  
- **`bf16` / `fp16`**：混合精度开关。  
- **`gradient_clipping`**：防爆梯度，稳定大模型训练。  
- **`overlap_comm`**：通信与计算重叠，提速。

### 6.2 与 Hugging Face Trainer 的关系

常见模式：`TrainingArguments` + `deepspeed=ds_config.json`。Trainer 把优化器交给 DeepSpeed 托管。

---

## 7. MedicalGPT 中的分布式训练配置（怎么在仓库里找）

**学习方法（比背文件更重要）**：

1. 打开 MedicalGPT 仓库根目录，搜索 `deepspeed`、`ds_config`、`torchrun`。  
2. 对照 `run_*.sh` 或 `*.sh` 启动脚本里的参数：`--deepspeed`、`--nproc_per_node`。  
3. 看 `train_*.py` 是否使用 `Trainer` 或自定义循环。

**面试表述**：

> 我们使用 **HuggingFace Trainer + DeepSpeed ZeRO-2/3**，通过 `torchrun` 启动多进程；配置文件里显式管理 **micro batch、梯度累积、bf16、ZeRO stage**，并与 **flash_attn**（若环境支持）联动降低显存。

---

## 8. torchrun 启动命令

### 8.1 典型单机多卡

```bash
torchrun --nproc_per_node=8 --master_port=29500 train_sft.py \
  --deepspeed ds_config_zero2.json \
  --output_dir ./out \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8
```

### 8.2 环境变量（常考）

| 变量 | 含义 |
|------|------|
| `LOCAL_RANK` | 本机第几张卡 |
| `RANK` | 全局进程号 |
| `WORLD_SIZE` | 总进程数 |
| `MASTER_ADDR` / `MASTER_PORT` | 多机通信 |

### 8.3 类比

`torchrun` 像「班主任」：给每个进程发学号（rank），指定班长地址（master），再一起跑操（同步初始化）。

---

## 9. 梯度累积（Gradient Accumulation）

### 9.1 解决什么问题

**显存不够上大 batch** → 用小 batch 多跑几步，**梯度相加**后再 `optimizer.step()`。

```
真实 batch = micro_batch * num_gpus * grad_accum_steps
```

### 9.2 迷你数值示例

- `micro_batch_per_gpu = 2`  
- `gpus = 8`  
- `grad_accum = 4`  

一步有效样本 ≈ `2 * 8 * 4 = 64`（演示用；注意 DP 实现细节是否与该直觉完全一致，以框架为准）。

### 9.3 面试题：梯度累积的作用？

**标准答法**：

> 在显存受限时，用多次前向反向累积梯度，以**逼近更大 batch 的优化效果**；它不减少单次激活峰值显存，但能提高**有效 batch size**，改善训练稳定性与指标；代价是**更新频次降低、墙钟时间变长**，且学习率调度有时需要按有效 batch 重新标定。

---

## 10. 混合精度训练（FP16 / BF16 / AMP）

### 10.1 为什么有用

- **Tensor Core** 对半精度矩阵乘更快。  
- **激活与部分状态**占用更少显存。

### 10.2 FP16 vs BF16

| 特性 | FP16 | BF16 |
|------|------|------|
| 动态范围 | 较小，易溢出 | 较大，训练更稳（很多大模型默认） |
| 硬件 | 老卡也常见 | Ampere+ 友好 |

### 10.3 AMP 直觉

**自动混合精度**：前向某些 op 用 fp16，敏感 op 用 fp32；**GradScaler**防止下溢（fp16 场景）。

**面试一句话**：混合精度是**算得快 + 省显存**，但要关注 **loss scaling 与数值稳定**。

---

## 11. Flash Attention 加速

### 11.1 解决什么问题

标准注意力显存随序列长度 **O(L²)** 存中间结果；FlashAttention 通过 **IO 感知分块**降低 HBM 读写。

**类比**：不把整张 `L×L` 表一次性摊在桌上，而是**分块翻页**算。

### 11.2 你能怎么说（面试）

> 在长上下文 SFT 上，开启 **Flash Attention 2** 通常能显著降低注意力显存与提升吞吐，但需要 **GPU 架构与 PyTorch/CUDA 版本**匹配；我们一般在 A100/H100 类环境启用，并在训练脚本里通过 `attn_implementation="flash_attention_2"` 或等价方式指定。

---

## 12. 显存优化技巧清单（背这张表就够用）

```
1) ZeRO-2/3
2) 梯度检查点 gradient checkpointing（换算力省激活）
3) 更小 micro batch + 梯度累积
4) 序列截断 / packing（谨慎医疗长文本）
5) Flash Attention
6) BF16 优先
7) 卸载 CPU offload（DeepSpeed offload，慢但救命）
8) 冻结部分层（只训 LoRA 时天然省）
```

---

## 13. 硬件需求表（7B 到 110B，经验量级）

> **强声明**：下表为**社区常见经验区间**，与并行策略、上下文长度、是否全参、是否开启 ZeRO-3/Offload、框架实现强相关。面试说「数量级」即可，不要当合同指标。

| 规模（Dense 全参微调直觉） | 训练（每张卡大致量级，ZeRO-2/3 混合口径） | 推理（FP16 权重一张卡「装得下吗」直觉） |
|----------------------------|------------------------------------------|----------------------------------------|
| **7B** | 24GB 级可尝试 LoRA/QLoRA；全参常需多卡或更大显存 | 约 **14GB+** 量级（视实现与 KV cache） |
| **13B** | 40GB 级更常见起步 | 约 **26GB+** |
| **34B** | 多卡 A100 40G/80G；ZeRO-3 | 单卡 80G 或量化 |
| **70B** | 多卡 80G × N；常 ZeRO-3 + PP/TP | 多卡或 INT4 级量化 |
| **110B+** | 大集群 + 复杂并行 | 几乎必然量化 + 多卡 |

**记忆口诀**：参数量（B）× 2 bytes（fp16）≈ **权重大致 GB 量级**（7B → ~14GB 权重，不含优化器与激活）。

---

## 14. 综合 ASCII：单机 8 卡 + ZeRO-2 信息流

```
  dataloader (不同 shard)
        │
   ┌────┴────┬────┬────┬────┐
   v         v    v    v    v
  GPU0      GPU1 ...        GPU7
   │  micro forward/backward
   v
 梯度分片存储 (ZeRO-2)
   │
   v
 reduce-scatter / allgather 组合通信
   │
   v
 优化器 step（分片状态）
```

---

## 15. 面试高频题速答

### Q1：ZeRO-2 和 ZeRO-3 的区别？

见 **§5.3**；补一句：**ZeRO-3 更吃网络带宽与实现调优**。

### Q2：梯度累积的作用？

见 **§9.3**；补一句：**不降低单次前向峰值激活**。

### Q3：数据并行和模型并行怎么选？

> 小模型优先 DP；单卡装不下单层用 **TP**；极深模型可 **PP**；超大集群常 **DP + TP + PP + ZeRO** 组合。

### Q4：bf16 还需要 loss scaler 吗？

> 多数场景 **不像 fp16 那样依赖 GradScaler**（实现仍取决于框架版本与模型）；fp16 常需要 scaler 防梯度下溢。

---

## 16. 排错思路（工程向）

```
loss NaN:
  - 检查学习率、bf16/fp16、数据异常标签
  - 梯度裁剪是否开启

速度很慢:
  - 是否 dataloader 瓶颈（num_workers、pin_memory）
  - ZeRO-3 通信是否过载
  - 是否小 batch 导致 GPU 利用率低

显存仍爆:
  - checkpointing
  - 降 seq length
  - ZeRO-3 / offload
  - LoRA
```

---

## 17. 本课自测

1. 画 DP 与 TP 的切分差异。  
2. 写出有效 batch 与 micro batch、卡数、累积步的关系（口语）。  
3. 为什么 ZeRO-3 更省显存但更依赖带宽？  
4. FlashAttention 主要优化的是算力还是显存带宽？  
5. 7B fp16 权重大约多少 GB（口算）？

---

## 18. 延伸阅读（可选）

- DeepSpeed ZeRO 论文与官方文档。  
- PyTorch FSDP vs DeepSpeed 对比（面试加分）。  
- NVIDIA Mixed Precision Training 指南。

---

## 19. 附录：多机训练最小概念

```
机器 A (rank 0..7)  ----以太网/IB----  机器 B (rank 8..15)
                \                    /
                 ---- NCCL 通信 ----
```

**面试点**：多机要关注 **`MASTER_ADDR`**、网络拓扑、**NCCL_IB_DISABLE** 等环境变量与防火墙。

---

## 20. 与 MedicalGPT 实战的衔接建议

1. 先用 **单卡小模型/小数据**跑通训练循环。  
2. 再切换到 **2 卡 ZeRO-1**确认通信正常。  
3. 再上 **ZeRO-2 + bf16**。  
4. 只有当真装不下时上 **ZeRO-3**。  

**渐进式心法**：每次只改一个变量，**把性能曲线与显存曲线记进实验表**。

---

## 21. ZeRO-Offload：把「桌子」延伸到「地板」（CPU / NVMe）

### 21.1 直觉

显存不够时，把**优化器状态**甚至**参数**临时「搬到」CPU 内存或 NVMe。代价是 **PCIe 带宽**与 **墙钟时间**。

```
  GPU HBM（快而小）     <----PCIe---->     CPU DRAM（慢而大）
         │                                      │
    计算热点                                 冷状态停放
```

### 21.2 面试怎么说

> Offload 是**救命配置**不是**性能配置**；适合资源受限 PoC，生产训练通常优先 **更大显存 / ZeRO-3 / 序列并行** 等路径，并对 offload 做 **吞吐 profiling**。

---

## 22. 激活检查点（Gradient Checkpointing）：用算力换显存

### 22.1 原理一句话

前向时**不存全部中间激活**，反向时**重算**部分前向 → 显存降、时间升。

### 22.2 医疗长文本场景

患者病史 + 检验单拼接可能很长；**checkpointing** 常和 **FlashAttention**、**更小 micro batch** 一起出现。

**记忆口诀**：**「省显存三件套」** = ZeRO + checkpointing + flash attn（环境允许时）。

---

## 23. FSDP vs DeepSpeed ZeRO（面试对比加分）

| 维度 | PyTorch FSDP | DeepSpeed ZeRO |
|------|--------------|----------------|
| 生态 | 原生 PyTorch 2.x 友好 | 与 HF Trainer 集成成熟 |
| 配置 | Python 包装多 | JSON `ds_config` 显式 |
| 高级特性 | 持续演进 | ZeRO-Infinity、offload 等历史积累多 |
| 选型口语 | 想少依赖、偏原生 | 要大模型训练「全家桶」与脚本范例 |

**标准答法**：

> 二者思想接近，都是 **分片参数/梯度/优化器状态** 降低冗余；我们选型看 **团队熟悉度、集群网络、与现有 Trainer 集成成本**。

---

## 24. ZeRO-3 配置片段（教学向合成）

```json
{
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "sub_group_size": 1e9
  }
}
```

**读字段直觉**：

- **`stage3_*_bucket_size`**：通信打包粒度，影响带宽利用与峰值显存。  
- **`persistence_threshold`**：小参数可常驻 GPU，减少频繁搬动（实现相关，面试提「调参 trade-off」即可）。

---

## 25. 有效吞吐量估算（把「感觉慢」变成数字）

### 25.1 公式骨架（口语版）

```
samples/sec ≈ (micro_batch * world_size) / step_time * (1 / grad_accum?) 
```

注意：`grad_accum` 让 **optimizer step** 变稀疏，**有效 batch** 变大但 **每秒更新次数**下降——汇报时要说明你看的是 **tokens/s** 还是 **updates/s**。

### 25.2 虚构 profiling 表示例

| 配置 | tokens/s（8×A100） | 备注 |
|------|---------------------|------|
| ZeRO-2 + bf16 | 18.2k | baseline |
| ZeRO-3 同设置 | 14.1k | 显存↓，通信↑ |
| + offload optimizer | 6.8k | 救命但慢 |

---

## 26. NCCL 与环境变量排雷（多卡训练「玄学」集中区）

```
常见症状                可能方向
---------               --------
hang 在 init            MASTER_ADDR/PORT、防火墙
周期性 timeout          IB/RoCE 配置、NCCL_P2P_DISABLE
速度异常慢              GPU 拓扑、是否跨 NUMA 乱绑核
loss 抖动大             学习率与有效 batch 不匹配（不全是 NCCL）
```

**面试点**：你会如何定位是 **数据加载** 还是 **通信** 瓶颈？

> 看 **GPU-Util** 是否长期低、**nvidia-smi dmon**、**Nsight Systems**；同时在 dataloader 侧试 `num_workers`、`persistent_workers`、`prefetch_factor`。

---

## 27. 序列并行与上下文扩展（进阶指路）

当 **单条序列长度** 成为瓶颈，除了 **FlashAttention**，还可了解：

- **Context Parallel / Ring Attention** 等把长序列切到多卡。  
- 医疗场景要先确认：**病史拼接策略**是否真需要一次喂 32k，还是 **RAG + 分段摘要**更稳。

---

## 28. 本课追加自测

1. Offload 主要救的是参数显存还是优化器显存（口语）？  
2. gradient checkpointing 对 **激活峰值** 与 **训练时间** 各有什么影响？  
3. 为什么 ZeRO-3 往往需要更认真地调 bucket size？  
4. 汇报吞吐时为什么要同时报 **tokens/s** 与 **有效 batch**？

---

**结语**：分布式不是「炫技」，是**把不可训练变成可训练**的桥梁。下一课我们讨论**评估与部署**——训练再分布式，最终价值仍要在线上指标与用户体验里兑现。

---

[← 上一课](../L13-数据处理与质量工程/README.md) | [📚 课程目录](../../README.md) | [下一课 →](../L15-模型评估与推理部署/README.md)
