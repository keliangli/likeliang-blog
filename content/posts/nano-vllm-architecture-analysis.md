---
title: "Nano-vLLM 架构深度解析：轻量级 vLLM 实现的技术剖析"
date: 2026-02-26T14:00:00+08:00
draft: false
tags: ["LLM", "vLLM", "推理优化", "架构设计", "Python", "PyTorch", "AI系统"]
categories: ["架构分析"]
author: "NEXUS"
description: "深入剖析 nano-vLLM 项目架构，仅用1200行Python代码实现与官方vLLM相媲美的推理性能"
---

> 一个从零构建的轻量级 vLLM 实现，仅用约 **1,200 行 Python 代码** 实现了与官方 vLLM 相媲美的推理性能。

---

## 一、项目概述

### 1.1 项目定位

[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 是一个教育导向的轻量级 vLLM 实现项目，由 GeeeekExplorer 开发。项目的核心目标是：

- **简洁性**：用约 1,200 行 Python 代码实现 vLLM 的核心功能
- **可读性**：代码结构清晰，便于学习和理解 vLLM 的工作原理
- **高性能**：在保持代码简洁的同时，实现与官方 vLLM 相当的推理速度

### 1.2 性能表现

根据官方基准测试结果：

| 推理引擎 | 输出 Token 数 | 时间 (s) | 吞吐量 (tokens/s) |
|:---:|:---:|:---:|:---:|
| vLLM | 133,966 | 98.37 | 1361.84 |
| **Nano-vLLM** | 133,966 | **93.41** | **1434.13** |

**测试配置**：
- 硬件：RTX 4070 Laptop (8GB)
- 模型：Qwen3-0.6B
- 请求数：256 序列
- 输入长度：100-1024 tokens（随机采样）
- 输出长度：100-1024 tokens（随机采样）

---

## 二、项目架构与技术栈

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Nano-vLLM 架构                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   LLM API   │───▶│  Scheduler  │───▶│   Model Execution   │ │
│  │   (llm.py)  │    │(scheduler.py)│   │    (model.py)       │ │
│  └─────────────┘    └──────┬──────┘    └─────────────────────┘ │
│         │                  │                    │               │
│         ▼                  ▼                    ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Sampler   │    │ Prefix Cache│    │  Paged Attention    │ │
│  │(sampler.py) │    │ (cache.py)  │    │  (attention.py)     │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 技术栈分析

| 组件 | 技术选择 | 说明 |
|:---|:---|:---|
| **深度学习框架** | PyTorch | 核心计算框架 |
| **注意力机制** | FlashAttention / xFormers | 高效注意力计算 |
| **模型格式** | HuggingFace Transformers | 模型加载和配置 |
| **量化支持** | GGUF / AWQ | 模型压缩和加速 |
| **并行计算** | PyTorch DDP / Tensor Parallel | 多 GPU 支持 |
| **编译优化** | torch.compile | 图编译优化 |
| **CUDA 优化** | CUDA Graphs | 减少 CPU 开销 |

### 2.3 项目结构

```
nano-vllm/
├── nanovllm/
│   ├── __init__.py      # 包入口，导出主要类
│   ├── llm.py           # LLM 主类，对外 API
│   ├── model.py         # 模型定义和推理逻辑
│   ├── attention.py     # PagedAttention 实现
│   ├── cache.py         # KV Cache 和 Prefix Caching
│   ├── scheduler.py     # 请求调度器
│   ├── sampler.py       # 采样策略实现
│   ├── parallel.py      # 张量并行支持
│   ├── loader.py        # 模型加载器
│   └── quant.py         # 量化支持
├── example.py           # 使用示例
├── bench.py             # 基准测试
└── pyproject.toml       # 项目配置
```

---

## 三、核心模块深度分析

### 3.1 LLM 主类 (`llm.py`)

LLM 类是用户交互的主要接口，负责协调各个组件：

```python
class LLM:
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
    ):
        # 初始化配置
        self.config = AutoConfig.from_pretrained(model_path)
        
        # 初始化并行环境
        if tensor_parallel_size > 1:
            init_parallel(tensor_parallel_size)
            
        # 加载模型权重
        self.model = Model(...)
        load_weights(self.model, model_path, dtype)
        
        # 初始化调度器和缓存管理器
        self.scheduler = Scheduler(...)
        
        # 编译优化
        if not enforce_eager:
            self.model = torch.compile(self.model)
```

**关键设计决策**：
1. **延迟初始化**：模型权重在初始化后才加载
2. **可选编译**：通过 `enforce_eager` 控制是否使用 `torch.compile`
3. **模块化设计**：Scheduler、Model、Cache 等组件解耦

### 3.2 PagedAttention 实现 (`attention.py`)

PagedAttention 是 vLLM 的核心创新，通过将 KV Cache 分页管理来解决内存碎片化和共享问题：

**核心思想对比**：

```
传统 KV Cache 存储：
┌─────────────────────────────────────────────────────┐
│  Seq 1: [████████████████████]  (连续内存)          │
│  Seq 2: [████████████]                              │
│  Seq 3: [████████████████████████████]              │
│  问题：内存碎片化，无法有效共享前缀                    │
└─────────────────────────────────────────────────────┘

PagedAttention 分页存储：
┌─────────────────────────────────────────────────────┐
│  Block 0: [████]  ← Seq 1, Seq 2, Seq 3 共享前缀    │
│  Block 1: [████]  ← Seq 1 独有                      │
│  Block 2: [████]  ← Seq 2 独有                      │
│  Block 3: [████]  ← Seq 3 独有                      │
│  优势：内存紧凑，前缀共享，动态分配                    │
└─────────────────────────────────────────────────────┘
```

### 3.3 KV Cache 与 Prefix Caching (`cache.py`)

KV Cache 管理是推理性能的关键：

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, 
                 enable_prefix_caching: bool = True):
        self.free_blocks = list(range(num_blocks))
        self.block_tables: Dict[int, List[int]] = {}
        self.prefix_cache: Dict[str, int] = {}
        
    def prefix_match(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """查找前缀缓存匹配"""
        prefix_hash = self._compute_hash(token_ids[:self.block_size])
        if prefix_hash in self.prefix_cache:
            return self.block_size, [self.prefix_cache[prefix_hash]]
        return 0, []
```

**Prefix Caching 示例**：

```
请求 1: "The quick brown fox jumps over the lazy dog"
         └─ Block 0 ─┘└─ Block 1 ─┘└─ Block 2 ─┘
         哈希: 0x1234   哈希: 0x5678   哈希: 0x9ABC
         存入 prefix_cache

请求 2: "The quick brown fox runs in the park"
         └─ Block 0 ─┘└─ Block 1 ─┘
         哈希: 0x1234 (命中!)  哈希: 0x5678 (命中!)
         结果：重用 Block 0 和 Block 1，只需计算新部分
```

### 3.4 调度器 (`scheduler.py`)

调度器负责管理请求队列、批处理和内存分配：

```python
class Scheduler:
    def __init__(self, config: PretrainedConfig, ...):
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
        self.swapped: deque[Sequence] = deque()  # 交换队列
        self.block_manager = BlockManager(...)
        
    def schedule(self) -> SchedulerOutput:
        # 1. 尝试将等待队列中的请求加入运行队列
        self._schedule_waiting()
        
        # 2. 处理运行队列中的请求
        seqs = list(self.running)
        
        # 3. 构建批次
        for seq in seqs:
            prefix_len, blocks = self.block_manager.prefix_match(seq.tokens)
            # ...
```

**调度策略**：
1. **FCFS (First-Come-First-Served)**：基本调度策略
2. **抢占 (Preemption)**：当内存不足时，暂停低优先级请求
3. **连续批处理 (Continuous Batching)**：动态合并解码阶段的新请求

---

## 四、关键算法实现

### 4.1 PagedAttention 算法详解

PagedAttention 的核心是将 KV Cache 组织成固定大小的块（blocks）：

```python
def paged_attention(
    query: torch.Tensor,      # [batch_size, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, ...]
    value_cache: torch.Tensor,# [num_blocks, block_size, ...]
    block_tables: torch.Tensor, # [batch_size, max_num_blocks]
    context_lens: torch.Tensor, # [batch_size]
) -> torch.Tensor:
    for i in range(batch_size):
        q = query[i]
        blocks = block_tables[i]
        
        # 从块中收集 key 和 value
        keys, values = [], []
        for block_id in blocks:
            keys.append(key_cache[block_id])
            values.append(value_cache[block_id])
            
        k = torch.cat(keys, dim=0)
        v = torch.cat(values, dim=0)
        
        # 标准注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_size)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        output.append(torch.matmul(attn_scores, v))
```

### 4.2 连续批处理 (Continuous Batching)

连续批处理允许在解码阶段动态加入新请求：

```python
def continuous_batching_step(self):
    # 1. 完成当前批次的解码
    outputs = self.model.forward(current_batch)
    next_tokens = self.sampler.sample(outputs)
    
    # 2. 更新序列状态
    for seq, token in zip(self.running, next_tokens):
        seq.append_token(token)
        if seq.is_finished():
            self.running.remove(seq)
            
    # 3. 尝试加入等待队列中的请求
    while self.waiting and len(self.running) < self.max_batch_size:
        new_seq = self.waiting.popleft()
        if self.can_allocate(new_seq):
            self.running.append(new_seq)
```

---

## 五、性能优化策略

### 5.1 内存优化

| 优化技术 | 实现方式 | 效果 |
|:---|:---|:---|
| **Paged KV Cache** | 固定大小的块分配 | 消除内存碎片，支持前缀共享 |
| **Prefix Caching** | Radix Tree 缓存 | 减少重复计算，提升命中率 |
| **量化 (Quantization)** | INT8/INT4/FP8 | 减少显存占用，提升吞吐 |

### 5.2 计算优化

| 优化技术 | 实现方式 | 效果 |
|:---|:---|:---|
| **FlashAttention** | 融合注意力内核 | 减少 HBM 访问，加速计算 |
| **torch.compile** | 图编译优化 | 融合算子，减少开销 |
| **CUDA Graphs** | 捕获和重放执行图 | 减少 CPU 启动开销 |
| **张量并行** | 多 GPU 并行 | 扩展模型规模 |

---

## 六、与官方 vLLM 的对比分析

### 6.1 功能对比

| 特性 | Nano-vLLM | 官方 vLLM |
|:---|:---:|:---:|
| **PagedAttention** | 支持 | 支持 |
| **Prefix Caching** | 支持 | 支持 |
| **连续批处理** | 支持 | 支持 |
| **张量并行** | 支持 | 支持 |
| **流水线并行** | 不支持 | 支持 |
| **投机解码** | 不支持 | 支持 |
| **OpenAI API** | 不支持 | 支持 |

### 6.2 代码复杂度对比

| 指标 | Nano-vLLM | 官方 vLLM |
|:---|:---:|:---:|
| **代码行数** | ~1,200 行 | ~100,000+ 行 |
| **核心模块** | 8 个文件 | 100+ 个文件 |
| **依赖数量** | 精简 | 较多 |
| **学习曲线** | 平缓 | 陡峭 |

### 6.3 性能对比

```
测试环境：RTX 4070 Laptop (8GB)，Qwen3-0.6B，256 序列

┌─────────────┬──────────────┬─────────────┬──────────────────┐
│   引擎       │ 输出 Token   │ 时间 (s)    │ 吞吐量 (tok/s)   │
├─────────────┼──────────────┼─────────────┼──────────────────┤
│ vLLM        │   133,966    │   98.37     │    1361.84       │
│ Nano-vLLM   │   133,966    │   93.41     │    1434.13       │
└─────────────┴──────────────┴─────────────┴──────────────────┘

性能差异：Nano-vLLM 快约 5.3%
```

### 6.4 适用场景

| 场景 | Nano-vLLM | 官方 vLLM |
|:---|:---:|:---:|
| **学习研究** | 推荐 | 可选 |
| **原型开发** | 推荐 | 可选 |
| **生产部署** | 谨慎 | 推荐 |
| **大模型 (>70B)** | 不推荐 | 推荐 |
| **高并发服务** | 不推荐 | 推荐 |

---

## 七、代码质量与工程实践

### 7.1 代码质量评估

**优点**：
1. **简洁清晰**：代码结构直观，易于理解
2. **类型注解**：充分使用 Python 类型提示
3. **文档完善**：关键函数有 docstring
4. **模块化**：组件职责明确，解耦良好

### 7.2 工程实践

项目使用现代 Python 打包标准（PEP 621）：

```toml
[project]
name = "nanovllm"
version = "0.1.0"
description = "A lightweight vLLM implementation"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "flash-attn>=2.0.0",
]
```

---

## 八、潜在改进点

### 8.1 功能扩展

1. **投机解码 (Speculative Decoding)**
   ```python
   class SpeculativeDecoder:
       def __init__(self, draft_model, target_model):
           self.draft_model = draft_model
           self.target_model = target_model
           
       def generate(self, prompt, num_draft_tokens=5):
           draft_tokens = self.draft_model.generate(prompt, num_draft_tokens)
           logits = self.target_model.forward(prompt + draft_tokens)
           accepted = self._verify_tokens(logits, draft_tokens)
           return accepted
   ```

2. **OpenAI 兼容 API**
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.post("/v1/chat/completions")
   async def chat_completions(request: ChatRequest):
       outputs = llm.generate(prompts=[request.messages], ...)
       return format_openai_response(outputs)
   ```

### 8.2 性能优化

1. **更高效的调度算法** - 实现 SRPT (Shortest Remaining Processing Time)
2. **更优的内存管理** - 引入 LRU 缓存淘汰策略
3. **编译优化** - 使用 Triton 自定义 CUDA 内核

---

## 九、学习价值与最佳实践

### 9.1 适合学习的重点

1. **PagedAttention 原理与实现** - 理解内存分页管理的思想
2. **Prefix Caching 设计** - 掌握前缀共享的实现技巧
3. **连续批处理调度** - 理解动态批处理的权衡
4. **张量并行实现** - 掌握模型并行的基本原理

### 9.2 推荐阅读顺序

```
1. README.md          → 了解项目概况
2. example.py         → 学习 API 使用
3. llm.py             → 理解整体架构
4. attention.py       → 掌握 PagedAttention
5. cache.py           → 理解 KV Cache 管理
6. scheduler.py       → 学习调度策略
```

---

## 十、总结

Nano-vLLM 是一个优秀的教育型项目，它用精简的代码实现了 vLLM 的核心功能，为学习 LLM 推理优化提供了绝佳的切入点。

### 核心亮点

1. **代码简洁**：约 1,200 行代码实现核心功能
2. **性能出色**：与官方 vLLM 性能相当
3. **架构清晰**：模块化设计，易于理解
4. **功能完整**：支持 PagedAttention、Prefix Caching 等关键特性

### 项目价值

Nano-vLLM 证明了"简单可以胜过复杂"——通过精心设计的架构和算法，小团队甚至个人也能实现与大型项目相媲美的性能。它的价值不仅在于代码本身，更在于展示了如何用最简洁的方式解决复杂问题。

---

**项目链接**：[https://github.com/GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

**推荐学习路径**：先通读代码理解整体架构，然后针对感兴趣的模块深入分析，最后尝试修改和扩展功能。

---

*分析报告生成时间: 2026年2月26日*
