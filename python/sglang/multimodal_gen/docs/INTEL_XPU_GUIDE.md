# SGLang Diffusion Intel XPU 完整部署指南

本文档详细介绍如何在Intel XPU平台上部署和使用SGLang Diffusion进行图片/视频生成，包括环境配置、启动方式、Serving Pipeline流程等。

---

## 目录

1. [系统要求](#1-系统要求)
2. [安装步骤](#2-安装步骤)
3. [启动方式](#3-启动方式)
   - [3.1 单卡启动](#31-单卡启动)
   - [3.2 多卡启动](#32-多卡启动)
4. [Serving Pipeline详解](#4-serving-pipeline详解)
5. [多卡场景下的请求处理流程](#5-多卡场景下的请求处理流程)
6. [性能调优](#6-性能调优)
7. [故障排查](#7-故障排查)

---

## 1. 系统要求

### 1.1 硬件要求

- **Intel XPU设备**:
  - Intel Data Center GPU Max系列 (Ponte Vecchio) - 推荐用于生产环境
  - Intel Arc GPU系列 - 可用于开发测试
  
- **内存**: 建议系统内存 >= 32GB（视模型大小而定）
- **存储**: SSD存储，至少100GB可用空间用于模型权重

### 1.2 软件要求

- **操作系统**: Ubuntu 22.04 或 CentOS 8+
- **Python**: 3.10 或 3.12
- **PyTorch**: 2.8.0+ (with XPU support)
- **Intel Extension for PyTorch (IPEX)**: 2.8.0+
- **oneCCL**: 通常随IPEX一起安装
- **驱动**: Intel GPU驱动 (Level Zero)

---

## 2. 安装步骤

### 2.1 安装Intel XPU驱动

```bash
# 安装Intel GPU驱动和运行时
# 详细步骤请参考: https://dgpu-docs.intel.com/

# 验证驱动安装
xpu-smi discovery

# 查看设备信息
xpu-smi dump -d 0
```

### 2.2 创建Python环境

```bash
# 使用conda创建环境（推荐）
conda create -n sglang-xpu python=3.12 -y
conda activate sglang-xpu

# 或使用venv
python3.12 -m venv sglang-xpu-env
source sglang-xpu-env/bin/activate
```

### 2.3 安装PyTorch with XPU支持

```bash
# 使用Intel提供的wheel包
pip install torch torchvision intel-extension-for-pytorch \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 验证XPU可用性
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}'); print(f'XPU device count: {torch.xpu.device_count()}')"
```

预期输出：
```
XPU available: True
XPU device count: 2  # 根据实际设备数量
```

### 2.4 安装oneCCL（分布式通信）

```bash
# 安装oneCCL绑定（如果未随IPEX安装）
pip install oneccl_bind_pt -f https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 验证xccl backend可用
python -c "import torch.distributed as dist; import oneccl_bindings_for_pytorch; print('oneCCL available')"
```

### 2.5 安装SGLang Diffusion

```bash
# 从源码安装（推荐，以便使用最新的XPU支持）
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 升级pip并安装
pip install --upgrade pip
pip install -e "python[diffusion]"
```

**说明**: 
- SGLang的Python包位于`python`目录下，所以使用`python[diffusion]`
- `-e`表示可编辑模式安装，方便开发和调试
- `[diffusion]`表示安装diffusion相关的额外依赖

### 2.6 验证安装

```bash
# 验证SGLang安装
python -c "from sglang.multimodal_gen import DiffGenerator; print('SGLang Diffusion installed successfully')"

# 验证XPU平台检测
python -c "from sglang.multimodal_gen.runtime.platforms import current_platform; print(f'Current platform: {current_platform.device_type}')"
```

预期输出：
```
Current platform: xpu
```

---

## 3. 启动方式

### 3.1 单卡启动

#### 3.1.1 使用Python API（本地模式）

```python
from sglang.multimodal_gen import DiffGenerator

# 创建生成器实例
generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,  # 单卡
    # XPU特定配置
    text_encoder_cpu_offload=True,  # 启用CPU offload以节省显存
    vae_cpu_offload=True,
    pin_cpu_memory=True,  # 启用内存固定以提高传输速度
)

# 生成视频
video = generator.generate(
    prompt="A curious raccoon peers through a vibrant field of yellow sunflowers",
    return_frames=True,
    save_output=True,
    output_path="./outputs/"
)

print(f"Generated video saved to: {video['output_path']}")
```

#### 3.1.2 使用CLI

```bash
# 基本生成命令
sglang generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 1 \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --pin-cpu-memory \
    --prompt "A curious raccoon in sunflowers" \
    --save-output \
    --output-path ./outputs/

# 查看所有可用选项
sglang generate --help
```

#### 3.1.3 启动HTTP服务器（单卡）

```bash
# 启动服务器
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --log-level info

# 服务器启动后，使用API请求生成
curl -X POST http://localhost:30000/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A curious raccoon in sunflowers",
    "num_frames": 81,
    "height": 480,
    "width": 720
  }'
```

### 3.2 多卡启动

#### 3.2.1 环境变量配置

```bash
# 设置可见的XPU设备
export ZE_AFFINITY_MASK=0,1,2,3  # 使用4张卡

# oneCCL配置（用于分布式通信）
export CCL_LOG_LEVEL=info  # 日志级别: trace, debug, info, warn, error
export CCL_ATL_TRANSPORT=mpi  # 或 ofi (libfabric)

# PyTorch分布式配置
export TORCH_DISTRIBUTED_BACKEND=xccl  # 使用XCCL backend for XPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

#### 3.2.2 数据并行模式（DP）

数据并行在多个设备上复制模型，每个设备处理不同的请求批次，适合高吞吐量场景。

```bash
# 启动4卡数据并行服务器
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 4 \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000 \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --log-level info
```

**Python API示例**:
```python
generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=4,
    dp_size=4,  # 数据并行大小
    text_encoder_cpu_offload=True,
    vae_cpu_offload=True,
)

# 每个DP副本可以独立处理请求
```

#### 3.2.3 序列并行模式（SP）

序列并行将长序列分割到多个设备，适合处理超长视频或高分辨率图像。

```bash
# 启动4卡序列并行服务器（使用Ulysses SP）
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 4 \
    --sp-degree 4 \
    --ulysses-degree 4 \
    --attention-backend fa3 \  # Ring Attention需要FA3
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info

# 或使用Ring Attention（处理更长序列）
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 4 \
    --sp-degree 4 \
    --ring-degree 2 \
    --ulysses-degree 2 \
    --attention-backend fa3 \
    --host 0.0.0.0 \
    --port 30000
```

**Python API示例**:
```python
generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=4,
    sp_degree=4,  # 序列并行度
    ulysses_degree=4,  # Ulysses序列并行
    attention_backend="fa3",
)
```

#### 3.2.4 混合并行模式

结合数据并行（DP）和序列并行（SP）以最大化吞吐量和序列长度。

```bash
# 8卡混合并行: 2x DP, 4x SP
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 8 \
    --dp-size 2 \      # 2个数据并行副本
    --sp-degree 4 \    # 每个副本使用4卡序列并行
    --ulysses-degree 4 \
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info
```

**计算方式**: `num_gpus = dp_size * sp_degree = 2 * 4 = 8`

**Python API示例**:
```python
generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=8,
    dp_size=2,
    sp_degree=4,
    ulysses_degree=4,
)
```

#### 3.2.5 CFG并行模式

Classifier-Free Guidance (CFG)并行在不同设备上同时计算条件和无条件分支，减少延迟。

```bash
# 启动CFG并行（需要偶数张卡）
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 2 \
    --enable-cfg-parallel \  # 启用CFG并行
    --host 0.0.0.0 \
    --port 30000 \
    --log-level info
```

**Python API示例**:
```python
generator = DiffGenerator.from_pretrained(
    model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=2,
    enable_cfg_parallel=True,
)
```

#### 3.2.6 查看分布式状态

```bash
# 在服务器启动后，查看分布式配置
curl http://localhost:30000/health

# 示例响应
{
  "status": "ready",
  "num_gpus": 4,
  "dp_size": 2,
  "sp_degree": 2,
  "platform": "xpu"
}
```

---

## 4. Serving Pipeline详解

SGLang Diffusion的图片/视频生成Pipeline由多个阶段（Stage）组成，每个阶段负责特定的处理任务。

### 4.1 整体架构

```
HTTP请求 → FastAPI Server → Scheduler Client → ZMQ → Scheduler → GPU Worker → Pipeline Stages → 输出
```

### 4.2 核心组件说明

#### 4.2.1 FastAPI Server (`http_server.py`)

- **职责**: 接收HTTP API请求，处理请求验证
- **端点**:
  - `/v1/images/generations` - 图片生成
  - `/v1/videos/generations` - 视频生成
  - `/v1/images/{image_id}` - 查询图片状态
  - `/v1/videos/{video_id}` - 查询视频状态

**关键代码位置**: `runtime/entrypoints/http_server.py`, `runtime/entrypoints/openai/image_api.py`, `runtime/entrypoints/openai/video_api.py`

#### 4.2.2 Scheduler Client (`scheduler_client.py`)

- **职责**: 作为FastAPI进程与Scheduler进程之间的桥梁
- **通信方式**: ZeroMQ (REQ-REP模式)
- **功能**:
  - 序列化HTTP请求为`Req`对象
  - 通过ZMQ发送到Scheduler
  - 等待Scheduler返回结果

**关键代码位置**: `runtime/scheduler_client.py`

#### 4.2.3 Scheduler (`scheduler.py`)

- **职责**: 协调多GPU Worker，管理请求队列和批处理
- **在Rank 0 GPU上运行**
- **功能**:
  - 接收来自Scheduler Client的请求
  - 将请求分发到所有GPU Worker（通过multiprocessing.Pipe）
  - 收集结果并返回给Client

**关键代码位置**: `runtime/managers/scheduler.py`

#### 4.2.4 GPU Worker (`gpu_worker.py`)

- **职责**: 在每个GPU上执行实际的模型推理
- **每个GPU一个Worker进程**
- **功能**:
  - 初始化设备和分布式环境
  - 加载模型组件（Transformer, VAE, Text Encoder等）
  - 执行Pipeline的forward方法

**关键代码位置**: `runtime/managers/gpu_worker.py`

#### 4.2.5 Pipeline Stages

Pipeline由多个Stage顺序执行，每个Stage处理特定任务：

| Stage名称 | 文件 | 功能描述 |
|----------|------|---------|
| `InputValidationStage` | `input_validation.py` | 验证输入参数的有效性 |
| `TextEncodingStage` | `text_encoding.py` | 使用Text Encoder编码提示词 |
| `ImageEncodingStage` | `image_encoding.py` | 编码参考图片（I2V/I2I任务） |
| `LatentPreparationStage` | `latent_preparation.py` | 准备初始噪声latent |
| `TimestepPreparationStage` | `timestep_preparation.py` | 准备去噪时间步 |
| `DenoisingStage` | `denoising.py` | 核心去噪循环（Transformer推理） |
| `DecodingStage` | `decoding.py` | VAE解码latent为像素 |

**关键代码位置**: `runtime/pipelines/stages/`

### 4.3 Pipeline执行流程

```python
# 伪代码展示Pipeline执行流程
class DiffusionPipeline:
    def forward(self, req: Req, server_args: ServerArgs) -> OutputBatch:
        # Stage 1: 输入验证
        batch = self.input_validation_stage.forward(req, server_args)
        
        # Stage 2: 文本编码
        batch = self.text_encoding_stage.forward(batch, server_args)
        # 输出: batch.prompt_embeds, batch.pooled_prompt_embeds
        
        # Stage 3: 图片编码（如果是I2V/I2I）
        if batch.image is not None:
            batch = self.image_encoding_stage.forward(batch, server_args)
            # 输出: batch.image_latent
        
        # Stage 4: Latent准备
        batch = self.latent_preparation_stage.forward(batch, server_args)
        # 输出: batch.latent (初始噪声)
        
        # Stage 5: 时间步准备
        batch = self.timestep_preparation_stage.forward(batch, server_args)
        # 输出: batch.timesteps
        
        # Stage 6: 去噪循环（核心计算）
        batch = self.denoising_stage.forward(batch, server_args)
        # 输出: batch.latent (去噪后的latent)
        
        # Stage 7: VAE解码
        output_batch = self.decoding_stage.forward(batch, server_args)
        # 输出: output_batch.frames (像素数组)
        
        return output_batch
```

### 4.4 请求对象（Req）数据流

```python
# Req对象在Pipeline中的数据变化
class Req:
    # 输入参数
    prompt: str                    # 文本提示词
    image: Optional[PIL.Image]     # 参考图片（I2V/I2I）
    height: int                    # 输出高度
    width: int                     # 输出宽度
    num_frames: int                # 视频帧数（T2V/I2V）
    num_inference_steps: int       # 去噪步数
    guidance_scale: float          # CFG强度
    seed: int                      # 随机种子
    
    # Stage间传递的中间结果
    prompt_embeds: torch.Tensor          # [Stage 2输出] 文本嵌入
    pooled_prompt_embeds: torch.Tensor   # [Stage 2输出] 池化文本嵌入
    image_latent: torch.Tensor           # [Stage 3输出] 图片latent
    latent: torch.Tensor                 # [Stage 4/6输出] 噪声/去噪latent
    timesteps: torch.Tensor              # [Stage 5输出] 时间步序列
    
    # 输出
    output_frames: np.ndarray      # [Stage 7输出] 最终像素数组
```

---

## 5. 多卡场景下的请求处理流程

### 5.1 数据并行（DP）模式流程

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Request                         │
│          POST /v1/videos/generations                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server (Main Process)              │
│  - 解析请求参数                                           │
│  - 创建Req对象                                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (ZMQ REQ-REP)
┌─────────────────────────────────────────────────────────┐
│            Scheduler Client (Singleton)                 │
│  - 序列化Req对象                                         │
│  - 发送到Scheduler                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (ZMQ Socket)
┌─────────────────────────────────────────────────────────┐
│           Scheduler (Rank 0 GPU Process)                │
│  - 接收请求                                              │
│  - 通过Pipe广播给所有Worker                              │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│Worker 0 │   │Worker 1 │   │Worker 2 │   │Worker 3 │
│ XPU:0   │   │ XPU:1   │   │ XPU:2   │   │ XPU:3   │
│         │   │         │   │         │   │         │
│同样的Req │   │同样的Req │   │同样的Req │   │同样的Req │
└────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
     │             │             │             │
     ▼             ▼             ▼             ▼
  Pipeline      Pipeline      Pipeline      Pipeline
  (独立推理)    (独立推理)    (独立推理)    (独立推理)
     │             │             │             │
     ▼             ▼             ▼             ▼
  Output        Output        Output        Output
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                     │
                     ▼ (Pipe收集)
┌─────────────────────────────────────────────────────────┐
│            Scheduler (收集结果)                          │
│  - 从Worker 0获取结果（其他Worker丢弃）                  │
│  - 返回给Scheduler Client                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (ZMQ REP)
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server                             │
│  - 返回生成的视频/图片                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                HTTP Response                            │
│  {                                                      │
│    "id": "video_123",                                   │
│    "frames": [...],                                     │
│    "status": "completed"                                │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

**关键点**:
- 所有GPU Worker独立执行相同的Req
- 只有Rank 0的结果被返回
- 适合高吞吐量场景（每个GPU处理不同请求）

### 5.2 序列并行（SP）模式流程

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Request                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
                [FastAPI → Scheduler Client → Scheduler]
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         所有Worker接收相同的Req                          │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │
     ▼              ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│Worker 0 │   │Worker 1 │   │Worker 2 │   │Worker 3 │
│ XPU:0   │   │ XPU:1   │   │ XPU:2   │   │ XPU:3   │
└────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
     │             │             │             │
     ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│          Text Encoding (Replicated)                     │
│  每个Worker独立编码文本                                  │
│  输出: prompt_embeds (相同)                              │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│        Latent Preparation (Sequence Parallel)           │
│  Worker 0: latent[:, 0:20, ...]                         │
│  Worker 1: latent[:, 20:40, ...]                        │
│  Worker 2: latent[:, 40:60, ...]                        │
│  Worker 3: latent[:, 60:81, ...]                        │
│  (假设81帧，每GPU处理约20帧)                             │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│       Denoising Stage (Sequence Parallel)               │
│  - Ulysses SP: 在attention计算前all-to-all交换          │
│  - Ring Attention: 在KV间循环传递                       │
│                                                         │
│  每个Worker处理序列的一部分，通过XCCL通信同步            │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│        VAE Decoding (Sequence Parallel)                 │
│  Worker 0: frames[0:20, ...]                            │
│  Worker 1: frames[20:40, ...]                           │
│  Worker 2: frames[40:60, ...]                           │
│  Worker 3: frames[60:81, ...]                           │
└────┬──────────────┬──────────────┬──────────────────────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                     │
                     ▼ (Gather到Rank 0)
┌─────────────────────────────────────────────────────────┐
│         Rank 0 收集所有帧                                │
│  frames = [frames_0, frames_1, frames_2, frames_3]      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
                [返回给Scheduler → Client → HTTP Response]
```

**关键点**:
- 序列维度（帧数、时间步等）被分割到多个GPU
- 通过XCCL进行all-to-all, all-gather等集合通信
- 适合超长序列（长视频、高分辨率）

### 5.3 分布式通信细节

在多卡场景下，SGLang使用以下通信方式：

#### 5.3.1 进程间通信（IPC）

**Scheduler ↔ Workers**:
- 使用`multiprocessing.Pipe`进行双向通信
- Scheduler通过Pipe发送Req到所有Worker
- Worker通过Pipe返回结果给Scheduler

#### 5.3.2 GPU间通信（Collective Communications）

**Worker ↔ Worker (XCCL)**:
- `all_reduce`: 聚合梯度（训练时）或中间结果
- `all_gather`: 收集所有GPU的partial序列
- `all_to_all`: Ulysses SP中的序列重排
- `send/recv`: Ring Attention中的KV传递
- `broadcast`: 广播参数更新

**通信组**:
```python
# 序列并行组（SP Group）
sp_group = get_sp_group()  # 包含sp_degree个GPU

# 数据并行组（DP Group）
dp_group = get_dp_group()  # 包含dp_size个副本

# CFG并行组
cfg_group = get_cfg_group()  # 条件/无条件分支

# 张量并行组（TP Group）
tp_group = get_tp_group()  # 模型分片
```

#### 5.3.3 XPU分布式Backend配置

```python
# runtime/distributed/parallel_state.py
def init_process_group(backend=None, ...):
    if is_xpu():
        backend = "xccl"  # Intel XPU使用XCCL backend
    elif is_cuda():
        backend = "nccl"  # NVIDIA GPU使用NCCL
    
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        ...
    )
```

---

## 6. 性能调优

### 6.1 显存优化

#### 6.1.1 CPU Offload

将不常用的模型组件卸载到CPU内存：

```bash
sglang serve \
    --model-path MODEL_PATH \
    --num-gpus 4 \
    --text-encoder-cpu-offload \    # Text Encoder卸载
    --image-encoder-cpu-offload \   # Image Encoder卸载
    --vae-cpu-offload \             # VAE卸载
    --pin-cpu-memory                # 固定CPU内存以加速传输
```

**适用场景**: 显存不足时

#### 6.1.2 FSDP推理

使用FSDP分片模型权重：

```bash
sglang serve \
    --model-path MODEL_PATH \
    --num-gpus 4 \
    --use-fsdp-inference \  # 启用FSDP推理
    --hsdp-shard-dim 4      # 分片维度
```

**适用场景**: 模型太大无法在单卡加载

### 6.2 计算优化

#### 6.2.1 使用IPEX优化

```python
import intel_extension_for_pytorch as ipex

# 在Pipeline初始化时应用IPEX优化
model = ipex.optimize(
    model.to('xpu'),
    dtype=torch.bfloat16,  # 使用BF16
    inplace=True
)
```

#### 6.2.2 Torch Compile

```bash
sglang serve \
    --model-path MODEL_PATH \
    --num-gpus 4 \
    --enable-torch-compile  # 启用torch.compile加速
```

**注意**: 可能导致精度漂移，见PyTorch issue #145213

### 6.3 通信优化

#### 6.3.1 CCL优化参数

```bash
# 启用CCL优化
export CCL_FUSION_THRESHOLD=1
export CCL_FUSION_COUNT_THRESHOLD=256

# 设置优先级模式
export CCL_PRIORITY=direct

# 选择最优传输层
export CCL_ATL_TRANSPORT=mpi  # 或 ofi
```

#### 6.3.2 Attention Backend选择

```bash
# 使用Flash Attention 3（最快，但需要支持）
sglang serve \
    --model-path MODEL_PATH \
    --attention-backend fa3

# 或使用Torch SDPA（XPU默认）
sglang serve \
    --model-path MODEL_PATH \
    --attention-backend sdpa
```

### 6.4 批处理优化

```python
# 批量生成多个请求
generator = DiffGenerator.from_pretrained(...)

prompts = [
    "A raccoon in sunflowers",
    "A cat playing piano",
    "A dog running on beach"
]

# 批量处理
for prompt in prompts:
    video = generator.generate(prompt, ...)
```

---

## 7. 故障排查

### 7.1 常见问题

#### 问题1: "CCL not available"

**症状**:
```
RuntimeError: oneCCL not found
```

**解决方案**:
```bash
# 检查oneCCL是否安装
python -c "import oneccl_bindings_for_pytorch; print('CCL available')"

# 如果未安装
pip install oneccl_bind_pt -f https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 验证xccl backend
python -c "import torch.distributed as dist; print('xccl' in dir(dist.Backend))"
```

#### 问题2: "No XPU devices found"

**症状**:
```
RuntimeError: XPU device not found
```

**解决方案**:
```bash
# 检查驱动
xpu-smi discovery

# 检查PyTorch XPU支持
python -c "import torch; print(torch.xpu.device_count())"

# 设置设备可见性
export ZE_AFFINITY_MASK=0,1,2,3
```

#### 问题3: 分布式初始化失败

**症状**:
```
RuntimeError: Failed to initialize distributed environment
```

**解决方案**:
```bash
# 使用gloo backend作为fallback
export TORCH_DISTRIBUTED_BACKEND=gloo

# 增加超时时间
export TORCH_DISTRIBUTED_TIMEOUT=1800

# 检查端口是否被占用
netstat -tuln | grep 29500
```

#### 问题4: 显存不足（OOM）

**症状**:
```
RuntimeError: XPU out of memory
```

**解决方案**:
```bash
# 启用CPU offload
sglang serve ... \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --dit-cpu-offload

# 或使用FSDP推理
sglang serve ... \
    --use-fsdp-inference

# 减小批量大小或序列长度
sglang generate ... \
    --num-frames 41  # 从81减到41
```

#### 问题5: 性能不佳

**症状**: 生成速度慢于预期

**解决方案**:
```bash
# 检查是否使用了正确的backend
python -c "import torch.distributed as dist; dist.init_process_group('xccl'); print(dist.get_backend())"

# 启用性能分析
export CCL_LOG_LEVEL=info
export IPEX_VERBOSE=1

# 使用性能优化选项
sglang serve ... \
    --enable-torch-compile \
    --attention-backend fa3 \
    --pin-cpu-memory
```

### 7.2 日志级别调整

```bash
# 详细日志
sglang serve ... --log-level debug

# CCL调试信息
export CCL_LOG_LEVEL=debug

# PyTorch分布式调试
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

### 7.3 性能分析

```bash
# 使用VTune分析（Intel工具）
vtune -collect gpu-hotspots -- python -m sglang.serve ...

# 或使用PyTorch Profiler
python -m torch.utils.bottleneck your_script.py
```

---

## 8. 参考资料

### 8.1 官方文档

- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [oneCCL Documentation](https://oneapi-src.github.io/oneCCL/)
- [Intel GPU驱动安装](https://dgpu-docs.intel.com/)
- [PyTorch XPU Backend](https://pytorch.org/docs/stable/notes/xpu.html)

### 8.2 相关文件

- XPU分布式支持: `docs/XPU_DISTRIBUTED.md`
- Backend命名说明: `docs/BACKEND_NAMING.md`
- CLI使用指南: `docs/cli.md`
- 安装指南: `docs/install.md`

### 8.3 代码位置

- XPU平台实现: `runtime/platforms/xpu.py`
- 分布式通信: `runtime/distributed/`
- Pipeline实现: `runtime/pipelines/`
- HTTP服务器: `runtime/entrypoints/http_server.py`
- GPU Worker: `runtime/managers/gpu_worker.py`

---

## 附录A: 完整启动脚本示例

### A.1 单卡T2V生成脚本

```bash
#!/bin/bash
# single_gpu_t2v.sh

export ZE_AFFINITY_MASK=0
export CCL_LOG_LEVEL=warn

sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 1 \
    --workload-type t2v \
    --host 0.0.0.0 \
    --port 30000 \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --pin-cpu-memory \
    --log-level info
```

### A.2 多卡DP+SP混合脚本

```bash
#!/bin/bash
# multi_gpu_hybrid.sh

export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
export CCL_LOG_LEVEL=info
export CCL_ATL_TRANSPORT=mpi
export TORCH_DISTRIBUTED_BACKEND=xccl

sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 8 \
    --dp-size 2 \
    --sp-degree 4 \
    --ulysses-degree 4 \
    --workload-type t2v \
    --host 0.0.0.0 \
    --port 30000 \
    --attention-backend fa3 \
    --enable-torch-compile \
    --log-level info
```

### A.3 Python客户端示例

```python
#!/usr/bin/env python3
# client_example.py

import requests
import json

def generate_video(prompt, server_url="http://localhost:30000"):
    """调用SGLang服务器生成视频"""
    
    endpoint = f"{server_url}/v1/videos/generations"
    
    payload = {
        "prompt": prompt,
        "num_frames": 81,
        "height": 480,
        "width": 720,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": 42
    }
    
    response = requests.post(
        endpoint,
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Video generated: {result['id']}")
        print(f"Status: {result['status']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers"
    result = generate_video(prompt)
```

---

## 附录B: 性能基准测试

### B.1 测试环境

*性能数据取决于具体硬件配置、模型大小、序列长度等因素。*

建议在您的实际环境中进行基准测试：

```bash
# 单卡基准测试
time sglang generate \
    --model-path MODEL_PATH \
    --num-gpus 1 \
    --prompt "Your test prompt" \
    --num-frames 81

# 多卡基准测试
time sglang generate \
    --model-path MODEL_PATH \
    --num-gpus 4 \
    --dp-size 4 \
    --prompt "Your test prompt" \
    --num-frames 81
```

### B.2 性能调优建议

参考[性能调优章节](#6-性能调优)进行优化配置。

---

**文档版本**: v1.0  
**最后更新**: 2024-11  
**维护者**: SGLang Diffusion Team
