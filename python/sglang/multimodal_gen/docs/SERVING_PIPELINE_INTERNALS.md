# SGLang Diffusion Serving Pipeline 技术详解

本文档深入剖析SGLang Diffusion的Serving架构和Pipeline执行流程，重点说明多卡场景下的请求处理机制。

---

## 目录

1. [架构概览](#1-架构概览)
2. [组件详细说明](#2-组件详细说明)
3. [请求生命周期](#3-请求生命周期)
4. [Pipeline Stage详解](#4-pipeline-stage详解)
5. [多卡分布式执行](#5-多卡分布式执行)
6. [数据流和内存管理](#6-数据流和内存管理)

---

## 1. 架构概览

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         HTTP Client                             │
│                    (curl / Python SDK)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP Request
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Image API    │  │ Video API    │  │ Health API   │          │
│  │ Router       │  │ Router       │  │ Router       │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                 │                                     │
│         └─────────┬───────┘                                     │
│                   ▼                                             │
│         ┌──────────────────┐                                    │
│         │ Request Handler  │                                    │
│         └────────┬─────────┘                                    │
└──────────────────┼─────────────────────────────────────────────┘
                   │ Create Req object
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Scheduler Client (Singleton)                   │
│  - ZeroMQ REQ Socket                                            │
│  - Async Request Queue                                          │
│  - Worker Task                                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │ ZMQ REQ-REP (TCP Socket)
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│             Scheduler Process (Rank 0 GPU)                      │
│  ┌─────────────────────────────────────────────┐                │
│  │ ZeroMQ REP Socket                           │                │
│  │  - Receive Req from Client                  │                │
│  │  - Return OutputBatch                       │                │
│  └───────────────────┬─────────────────────────┘                │
│                      │                                          │
│  ┌───────────────────▼─────────────────────────┐                │
│  │ Request Distribution                        │                │
│  │  - Broadcast Req to all Workers via Pipe   │                │
│  │  - Coordinate execution                     │                │
│  └───────────────────┬─────────────────────────┘                │
└────────────────────┬─┼──────────────┬──────────────────────────┘
                     │ │              │
        ┌────────────┘ │              └────────────┐
        │ Pipe         │ Pipe                Pipe  │
        ▼              ▼                      ▼    ▼
┌─────────────┐ ┌─────────────┐ ... ┌─────────────┐
│ GPU Worker 0│ │ GPU Worker 1│     │ GPU Worker N│
│  (Rank 0)   │ │  (Rank 1)   │     │  (Rank N)   │
│             │ │             │     │             │
│ ┌─────────┐ │ │ ┌─────────┐ │     │ ┌─────────┐ │
│ │Pipeline │ │ │ │Pipeline │ │     │ │Pipeline │ │
│ │         │ │ │ │         │ │     │ │         │ │
│ │[Stages] │ │ │ │[Stages] │ │     │ │[Stages] │ │
│ └─────────┘ │ │ └─────────┘ │     │ └─────────┘ │
│      │      │ │      │      │     │      │      │
│   XPU:0     │ │   XPU:1     │     │   XPU:N     │
└──────┼──────┘ └──────┼──────┘     └──────┼──────┘
       │               │                   │
       └───────────────┴───────────────────┘
                       │
          Distributed Communication (XCCL)
          - all_reduce, all_gather, etc.
```

### 1.2 进程模型

SGLang Diffusion采用**多进程架构**：

1. **主进程**: 运行FastAPI服务器
2. **Worker进程**: 每个GPU一个进程，执行模型推理
   - 进程0: Scheduler + Worker (Rank 0)
   - 进程1-N: Worker (Rank 1-N)

**进程启动流程**:
```python
# launch_server.py
def launch_server(server_args):
    processes = []
    
    # 为每个GPU创建进程
    for i in range(num_gpus):
        process = mp.Process(
            target=run_scheduler_process,
            args=(
                local_rank=i,
                rank=i,
                server_args,
                ...
            )
        )
        process.start()
        processes.append(process)
    
    # 启动FastAPI服务器
    uvicorn.run(app, host=host, port=port)
```

---

## 2. 组件详细说明

### 2.1 FastAPI Server

**文件位置**: `runtime/entrypoints/http_server.py`

**职责**:
- 提供HTTP API端点
- 请求参数验证和解析
- 将HTTP请求转换为内部`Req`对象
- 返回生成结果

**核心代码**:
```python
# http_server.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化Scheduler Client
    scheduler_client.initialize(server_args)
    
    # 启动ZMQ Broker（处理离线请求）
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))
    
    yield
    
    # 关闭时清理资源
    broker_task.cancel()
    scheduler_client.close()

def create_app(server_args):
    app = FastAPI(lifespan=lifespan)
    app.include_router(image_api.router)
    app.include_router(video_api.router)
    return app
```

**API端点示例**:
```python
# runtime/entrypoints/openai/video_api.py
@router.post("/v1/videos/generations")
async def create_video(request: VideoGenerationRequest):
    # 1. 创建Req对象
    req = Req(
        prompt=request.prompt,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        ...
    )
    
    # 2. 通过Scheduler Client转发
    output_batch = await scheduler_client.forward(req)
    
    # 3. 返回结果
    return VideoResponse(
        id=output_batch.video_id,
        frames=output_batch.frames,
        status="completed"
    )
```

### 2.2 Scheduler Client

**文件位置**: `runtime/scheduler_client.py`

**职责**:
- 单例模式，连接到Scheduler进程
- 使用ZeroMQ REQ-REP模式通信
- 异步队列保证请求顺序

**核心实现**:
```python
class SchedulerClient:
    def __init__(self):
        self._request_queue = asyncio.Queue()  # 请求队列
        self._worker_task = None  # 后台worker
    
    def initialize(self, server_args):
        self.context = zmq.asyncio.Context()
        self.scheduler_socket = self.context.socket(zmq.REQ)
        # 连接到Scheduler的ZMQ端点
        self.scheduler_socket.connect(server_args.scheduler_endpoint())
    
    async def forward(self, batch: Req) -> Req:
        # 将请求加入队列
        future = asyncio.Future()
        await self._request_queue.put((batch, future))
        return await future
    
    async def _worker_loop(self):
        """后台worker循环处理队列中的请求"""
        while True:
            batch, future = await self._request_queue.get()
            try:
                # 发送Req到Scheduler
                await self.scheduler_socket.send_pyobj(batch)
                # 等待Scheduler返回结果
                response = await self.scheduler_socket.recv_pyobj()
                future.set_result(response)
            except Exception as e:
                future.set_exception(e)
```

**为什么使用队列**:
- ZeroMQ REQ-REP是严格的请求-响应模式
- 必须等待一个响应后才能发送下一个请求
- 队列保证请求的序列化处理

### 2.3 Scheduler

**文件位置**: `runtime/managers/scheduler.py`

**职责**:
- 运行在Rank 0 GPU进程中
- 接收来自Scheduler Client的请求
- 协调所有GPU Worker执行
- 收集结果并返回

**核心流程**:
```python
class Scheduler:
    def __init__(self, server_args, task_pipes_to_slaves, result_pipes_from_slaves):
        self.gpu_worker = GPUWorker(...)  # Rank 0的Worker
        self.task_pipes = task_pipes_to_slaves  # 发送任务到其他Worker
        self.result_pipes = result_pipes_from_slaves  # 接收结果
        
        # ZMQ REP socket监听Client请求
        self.zmq_socket = context.socket(zmq.REP)
        self.zmq_socket.bind(f"tcp://*:{port}")
    
    def event_loop(self):
        """主事件循环"""
        while True:
            # 1. 从ZMQ socket接收请求
            req = self.zmq_socket.recv_pyobj()
            
            # 2. 广播请求到所有slave workers
            for pipe in self.task_pipes:
                pipe.send(req)
            
            # 3. 本地执行（Rank 0）
            output_batch = self.gpu_worker.execute_forward([req], server_args)
            
            # 4. 等待slave workers完成（同步点）
            for pipe in self.result_pipes:
                _ = pipe.recv()  # 仅等待完成信号，不使用结果
            
            # 5. 返回结果给Client
            self.zmq_socket.send_pyobj(output_batch)
```

### 2.4 GPU Worker

**文件位置**: `runtime/managers/gpu_worker.py`

**职责**:
- 每个GPU一个Worker实例
- 初始化设备和分布式环境
- 加载模型组件
- 执行Pipeline推理

**初始化流程**:
```python
class GPUWorker:
    def __init__(self, local_rank, rank, master_port, server_args):
        self.local_rank = local_rank
        self.rank = rank
        
        # 1. 设置设备
        set_device(self.local_rank)  # 设置XPU设备
        
        # 2. 设置分布式环境变量
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(num_gpus)
        os.environ["MASTER_PORT"] = str(master_port)
        
        # 3. 初始化分布式环境
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=server_args.tp_size,
            sp_size=server_args.sp_degree,
            dp_size=server_args.dp_size,
            ...
        )
        
        # 4. 构建Pipeline
        self.pipeline = build_pipeline(server_args)
    
    def execute_forward(self, batch, server_args):
        """执行前向传播"""
        req = batch[0]
        output_batch = self.pipeline.forward(req, server_args)
        return output_batch
```

**进程启动**:
```python
def run_scheduler_process(local_rank, rank, master_port, server_args, ...):
    # 配置日志
    configure_logger(server_args)
    
    # 创建Scheduler（仅Rank 0）或等待任务（Rank 1-N）
    if rank == 0:
        scheduler = Scheduler(...)
        scheduler.event_loop()  # 进入事件循环
    else:
        # Slave workers等待任务
        while True:
            req = task_pipe.recv()
            output = gpu_worker.execute_forward([req], server_args)
            result_pipe.send("done")
```

---

## 3. 请求生命周期

### 3.1 完整流程时序图

```
Client          FastAPI         SchedulerClient    Scheduler(R0)   Worker1-N       Pipeline
  │                │                   │                │              │              │
  ├─HTTP POST──────▶                   │                │              │              │
  │                ├─parse request─────▶                │              │              │
  │                │                   │                │              │              │
  │                │                   ├─create Req─────▶              │              │
  │                │                   │                │              │              │
  │                │                   ├─ZMQ send───────▶              │              │
  │                │                   │                │              │              │
  │                │                   │                ├─Pipe send────▶              │
  │                │                   │                │              │              │
  │                │                   │                ├─execute──────┼──forward────▶
  │                │                   │                │              │              │
  │                │                   │                │              ├──execute─────┼──forward──▶
  │                │                   │                │              │              │
  │                │                   │                │              │              ├─Stage 1: Validate
  │                │                   │                │              │              ├─Stage 2: Text Encode
  │                │                   │                │              │              ├─Stage 3: Image Encode
  │                │                   │                │              │              ├─Stage 4: Latent Prep
  │                │                   │                │              │              ├─Stage 5: Timestep Prep
  │                │                   │                │              │              ├─Stage 6: Denoising◄─┐
  │                │                   │                │              │              │  (XCCL sync)        │
  │                │                   │                │              │◄─────────────┼─────────────────────┘
  │                │                   │                │              │              ├─Stage 7: VAE Decode
  │                │                   │                │              │              │
  │                │                   │                │              │              ├─return OutputBatch──▶
  │                │                   │                │              │◄─────────────┤
  │                │                   │                │◄─────────────┤              │
  │                │                   │                │              │              │
  │                │                   │                ├─Pipe recv────┤              │
  │                │                   │                │  (sync)      │              │
  │                │                   │                │              │              │
  │                │                   │◄─ZMQ reply─────┤              │              │
  │                │◄──────────────────┤                │              │              │
  │◄─HTTP Response─┤                   │                │              │              │
  │                │                   │                │              │              │
```

### 3.2 详细步骤说明

#### 步骤1: HTTP请求到达

```python
# Client发送请求
POST /v1/videos/generations
{
  "prompt": "A curious raccoon",
  "num_frames": 81,
  "height": 480,
  "width": 720,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42
}
```

#### 步骤2: FastAPI解析请求

```python
# runtime/entrypoints/openai/video_api.py
@router.post("/v1/videos/generations")
async def create_video(request: VideoGenerationRequest):
    # 验证参数
    if request.height % 8 != 0:
        raise HTTPException(status_code=400, detail="Height must be divisible by 8")
    
    # 创建Req对象
    req = Req(
        prompt=request.prompt,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        # 生成唯一ID
        request_id=str(uuid.uuid4()),
        # 性能日志
        perf_logger=PerfLogger() if enable_perf else None,
    )
    
    # 转发到Scheduler Client
    output_batch = await scheduler_client.forward(req)
    
    return output_batch
```

#### 步骤3: Scheduler Client转发

```python
# runtime/scheduler_client.py
async def forward(self, batch: Req) -> Req:
    # 创建Future等待结果
    future = asyncio.Future()
    
    # 加入请求队列
    await self._request_queue.put((batch, future))
    
    # 等待Worker处理完成
    return await future

# Worker Loop处理队列
async def _worker_loop(self):
    batch, future = await self._request_queue.get()
    
    # 通过ZMQ发送到Scheduler
    await self.scheduler_socket.send_pyobj(batch)
    
    # 等待Scheduler返回
    response = await self.scheduler_socket.recv_pyobj()
    
    # 设置Future结果
    future.set_result(response)
```

#### 步骤4: Scheduler分发任务

```python
# runtime/managers/scheduler.py
def event_loop(self):
    # 接收ZMQ请求
    req = self.zmq_socket.recv_pyobj()
    
    # 广播到所有slave workers
    for pipe in self.task_pipes_to_slaves:
        pipe.send(req)  # 发送完整Req对象
    
    # 本地执行（Rank 0）
    output_batch = self.gpu_worker.execute_forward([req], server_args)
    
    # 等待其他workers完成
    for pipe in self.result_pipes_from_slaves:
        _ = pipe.recv()  # 同步点
    
    # 返回结果
    self.zmq_socket.send_pyobj(output_batch)
```

#### 步骤5: GPU Workers执行

```python
# 所有Workers同时执行
# runtime/managers/gpu_worker.py
def execute_forward(self, batch, server_args):
    req = batch[0]
    
    # 执行Pipeline
    output_batch = self.pipeline.forward(req, server_args)
    
    return output_batch
```

#### 步骤6: Pipeline执行（见下节详细说明）

#### 步骤7: 结果返回

```python
# Rank 0返回结果
output_batch = {
    "request_id": req.request_id,
    "frames": np.array([...]),  # 生成的视频帧
    "status": "completed",
    "metrics": {
        "total_time": 25.3,
        "denoising_time": 22.1,
        "decoding_time": 2.8,
    }
}
```

---

## 4. Pipeline Stage详解

### 4.1 Stage基类

**文件位置**: `runtime/pipelines/stages/base.py`

```python
class PipelineStage(ABC):
    """Pipeline Stage基类"""
    
    @abstractmethod
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """执行Stage的前向传播"""
        pass
    
    @property
    def parallelism_type(self) -> StageParallelismType:
        """Stage的并行类型"""
        return StageParallelismType.REPLICATED  # 默认在所有GPU上复制执行
    
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """验证输入数据"""
        return VerificationResult()  # 默认不验证
```

### 4.2 Stage 1: Input Validation

**文件**: `input_validation.py`

**职责**: 验证请求参数的合法性

```python
class InputValidationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 验证维度
        assert batch.height % 8 == 0, "Height must be divisible by 8"
        assert batch.width % 8 == 0, "Width must be divisible by 8"
        assert batch.num_frames > 0, "num_frames must be positive"
        
        # 验证参数范围
        assert 1.0 <= batch.guidance_scale <= 20.0, "guidance_scale out of range"
        assert 1 <= batch.num_inference_steps <= 100, "steps out of range"
        
        # 设置默认值
        if batch.seed is None:
            batch.seed = random.randint(0, 2**32 - 1)
        
        return batch
```

### 4.3 Stage 2: Text Encoding

**文件**: `text_encoding.py`

**职责**: 使用Text Encoder将提示词编码为嵌入向量

```python
class TextEncodingStage(PipelineStage):
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
    
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Tokenize文本
        text_inputs = self.tokenizer(
            batch.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # 2. 编码为embedding
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to(device),
            )[0]  # [batch_size, seq_len, hidden_dim]
        
        # 3. 如果启用CFG，还需要编码空提示
        if batch.guidance_scale > 1.0:
            negative_embeds = self.text_encoder(
                self.tokenizer("", ...).input_ids.to(device)
            )[0]
            
            # 拼接条件和无条件嵌入
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        
        # 4. 保存到batch
        batch.prompt_embeds = prompt_embeds
        batch.pooled_prompt_embeds = pooled_embeds  # 如果模型需要
        
        return batch
```

**并行特性**:
- **Replicated**: 所有GPU独立执行
- 结果相同（相同的prompt产生相同的embedding）

### 4.4 Stage 3: Image Encoding (I2V/I2I)

**文件**: `image_encoding.py`

**职责**: 编码参考图片为latent表示

```python
class ImageEncodingStage(PipelineStage):
    def __init__(self, vae):
        self.vae = vae
    
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.image is None:
            return batch  # T2V任务跳过
        
        # 1. 预处理图片
        image = batch.image.resize((batch.width, batch.height))
        image_tensor = self.preprocess(image).to(device)
        
        # 2. VAE编码
        with torch.no_grad():
            image_latent = self.vae.encode(image_tensor).latent_dist.sample()
            image_latent = image_latent * self.vae.config.scaling_factor
        
        # 3. 保存到batch
        batch.image_latent = image_latent
        
        return batch
```

### 4.5 Stage 4: Latent Preparation

**文件**: `latent_preparation.py`

**职责**: 准备初始噪声latent

```python
class LatentPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. 计算latent维度
        latent_height = batch.height // 8  # VAE下采样因子
        latent_width = batch.width // 8
        latent_frames = batch.num_frames
        
        # 2. 如果是I2V，使用图片latent作为初始
        if batch.image_latent is not None:
            latent = batch.image_latent
        else:
            # 3. 生成随机噪声
            generator = torch.Generator(device=device).manual_seed(batch.seed)
            latent = torch.randn(
                (1, 4, latent_frames, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        
        # 4. 序列并行：分割latent到不同GPU
        if server_args.sp_degree > 1:
            sp_rank = get_sp_group().rank
            sp_size = get_sp_group().world_size
            
            # 在帧维度上分割
            chunk_size = latent_frames // sp_size
            start = sp_rank * chunk_size
            end = start + chunk_size if sp_rank < sp_size - 1 else latent_frames
            
            latent = latent[:, :, start:end, :, :]
        
        batch.latent = latent
        return batch
```

**序列并行示例**:
```
总帧数: 81
SP度: 4

GPU 0: latent[:, :, 0:20, :, :]    # 帧 0-19
GPU 1: latent[:, :, 20:40, :, :]   # 帧 20-39
GPU 2: latent[:, :, 40:60, :, :]   # 帧 40-59
GPU 3: latent[:, :, 60:81, :, :]   # 帧 60-80
```

### 4.6 Stage 5: Timestep Preparation

**文件**: `timestep_preparation.py`

**职责**: 准备去噪时间步序列

```python
class TimestepPreparationStage(PipelineStage):
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. 设置scheduler的时间步数
        self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
        
        # 2. 获取时间步序列
        timesteps = self.scheduler.timesteps  # [1000, 950, 900, ..., 50, 0]
        
        # 3. 保存到batch
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        
        return batch
```

### 4.7 Stage 6: Denoising (核心)

**文件**: `denoising.py`

**职责**: 核心去噪循环，执行Transformer推理

```python
class DenoisingStage(PipelineStage):
    def __init__(self, transformer, scheduler):
        self.transformer = transformer
        self.scheduler = scheduler
    
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        latent = batch.latent
        timesteps = batch.timesteps
        
        # 去噪循环
        for i, t in enumerate(timesteps):
            # 1. 准备timestep输入
            timestep = t.expand(latent.shape[0])
            
            # 2. 如果启用CFG，复制latent
            if batch.guidance_scale > 1.0:
                latent_model_input = torch.cat([latent] * 2)  # [uncond, cond]
            else:
                latent_model_input = latent
            
            # 3. Transformer前向传播
            noise_pred = self.transformer(
                latent_model_input,
                timestep=timestep,
                encoder_hidden_states=batch.prompt_embeds,
                return_dict=False,
            )[0]
            
            # 4. 执行CFG
            if batch.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + batch.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            
            # 5. Scheduler更新latent
            latent = self.scheduler.step(
                noise_pred, t, latent, return_dict=False
            )[0]
            
            # 6. 序列并行同步（如果启用）
            if server_args.sp_degree > 1:
                # Ulysses SP: all-to-all交换
                # Ring Attention: send/recv KV
                latent = self._sync_sequence_parallel(latent)
        
        # 保存去噪后的latent
        batch.latent = latent
        return batch
    
    def _sync_sequence_parallel(self, latent):
        """序列并行同步"""
        sp_group = get_sp_group()
        
        if self.sp_type == "ulysses":
            # Ulysses: all-to-all重排序列
            latent = sp_group.all_to_all(latent, dim=2)  # 帧维度
        elif self.sp_type == "ring":
            # Ring: 循环传递KV
            latent = sp_group.ring_forward(latent)
        
        return latent
```

**多卡通信示例（Ulysses SP）**:
```python
# 去噪前
GPU 0: latent[0:20]   GPU 1: latent[20:40]   GPU 2: latent[40:60]   GPU 3: latent[60:81]

# Transformer.forward中的all-to-all交换
# 目的：每个GPU获得完整序列的一部分token

# all-to-all后（用于attention计算）
GPU 0: [latent[0], latent[20], latent[40], latent[60], ...]  # 每个GPU获得所有帧的1/4 tokens
GPU 1: [latent[1], latent[21], latent[41], latent[61], ...]
GPU 2: [latent[2], latent[22], latent[42], latent[62], ...]
GPU 3: [latent[3], latent[23], latent[43], latent[63], ...]

# 计算attention

# all-to-all恢复
GPU 0: latent[0:20]   GPU 1: latent[20:40]   GPU 2: latent[40:60]   GPU 3: latent[60:81]
```

### 4.8 Stage 7: VAE Decoding

**文件**: `decoding.py`

**职责**: 将latent解码为像素

```python
class DecodingStage(PipelineStage):
    def __init__(self, vae):
        self.vae = vae
    
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        latent = batch.latent
        
        # 1. 缩放latent
        latent = latent / self.vae.config.scaling_factor
        
        # 2. 序列并行：每个GPU解码自己的部分
        # latent已经是分片的，直接解码
        
        # 3. VAE解码
        with torch.no_grad():
            frames = self.vae.decode(latent).sample
        
        # 4. 后处理
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().permute(0, 2, 3, 4, 1).float().numpy()
        frames = (frames * 255).round().astype("uint8")
        
        # 5. 序列并行：收集所有帧到Rank 0
        if server_args.sp_degree > 1:
            sp_group = get_sp_group()
            if sp_group.rank == 0:
                # Rank 0收集所有帧
                all_frames = [frames]
                for rank in range(1, sp_group.world_size):
                    recv_frames = sp_group.recv(src=rank)
                    all_frames.append(recv_frames)
                frames = np.concatenate(all_frames, axis=1)  # 在帧维度拼接
            else:
                # 其他rank发送给Rank 0
                sp_group.send(frames, dst=0)
                frames = None
        
        # 6. 创建OutputBatch
        output_batch = OutputBatch(
            request_id=batch.request_id,
            frames=frames if sp_group.rank == 0 else None,
            status="completed",
            metrics=batch.perf_logger.get_metrics() if batch.perf_logger else None,
        )
        
        return output_batch
```

---

## 5. 多卡分布式执行

### 5.1 数据并行（DP）详细流程

```python
# 假设4卡DP
# 每个GPU独立处理不同的请求

# Scheduler分发
for gpu_id in range(4):
    gpu_workers[gpu_id].execute_forward([reqs[gpu_id]], server_args)

# GPU 0处理req_0
output_0 = pipeline.forward(req_0)

# GPU 1处理req_1
output_1 = pipeline.forward(req_1)

# GPU 2处理req_2
output_2 = pipeline.forward(req_2)

# GPU 3处理req_3
output_3 = pipeline.forward(req_3)

# 收集结果
outputs = [output_0, output_1, output_2, output_3]
```

**特点**:
- 无GPU间通信（完全独立）
- 吞吐量 = 单卡吞吐量 × GPU数量
- 延迟 = 单卡延迟

### 5.2 序列并行（SP）详细流程

```python
# 假设4卡SP，81帧视频

# Stage 4: Latent Preparation
# 每个GPU生成自己负责的帧
if sp_rank == 0:
    latent = randn([1, 4, 20, H, W])  # 帧0-19
elif sp_rank == 1:
    latent = randn([1, 4, 20, H, W])  # 帧20-39
elif sp_rank == 2:
    latent = randn([1, 4, 20, H, W])  # 帧40-59
elif sp_rank == 3:
    latent = randn([1, 4, 21, H, W])  # 帧60-80

# Stage 6: Denoising
for t in timesteps:
    # Transformer forward中的all-to-all
    # 在attention前：重排序列使每个GPU能看到所有帧
    latent_for_attn = sp_group.all_to_all(latent, dim=2)
    
    # 计算attention（每个GPU处理部分tokens）
    attn_output = attention(latent_for_attn, ...)
    
    # all-to-all恢复
    latent = sp_group.all_to_all(attn_output, dim=2)
    
    # FFN（在原始分片上计算，无通信）
    latent = ffn(latent)

# Stage 7: Decoding
# 每个GPU解码自己的帧
frames_local = vae.decode(latent)

# Gather到Rank 0
if sp_rank == 0:
    frames = [frames_local]
    frames.append(sp_group.recv(src=1))
    frames.append(sp_group.recv(src=2))
    frames.append(sp_group.recv(src=3))
    frames = concat(frames, dim=1)
else:
    sp_group.send(frames_local, dst=0)
```

### 5.3 通信操作详解

#### All-Reduce
```python
# 用于聚合梯度或中间结果
# 每个GPU输入不同的tensor，输出相同的sum

# 输入
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]
GPU 3: [10, 11, 12]

# all_reduce(op=SUM)
result = sp_group.all_reduce(tensor, op=dist.ReduceOp.SUM)

# 输出（每个GPU都得到相同结果）
GPU 0: [22, 26, 30]  # [1+4+7+10, 2+5+8+11, 3+6+9+12]
GPU 1: [22, 26, 30]
GPU 2: [22, 26, 30]
GPU 3: [22, 26, 30]
```

#### All-Gather
```python
# 收集所有GPU的tensor

# 输入
GPU 0: [0, 1]
GPU 1: [2, 3]
GPU 2: [4, 5]
GPU 3: [6, 7]

# all_gather
gathered = sp_group.all_gather(tensor)

# 输出（每个GPU都得到完整数据）
GPU 0: [0, 1, 2, 3, 4, 5, 6, 7]
GPU 1: [0, 1, 2, 3, 4, 5, 6, 7]
GPU 2: [0, 1, 2, 3, 4, 5, 6, 7]
GPU 3: [0, 1, 2, 3, 4, 5, 6, 7]
```

#### All-to-All
```python
# Ulysses SP的核心操作：重排序列

# 输入（每个GPU持有连续的帧）
GPU 0: frames[0:20]
GPU 1: frames[20:40]
GPU 2: frames[40:60]
GPU 3: frames[60:80]

# all_to_all(scatter_dim=2, gather_dim=1)
# 将每个GPU的数据分成4份，交换

# 输出（每个GPU持有所有帧的部分tokens）
GPU 0: [frame0_tokens0, frame20_tokens0, frame40_tokens0, frame60_tokens0, ...]
GPU 1: [frame0_tokens1, frame20_tokens1, frame40_tokens1, frame60_tokens1, ...]
GPU 2: [frame0_tokens2, frame20_tokens2, frame40_tokens2, frame60_tokens2, ...]
GPU 3: [frame0_tokens3, frame20_tokens3, frame40_tokens3, frame60_tokens3, ...]
```

---

## 6. 数据流和内存管理

### 6.1 内存布局

```
┌──────────────────────────────────────────────────────────┐
│                    GPU Memory Layout                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────┐          │
│  │ Model Weights (Frozen)                     │          │
│  │  - Transformer: ~8GB                       │          │
│  │  - Text Encoder: ~1GB (可offload)          │          │
│  │  - VAE: ~3GB (可offload)                   │          │
│  └────────────────────────────────────────────┘          │
│                                                          │
│  ┌────────────────────────────────────────────┐          │
│  │ Activations (Dynamic)                      │          │
│  │  - Latent: [1, 4, F, H/8, W/8] ~500MB      │          │
│  │  - Prompt Embeds: [77, 768] ~0.3MB         │          │
│  │  - Intermediate: varies                    │          │
│  └────────────────────────────────────────────┘          │
│                                                          │
│  ┌────────────────────────────────────────────┐          │
│  │ KV Cache (if enabled)                      │          │
│  │  - Per layer: ~200MB                       │          │
│  └────────────────────────────────────────────┘          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 CPU Offload机制

```python
# runtime/loader/component_loader.py
class PipelineComponentLoader:
    def load_text_encoder(self, server_args):
        text_encoder = TextEncoder.from_pretrained(...)
        
        if server_args.text_encoder_cpu_offload:
            # 加载到CPU
            text_encoder = text_encoder.to("cpu")
            
            if server_args.pin_cpu_memory:
                # 固定内存以加速PCIe传输
                text_encoder = self._pin_memory(text_encoder)
        else:
            # 加载到GPU
            text_encoder = text_encoder.to(device)
        
        return text_encoder
    
    def _pin_memory(self, model):
        """固定模型参数在CPU内存中"""
        for param in model.parameters():
            if not param.is_pinned():
                param.data = param.data.pin_memory()
        return model
```

**使用流程**:
```python
# Stage 2: Text Encoding
if server_args.text_encoder_cpu_offload:
    # 1. 将输入移到CPU
    text_inputs = text_inputs.to("cpu")
    
    # 2. 在CPU上执行
    prompt_embeds = text_encoder(text_inputs)
    
    # 3. 将输出移回GPU
    prompt_embeds = prompt_embeds.to(device)
else:
    # 直接在GPU上执行
    prompt_embeds = text_encoder(text_inputs.to(device))
```

### 6.3 FSDP推理

```python
# runtime/managers/gpu_worker.py
if server_args.use_fsdp_inference:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    # 分片Transformer权重
    transformer = FSDP(
        transformer,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=server_args.dit_cpu_offload),
    )
```

**内存节省**:
```
单GPU模式:
- Transformer权重: 8GB
- 总内存: ~12GB

FSDP 4-GPU模式:
- 每GPU Transformer权重: 8GB / 4 = 2GB
- 每GPU总内存: ~6GB
```

---

**文档版本**: v1.0  
**最后更新**: 2024-11  
**作者**: SGLang Diffusion Development Team
