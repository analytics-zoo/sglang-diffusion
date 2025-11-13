# 分布式初始化死锁问题诊断和修复

## 问题描述

在启动多卡时，只有 worker 0 能成功完成分布式环境初始化并打印 "Worker 0: Distributed environment initialized."，而其他 workers (1-3) 卡在 `maybe_init_distributed_environment_and_model_parallel` 阶段。

## 根本原因分析

### 1. 硬编码的端口冲突（已修复 ✅）

**问题：**
- 在 `parallel_state.py` 的 `init_distributed_environment` 函数中，`init_method` 被硬编码为 `"tcp://localhost:29500"`
- 但在 `gpu_worker.py` 中，通过环境变量 `MASTER_PORT` 设置了不同的端口
- 这导致不同的 workers 可能尝试连接到不同的端点

**修复：**
```python
# 修复前（错误）:
torch.distributed.init_process_group(
    backend=backend,
    init_method=f"tcp://localhost:29500",  # 硬编码端口
    world_size=world_size,
    rank=rank,
)

# 修复后（正确）:
torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,  # 使用参数（默认 "env://"）
    world_size=world_size,
    rank=rank,
    **extra_args,  # 条件性传递 extra_args
)
```

### 2. XPU + XCCL backend 的 device_id 问题（已修复 ✅）

**问题：**
- 当使用 XCCL backend 初始化全局进程组时，如果传递了 `device_id`（XPU 设备）
- 后续创建 Gloo CPU 组时，PyTorch 会检查全局进程组的设备类型
- Gloo backend 不支持 XPU 设备类型，导致错误：`RuntimeError: No backend type associated with device type xpu`

**错误堆栈：**
```python
File "group_coordinator.py", line 196, in __init__
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
RuntimeError: No backend type associated with device type xpu
```

**修复：**
```python
# 修复前（错误）:
extra_args = {}
if not current_platform.is_mps() and backend not in ["gloo"]:
    extra_args = dict(device_id=device_id)  # XPU 也会传递 device_id

# 修复后（正确）:
extra_args = {}
if (not current_platform.is_mps() 
    and backend not in ["gloo", "xccl"]
    and not current_platform.is_xpu()):
    # 只有 CUDA/ROCm + NCCL 才传递 device_id
    # XPU + XCCL 不传递，避免后续 Gloo 组创建失败
    extra_args = dict(device_id=device_id)
```

**原因分析：**
1. PyTorch 的 `init_process_group` 在传递 `device_id` 后，会将设备信息保存到全局状态
2. 后续调用 `new_group()` 创建新的进程组时，即使指定了不同的 backend（如 gloo），PyTorch 仍会检查全局进程组的设备类型
3. Gloo backend 只支持 CPU，不支持 XPU，因此会抛出错误
4. 解决方案：对于 XPU + XCCL，不在全局进程组初始化时传递 `device_id`

### 2. 集合操作同步问题

**潜在问题：**
PyTorch 分布式的集合操作（collective operations）是同步的，必须所有进程都调用才能完成：

- `torch.distributed.new_group()` - 创建进程组
- `torch.distributed.barrier()` - 屏障同步
- `torch.distributed.broadcast()` - 广播
- `torch.distributed.all_reduce()` - 全归约
- `set_seq_parallel_pg()` - yunchang 库的序列并行设置（可能包含集合操作）

**如果某个 rank 没有到达集合操作调用点，所有其他 ranks 会永久等待（死锁）**

### 3. GroupCoordinator 初始化的循环问题

在 `group_coordinator.py` 的 `GroupCoordinator.__init__` 中：

```python
for ranks in group_ranks:
    device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
    
    if self.rank in ranks:
        self.ranks = ranks
        self.world_size = len(ranks)
        # ...
```

**关键点：**
- `new_group()` 必须被所有进程调用，即使某个进程不在该组中
- 循环会为 `group_ranks` 中的每个子列表创建一个进程组
- 所有进程必须以相同的顺序调用相同次数的 `new_group()`

## 已添加的调试日志

为了定位死锁的确切位置，已在以下位置添加了详细日志：

### 1. `maybe_init_distributed_environment_and_model_parallel`
```python
logger.info(f"[Rank {rank}] Before init_distributed_environment")
logger.info(f"[Rank {rank}] After init_distributed_environment")
logger.info(f"[Rank {rank}] Before initialize_model_parallel")
logger.info(f"[Rank {rank}] After initialize_model_parallel")
```

### 2. `init_distributed_environment`
```python
logger.info(f"[Rank {rank}] Calling init_process_group...")
logger.info(f"[Rank {rank}] init_process_group completed successfully")
```

### 3. `initialize_model_parallel`
```python
logger.info(f"[Rank {rank}] Starting model parallel initialization")
logger.info(f"[Rank {rank}] Creating DP group")
logger.info(f"[Rank {rank}] DP group created")
logger.info(f"[Rank {rank}] Creating CFG group")
logger.info(f"[Rank {rank}] CFG group created")
logger.info(f"[Rank {rank}] Creating PP group")
logger.info(f"[Rank {rank}] PP group created")
logger.info(f"[Rank {rank}] Before set_seq_parallel_pg")
logger.info(f"[Rank {rank}] After set_seq_parallel_pg")
logger.info(f"[Rank {rank}] After creating SP group coordinator")
```

## 调试步骤

### 1. 查看所有 ranks 的日志

运行您的多卡程序，查看每个 rank 打印的最后一条日志：

```bash
# 示例：如果卡在 set_seq_parallel_pg
[Rank 0] Before set_seq_parallel_pg  # Rank 0 到达这里
[Rank 0] After set_seq_parallel_pg   # Rank 0 通过了
[Rank 1] Before set_seq_parallel_pg  # Rank 1 卡在这里（没有 After 日志）
[Rank 2] Before set_seq_parallel_pg  # Rank 2 卡在这里
[Rank 3] Before set_seq_parallel_pg  # Rank 3 卡在这里
```

### 2. 确认环境变量设置

确保所有 workers 都有正确的环境变量：

```python
# 在 gpu_worker.py 中检查
logger.info(f"Worker {rank} environment: "
           f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
           f"MASTER_PORT={os.environ.get('MASTER_PORT')}, "
           f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
           f"RANK={os.environ.get('RANK')}, "
           f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
```

### 3. 检查进程组创建顺序

确保所有 ranks 以相同顺序创建进程组：

```python
# RankGenerator 应该为所有 ranks 生成相同的 group_ranks 结构
rank_generator = RankGenerator(...)
logger.info(f"[Rank {rank}] DP groups: {rank_generator.get_ranks('dp')}")
logger.info(f"[Rank {rank}] CFG groups: {rank_generator.get_ranks('cfg')}")
logger.info(f"[Rank {rank}] PP groups: {rank_generator.get_ranks('pp')}")
logger.info(f"[Rank {rank}] SP groups: {rank_generator.get_ranks('sp')}")
```

## 可能的解决方案

### 方案 1：使用环境变量初始化（推荐）

确保使用 `"env://"` 初始化方法，并正确设置环境变量：

```python
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(master_port)
os.environ["WORLD_SIZE"] = str(num_gpus)
os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(local_rank)

# 使用 env:// 初始化
maybe_init_distributed_environment_and_model_parallel(
    distributed_init_method="env://",
    ...
)
```

### 方案 2：添加同步点

在关键位置添加显式屏障：

```python
if torch.distributed.is_initialized():
    logger.info(f"[Rank {rank}] Barrier before critical section")
    torch.distributed.barrier()
    logger.info(f"[Rank {rank}] Passed barrier")
```

### 方案 3：检查 yunchang 库的兼容性

如果卡在 `set_seq_parallel_pg`，可能是 yunchang 库的问题：

```python
# 临时注释掉 set_seq_parallel_pg 调用，看是否能通过初始化
# set_seq_parallel_pg(...)
```

### 方案 4：逐步初始化

将初始化拆分成多个阶段，每个阶段后添加同步：

```python
# 阶段 1: 初始化基础分布式环境
init_distributed_environment(...)
torch.distributed.barrier()

# 阶段 2: 创建 DP 组
_DP = init_parallel_group_coordinator(...)
torch.distributed.barrier()

# 阶段 3: 创建 CFG 组
_CFG = init_parallel_group_coordinator(...)
torch.distributed.barrier()

# ... 依此类推
```

## XPU 特定问题

对于 Intel XPU，还需要注意：

1. **XCCL backend 支持**
   - 确保 `CCL_ATL_TRANSPORT` 和其他 CCL 环境变量正确设置
   - XCCL 可能不支持所有集合操作

2. **Gloo backend 与 XPU 设备关联**
   - 创建 Gloo CPU 组时不要传递 XPU device_id
   - 已在 `group_coordinator.py` 中添加了 XPU 特定的错误处理

3. **设备同步**
   - XPU 使用 `torch.xpu.synchronize()`
   - 确保在正确的位置调用

## 下一步

1. **运行带日志的版本**，查看具体卡在哪个步骤
2. **比较所有 ranks 的日志输出**，找出差异
3. **根据卡住的位置**应用相应的解决方案
4. **如果是 yunchang 库的问题**，考虑提交 issue 或寻找替代方案

## 相关文件

- `/home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py`
- `/home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py`
- `/home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py`
