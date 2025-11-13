# XPU + Gloo Backend 兼容性修复

## 问题描述

在使用 Intel XPU 时，分布式初始化失败，错误信息：

```
RuntimeError: No backend type associated with device type xpu
```

**错误堆栈：**
```python
File "parallel_state.py", line 290, in init_distributed_environment
    _WORLD = init_world_group(ranks, local_rank, backend)
File "parallel_state.py", line 1126, in init_world_group
    return GroupCoordinator(...)
File "group_coordinator.py", line 196, in __init__
    cpu_group = torch.distributed.new_group(ranks, backend="gloo")
RuntimeError: No backend type associated with device type xpu
```

## 根本原因

PyTorch 分布式通信的架构问题：

1. **全局进程组初始化时传递了 XPU device_id**
   ```python
   torch.distributed.init_process_group(
       backend="xccl",
       device_id=torch.device("xpu:0"),  # ← 问题根源
       ...
   )
   ```

2. **后续创建 Gloo CPU 组时，PyTorch 检查全局设备类型**
   ```python
   # GroupCoordinator 尝试创建 CPU 组用于跨进程协调
   cpu_group = torch.distributed.new_group(ranks, backend="gloo")
   # ↑ PyTorch 内部会检查全局进程组的设备类型
   # ↑ 发现是 XPU，但 Gloo 不支持 XPU，抛出错误
   ```

3. **PyTorch 的内部逻辑**
   - 在 `_new_process_group_helper` 中：
     ```python
     if device_id and pg._get_backend(device_id).supports_splitting:
         # ↑ 尝试获取 XPU 设备的 backend
         # ↑ Gloo backend 没有 XPU 支持，抛出异常
     ```

## 解决方案

**核心原则：对于 XPU + XCCL，在全局进程组初始化时不传递 `device_id`**

### 修复前（错误）

```python
def init_distributed_environment(...):
    extra_args = {}
    if not current_platform.is_mps() and backend not in ["gloo"]:
        # ❌ 问题：XPU + XCCL 也会传递 device_id
        extra_args = dict(device_id=device_id)
    
    torch.distributed.init_process_group(
        backend=backend,  # "xccl" for XPU
        **extra_args,     # device_id=xpu:0 被传递
    )
```

### 修复后（正确）

```python
def init_distributed_environment(...):
    extra_args = {}
    if (not current_platform.is_mps() 
        and backend not in ["gloo", "xccl"]  # ✅ 排除 XCCL
        and not current_platform.is_xpu()):  # ✅ 排除 XPU 平台
        # 只有 CUDA/ROCm + NCCL 才传递 device_id
        extra_args = dict(device_id=device_id)
    
    torch.distributed.init_process_group(
        backend=backend,  # "xccl" for XPU
        **extra_args,     # XPU 情况下为空字典 {}
    )
```

## 为什么这样修复可行

1. **XCCL backend 不需要显式的 device_id**
   - XCCL 可以通过环境变量 `LOCAL_RANK` 或进程上下文自动确定设备
   - 类似于 NCCL，但 NCCL 在某些场景需要 `device_id` 来优化

2. **避免了全局设备类型污染**
   - 不传递 `device_id` 后，全局进程组不会绑定到特定设备类型
   - 后续创建 Gloo CPU 组时，PyTorch 不会检查 XPU 设备

3. **Gloo 组可以正常创建**
   - Gloo backend 纯粹用于 CPU 通信
   - 不需要关心 XPU 设备信息

## 影响范围

### 受影响的后端组合

| Backend | Platform | device_id 是否传递 | 原因 |
|---------|----------|-------------------|------|
| NCCL | CUDA | ✅ 是 | NCCL 需要知道设备 ID 进行优化 |
| NCCL | ROCm | ✅ 是 | 同上 |
| XCCL | XPU | ❌ 否 | 避免 Gloo 组创建失败 |
| Gloo | CPU | ❌ 否 | 纯 CPU backend |
| Gloo | MPS | ❌ 否 | MPS 不支持设备索引 |

### 不受影响的功能

- ✅ XCCL 集合通信（allreduce, broadcast, etc.）仍然正常工作
- ✅ XPU 设备内存和计算操作不受影响
- ✅ 混合 XCCL（设备通信）+ Gloo（CPU 通信）架构正常
- ✅ 其他平台（CUDA、ROCm）不受影响

## 相关代码位置

### 1. `parallel_state.py` - 全局进程组初始化
```python
# 文件：/home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/runtime/distributed/parallel_state.py
# 函数：init_distributed_environment
# 行号：~250-270

extra_args = {}
if (not current_platform.is_mps() 
    and backend not in ["gloo", "xccl"]
    and not current_platform.is_xpu()):
    extra_args = dict(device_id=device_id)

torch.distributed.init_process_group(
    backend=backend,
    init_method=distributed_init_method,
    world_size=world_size,
    rank=rank,
    **extra_args,  # XPU 时为 {}
)
```

### 2. `group_coordinator.py` - Gloo CPU 组创建
```python
# 文件：/home/intel/xiangyu/study/sglang/python/sglang/multimodal_gen/runtime/distributed/group_coordinator.py
# 类：GroupCoordinator.__init__
# 行号：~196

if current_platform.is_xpu():
    try:
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
    except RuntimeError as e:
        if "No backend type associated with device type xpu" in str(e):
            logger.error("Failed to create Gloo CPU group...")
        raise
```

## 测试验证

修复后，应该能够成功：

1. ✅ 初始化 4 卡 XPU 分布式环境
2. ✅ 创建 XCCL 设备通信组
3. ✅ 创建 Gloo CPU 协调组
4. ✅ 执行 all-to-all、allreduce 等集合通信
5. ✅ 所有 ranks 完成初始化并打印日志

**验证命令：**
```bash
# 运行多卡程序，查看所有 ranks 是否成功初始化
torchrun --nproc_per_node=4 your_script.py

# 预期输出（所有 ranks）：
[Rank 0] init_process_group completed successfully
[Rank 1] init_process_group completed successfully
[Rank 2] init_process_group completed successfully
[Rank 3] init_process_group completed successfully
[Rank 0] Worker 0: Distributed environment initialized.
[Rank 1] Worker 1: Distributed environment initialized.
[Rank 2] Worker 2: Distributed environment initialized.
[Rank 3] Worker 3: Distributed environment initialized.
```

## 相关 Issue

这是一个已知的 PyTorch 限制：
- Gloo backend 不支持 GPU 设备类型（包括 XPU、MPS 等）
- PyTorch 在创建新进程组时会检查全局进程组的设备类型
- 如果全局进程组绑定了不支持的设备，创建 Gloo 组会失败

## 总结

**问题**：XPU + XCCL 初始化时传递 `device_id` 导致后续 Gloo CPU 组创建失败

**解决**：对于 XPU/XCCL，不在全局进程组初始化时传递 `device_id`

**影响**：无负面影响，XCCL 和 Gloo 都能正常工作

**验证**：所有 ranks 都能成功完成分布式初始化
