# Intel XPU Distributed Communication Support

本文档介绍如何在SGLang Diffusion中使用Intel XPU的分布式通信功能。

## 概述

SGLang Diffusion现在支持Intel XPU (GPU)设备的分布式训练和推理，使用oneCCL (Collective Communications Library)作为通信后端（在PyTorch中注册为`xccl` backend）。

## 系统要求

### 硬件要求
- Intel Data Center GPU Max系列 (Ponte Vecchio)
- Intel Arc GPU系列 (可选，用于开发测试)

### 软件要求
- PyTorch 2.8.0+ (with XPU support)
- Intel Extension for PyTorch (IPEX) 2.8.0+
- oneCCL (通常随IPEX一起安装)
- Python 3.10+

## 安装

### 1. 安装Intel XPU驱动

```bash
# 安装Intel GPU驱动和运行时
# 详细步骤请参考: https://dgpu-docs.intel.com/
```

### 2. 安装PyTorch with XPU支持

```bash
# 使用Intel提供的wheel包
pip install torch torchvision intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 3. 验证XPU可用性

```python
import torch
print(f"XPU available: {torch.xpu.is_available()}")
print(f"XPU device count: {torch.xpu.device_count()}")
```

## 使用方法

### 单GPU推理

```python
import torch
from sglang.multimodal_gen import ...

# 自动检测并使用XPU
# 无需额外配置
```

### 多GPU分布式推理

#### 使用torchrun启动

```bash
# 2个GPU
torchrun --nproc_per_node=2 your_script.py

# 4个GPU
torchrun --nproc_per_node=4 your_script.py

# 多节点（例如2个节点，每个节点4个GPU）
# 节点0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<MASTER_IP> --master_port=29500 your_script.py

# 节点1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<MASTER_IP> --master_port=29500 your_script.py
```

#### 代码示例

```python
import os
import torch
import torch.distributed as dist

def main():
    # 初始化分布式环境（使用XCCL backend）
    if not dist.is_initialized():
        dist.init_process_group(backend='xccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 设置当前设备
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.xpu.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size} using XPU:{local_rank}")
    
    # 你的模型和推理代码
    # SGLang会自动使用XPU communicator
    
    # 清理
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

## 环境变量

### XPU设备控制

```bash
# 控制可见的XPU设备（类似CUDA_VISIBLE_DEVICES）
export ZE_AFFINITY_MASK=0,1  # 只使用GPU 0和1

# 或使用设备索引
export ZE_AFFINITY_MASK=0
```

### oneCCL配置

```bash
# 设置CCL日志级别
export CCL_LOG_LEVEL=info  # trace, debug, info, warn, error, fatal

# 启用CCL性能分析
export CCL_ITT_LEVEL=1

# 设置CCL通信后端
export CCL_ATL_TRANSPORT=mpi  # 或 ofi (libfabric)
```

### 分布式配置

```bash
# PyTorch分布式backend (使用xccl for XPU)
export TORCH_DISTRIBUTED_BACKEND=xccl

# 主节点地址（多节点时需要）
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 当前节点信息
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
```

## 测试

运行测试脚本验证XPU分布式功能：

```bash
# 单GPU测试
python test/test_xpu_distributed.py

# 多GPU测试（2个GPU）
torchrun --nproc_per_node=2 test/test_xpu_distributed.py

# 使用gloo backend（fallback）
torchrun --nproc_per_node=2 test/test_xpu_distributed.py --backend gloo
```

测试包括：
- ✓ All-Reduce操作
- ✓ Broadcast操作
- ✓ All-Gather操作
- ✓ Send/Recv点对点通信
- ✓ Barrier同步

## 性能优化

### 1. 内存优化

```bash
# 启用内存池
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# 设置最大内存使用
export ZE_AFFINITY_MASK=0
```

### 2. 通信优化

```bash
# 启用CCL优化
export CCL_FUSION=1
export CCL_FUSION_BYTES_THRESHOLD=16384
export CCL_FUSION_COUNT_THRESHOLD=256

# 设置优先级模式
export CCL_PRIORITY=direct
```

### 3. 计算优化

```python
# 使用混合精度训练
import intel_extension_for_pytorch as ipex

model = model.to('xpu')
model = ipex.optimize(model, dtype=torch.bfloat16)
```

## 已知限制

1. **Attention Backend**: XPU目前只支持Torch SDPA，不支持Flash Attention等CUDA特定优化
2. **Custom Kernels**: CUDA编译的自定义kernel（如VMoBA）不可用，使用fallback实现
3. **CUDA Graphs**: XPU不支持CUDA graphs
4. **Triton Kernels**: 需要验证Triton对XPU的支持情况

## 故障排查

### 问题: "CCL not available"

```bash
# 检查oneCCL是否安装
python -c "import oneccl_bindings_for_pytorch; print('CCL available')"

# 如果未安装，安装oneCCL
pip install oneccl_bind_pt -f https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# 验证xccl backend可用
python -c "import torch; print('xccl' in dir(torch.distributed.Backend))"
```

### 问题: "No XPU devices found"

```bash
# 检查驱动
xpu-smi discovery

# 检查设备
python -c "import torch; print(torch.xpu.device_count())"
```

### 问题: 分布式初始化失败

```bash
# 使用gloo backend作为fallback
export TORCH_DISTRIBUTED_BACKEND=gloo

# 增加超时时间
export TORCH_DISTRIBUTED_TIMEOUT=1800
```

### 问题: 性能不佳

```bash
# 检查是否使用了正确的backend
python -c "import torch.distributed as dist; dist.init_process_group('xccl'); print(dist.get_backend())"

# 启用性能分析
export CCL_LOG_LEVEL=info
export IPEX_VERBOSE=1
```

## 参考资料

- [Intel Extension for PyTorch文档](https://intel.github.io/intel-extension-for-pytorch/)
- [oneCCL文档](https://oneapi-src.github.io/oneCCL/)
- [Intel GPU驱动安装指南](https://dgpu-docs.intel.com/)
- [PyTorch XPU Backend](https://pytorch.org/docs/stable/notes/xpu.html)

## 支持

如遇到问题，请：
1. 查看日志输出（使用`CCL_LOG_LEVEL=debug`）
2. 验证环境配置（运行test_xpu_distributed.py）
3. 提交Issue并附上详细的错误信息和环境配置
