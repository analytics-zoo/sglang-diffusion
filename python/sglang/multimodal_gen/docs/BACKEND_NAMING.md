# Backend Naming Convention

## XPU Distributed Backend: "xccl"

### 命名说明

在SGLang的Intel XPU支持中，分布式通信backend使用 **"xccl"** 作为名称，这与PyTorch的命名约定保持一致。

### 重要区分

1. **oneCCL / CCL** - Intel的通信库全称
   - 全称: One Collective Communications Library
   - 用于环境变量: `CCL_LOG_LEVEL`, `CCL_ATL_TRANSPORT`, etc.
   - 用于导入: `import oneccl_bindings_for_pytorch`
   - 这是底层通信库的名称

2. **xccl** - PyTorch中注册的backend名称
   - 用于PyTorch API: `torch.distributed.init_process_group(backend='xccl')`
   - 用于环境变量: `TORCH_DISTRIBUTED_BACKEND=xccl`
   - 这是PyTorch分布式系统中的backend标识符

### 代码中的使用

```python
# ✅ 正确 - 使用 "xccl" 作为PyTorch backend
import torch.distributed as dist
dist.init_process_group(backend='xccl', rank=rank, world_size=world_size)

# ✅ 正确 - 验证backend类型
if is_xpu():
    backend = "xccl"

# ✅ 正确 - oneCCL相关环境变量仍使用CCL
os.environ['CCL_LOG_LEVEL'] = 'info'
```

### 更新历史

- **2024**: 初始实现使用 "ccl" 作为backend名称
- **2024**: 更新为 "xccl" 以匹配PyTorch的官方命名约定

### 相关文件

Backend名称在以下文件中使用：

1. `runtime/distributed/parallel_state.py` - Backend选择逻辑
2. `runtime/distributed/device_communicators/xpu_communicator.py` - Communicator实现
3. `test/test_xpu_distributed.py` - 测试默认backend
4. `docs/XPU_DISTRIBUTED.md` - 用户文档

### 验证方法

```python
# 验证xccl backend是否可用
import torch
import torch.distributed as dist

# 检查backend注册
print('xccl' in dir(torch.distributed.Backend))  # 应输出True

# 初始化并检查backend名称
dist.init_process_group(backend='xccl', ...)
print(dist.get_backend())  # 应输出 'xccl'
```

### 常见错误

❌ **错误**: 使用 "ccl" 作为backend名称
```python
# 这会导致错误，因为PyTorch不识别 "ccl"
dist.init_process_group(backend='ccl', ...)
```

✅ **正确**: 使用 "xccl"
```python
dist.init_process_group(backend='xccl', ...)
```

### 参考资料

- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
- [oneCCL Documentation](https://oneapi-src.github.io/oneCCL/)
