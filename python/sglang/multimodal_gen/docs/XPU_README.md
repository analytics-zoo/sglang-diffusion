# Intel XPU支持文档

SGLang Diffusion现已支持Intel XPU（Data Center GPU Max / Arc GPU）平台！

## 📖 快速导航

### 🚀 新手入门
如果您是第一次在Intel XPU上部署SGLang Diffusion，请从这里开始：

**👉 [完整文档索引](./XPU_DOCS_INDEX.md)**

### 📚 主要文档

1. **[Intel XPU 完整部署指南](./INTEL_XPU_GUIDE.md)** (34 KB)
   - 系统要求和安装步骤
   - 单卡/多卡启动方式
   - 性能调优和故障排查
   - **推荐首先阅读此文档**

2. **[Serving Pipeline 技术详解](./SERVING_PIPELINE_INTERNALS.md)** (40 KB)
   - 架构和组件详解
   - 请求处理流程
   - Pipeline Stage分析
   - 多卡分布式机制
   - **适合需要深入了解系统的开发者**

3. **[XPU 分布式通信](./XPU_DISTRIBUTED.md)** (6 KB)
   - oneCCL/XCCL配置
   - 分布式环境变量

4. **[Backend 命名规范](./BACKEND_NAMING.md)** (2.4 KB)
   - "xccl" vs "CCL" 说明

## 🎯 快速示例

### 单卡推理
```bash
# 安装依赖
pip install torch intel-extension-for-pytorch \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    
# 安装SGLang Diffusion
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python[diffusion]"

# 生成视频
sglang generate \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 1 \
    --text-encoder-cpu-offload \
    --vae-cpu-offload \
    --prompt "A curious raccoon in sunflowers" \
    --save-output
```

### 多卡服务器（4卡数据并行）
```bash
# 设置环境
export ZE_AFFINITY_MASK=0,1,2,3
export CCL_LOG_LEVEL=info
export TORCH_DISTRIBUTED_BACKEND=xccl

# 启动服务器
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --num-gpus 4 \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000 \
    --text-encoder-cpu-offload \
    --vae-cpu-offload
```

## ✅ 系统要求

- **硬件**: Intel Data Center GPU Max / Arc GPU
- **驱动**: Intel GPU驱动 (Level Zero)
- **软件**: PyTorch 2.8+, IPEX 2.8+, oneCCL
- **操作系统**: Ubuntu 22.04+ / CentOS 8+

## 🔧 特性支持

| 特性 | 状态 | 说明 |
|------|------|------|
| 单卡推理 | ✅ 已支持 | 完整功能 |
| 数据并行（DP） | ✅ 已支持 | 使用XCCL backend |
| 序列并行（SP） | ✅ 已支持 | Ulysses + Ring Attention |
| 张量并行（TP） | ✅ 已支持 | 模型分片 |
| CFG并行 | ✅ 已支持 | 条件/无条件并行 |
| CPU Offload | ✅ 已支持 | Text/Image Encoder, VAE |
| FSDP推理 | ✅ 已支持 | 权重分片 |
| Flash Attention | ⚠️ 部分支持 | 使用Torch SDPA（FA3需验证） |
| IPEX优化 | ✅ 已支持 | BF16, 优化算子 |

## 📊 性能参考

*实际性能取决于具体硬件配置、模型大小、并行策略等因素。请在您的环境中进行基准测试以获得准确的性能数据。*

## 🆘 获取帮助

### 常见问题
详见 [故障排查章节](./INTEL_XPU_GUIDE.md#7-故障排查)

### 问题反馈
- GitHub Issues: https://github.com/sgl-project/sglang/issues
- 请附带完整错误日志和环境信息

### 更多资源
- [Intel IPEX文档](https://intel.github.io/intel-extension-for-pytorch/)
- [oneCCL文档](https://oneapi-src.github.io/oneCCL/)
- [Intel GPU驱动](https://dgpu-docs.intel.com/)

---

**文档维护**: SGLang Diffusion Team  
**最后更新**: 2024-11-11
