# Nsight Systems Profiling Guide

本文档说明如何使用Nsight Systems对benchmark脚本进行性能分析。

## 什么是Nsight Systems？

NVIDIA Nsight Systems是一个系统级性能分析工具，可以可视化应用程序在CPU和GPU上的活动，帮助识别性能瓶颈。

## 前置要求

### 1. 安装Nsight Systems

**Linux/Mac**:
```bash
# 下载并安装 NVIDIA Nsight Systems
# https://developer.nvidia.com/nsight-systems
```

**验证安装**:
```bash
nsys --version
```

### 2. GPU环境（推荐）

NVTX标记在GPU上最有用，但CPU环境也可以使用。

## 使用方法

### 基础用法

#### 1. 不带profiling运行（正常benchmark）
```bash
uv run python cs336_systems/benchmark.py \
  --device cpu \
  --num-iterations 5
```

#### 2. 带NVTX标记运行（用于Nsight profiling）
```bash
# 启用NVTX标记
uv run python cs336_systems/benchmark.py \
  --device cuda \
  --enable-profiling \
  --num-iterations 5
```

#### 3. 使用Nsight Systems捕获profile
```bash
nsys profile \
  -o benchmark_profile \
  --trace=cuda,nvtx \
  uv run python cs336_systems/benchmark.py \
    --device cuda \
    --enable-profiling \
    --num-iterations 5
```

参数说明：
- `-o benchmark_profile`: 输出文件名
- `--trace=cuda,nvtx`: 追踪CUDA和NVTX事件
- `--enable-profiling`: 启用benchmark脚本中的NVTX标记

### 高级选项

#### 只采样特定迭代

```bash
# 只profile前3次迭代
nsys profile \
  -o benchmark_profile \
  --trace=cuda,nvtx \
  --duration=30 \
  uv run python cs336_systems/benchmark.py \
    --device cuda \
    --enable-profiling \
    --num-iterations 3
```

#### 更详细的追踪

```bash
nsys profile \
  -o benchmark_profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  uv run python cs336_systems/benchmark.py \
    --device cuda \
    --enable-profiling
```

## 查看Profile结果

### 方法1: Nsight Systems GUI

```bash
# 打开profile结果
nsys-ui benchmark_profile.nsys-rep
```

### 方法2: 命令行导出

```bash
# 导出统计信息
nsys stats benchmark_profile.nsys-rep

# 导出为CSV
nsys export benchmark_profile.nsys-rep \
  --type=csv \
  --output=benchmark_stats
```

## NVTX标记说明

benchmark脚本中添加了以下NVTX标记：

### 模型初始化
- `model_initialization`: 模型创建和移动到设备

### 前向传播
- `warmup_forward`: 预热迭代
- `forward_iter_{i}`: 每次前向传播迭代

### 训练步骤（前向+反向）
- `warmup_train_step`: 预热训练步骤
- `train_iter_{i}`: 每次训练迭代
  - `zero_grad_{i}`: 清零梯度
  - `forward_{i}`: 前向传播
  - `backward_{i}`: 反向传播
  - `optimizer_step_{i}`: 优化器更新

## 分析示例

### 1. 查找性能瓶颈

在Nsight Systems GUI中：
1. 打开 `benchmark_profile.nsys-rep`
2. 查看 "NVTX" 行
3. 找到耗时最长的标记
4. 下钻到CUDA kernels查看具体操作

### 2. 分析内存使用

```bash
nsys stats benchmark_profile.nsys-rep \
  --report cuda_gpu_kern_sum \
  --format csv
```

### 3. 识别GPU利用率

在GUI中查看：
- GPU利用率时间线
- Kernel执行overlap
- Memory transfer patterns

## 常见问题

### Q1: 命令找不到 `nsys`
**A**: 确保已安装Nsight Systems并添加到PATH。

### Q2: CPU上可以使用NVTX吗？
**A**: 可以运行，但NVTX标记主要用于GPU分析。CPU上建议使用其他profiler如`py-spy`。

### Q3: Profile文件太大
**A**: 减少`--num-iterations`或使用`--duration`限制采样时间。

### Q4: 看不到NVTX标记
**A**: 确保：
- 使用了`--enable-profiling`参数
- `nsys`使用了`--trace=nvtx`选项
- 在CUDA设备上运行

## 性能优化建议

根据Nsight profile结果，可以优化：

1. **Kernel融合**: 合并小的CUDA操作
2. **内存传输**: 减少CPU-GPU数据传输
3. **并发执行**: 利用CUDA streams
4. **批处理**: 增加batch size提高GPU利用率

## 完整示例

```bash
# 1. 运行带profiling的benchmark（小配置用于快速测试）
nsys profile \
  -o small_model_profile \
  --trace=cuda,nvtx \
  uv run python cs336_systems/benchmark.py \
    --device cuda \
    --enable-profiling \
    --d-model 256 \
    --num-layers 2 \
    --num-iterations 3 \
    --batch-size 4

# 2. 查看profile
nsys-ui small_model_profile.nsys-rep

# 3. 导出统计
nsys stats small_model_profile.nsys-rep

# 4. 对比不同配置
nsys profile -o large_model_profile --trace=cuda,nvtx \
  uv run python cs336_systems/benchmark.py \
    --device cuda \
    --enable-profiling \
    --d-model 512 \
    --num-layers 6 \
    --num-iterations 3
```

## 更多资源

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [NVTX Documentation](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx)
- [PyTorch Profiling Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
