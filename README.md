# CUDA Study

这个仓库用于记录 CUDA 编程学习过程。目前主要包含向量加法和矩阵乘法示例，并逐步使用 NVIDIA profiling 工具分析性能瓶颈。README 初稿由 GPT 辅助整理，内容根据个人实验结果持续更新。

## Environment

当前开发环境：

- OS: WSL2
- CUDA: 12.4
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compiler: `nvcc`
- Profiling tools:
  - Nsight Systems
  - Nsight Compute

可以用以下命令检查环境：

```bash
nvcc --version
nvidia-smi
nsys --version
ncu --version
```

## Files

```text
.
├── Makefile
├── vector_add.cu
├── matrix_product.cu
└── README.md
```

文件说明：

- `vector_add.cu`: CUDA 向量加法入门示例。目前存在一个索引相关 bug，因为测试数据全部相同，暂时没有暴露错误。
- `matrix_product.cu`: CUDA 矩阵乘法示例
- `Makefile`: 编译 `matrix_product.cu`
- `.gitignore`: 忽略编译产物、profiling 报告和本地临时文件

## Matrix Multiplication

当前 `matrix_product.cu` 实现的是朴素版本矩阵乘法。

矩阵尺寸：

```cpp
const int N = 4096;
const int M1 = 4096;
const int M2 = 4096;
```

矩阵形状：

```text
A: N x M1
B: M1 x M2
C: N x M2
```

kernel 设计：

- 每个 CUDA thread 计算 `C` 中的一个元素
- 每个 block 使用 `16 x 16` 个 thread
- 每个 thread 沿着 `M1` 方向做一次 dot product
- 当前版本直接从 global memory 读取 `A` 和 `B`

核心计算逻辑：

```cpp
for (int i = 0; i < m1; ++i) {
    C[idy * m2 + idx] += A[idy * m1 + i] * B[i * m2 + idx];
}
```

## Build

使用 Makefile 编译：

```bash
make
```

或者手动编译，并保留 profiling 需要的源码行号信息：

```bash
nvcc -O3 -lineinfo matrix_product.cu -o matrix_product
```

## Run

运行矩阵乘法：

```bash
./matrix_product
```

正确输出：

```text
OK! Correct!
```

## Profiling

### Nsight Systems

Nsight Systems 用于观察程序整体时间线，包括 CUDA API、kernel、memcpy 和 CPU/GPU 活动。

```bash
nsys profile --stats=true --force-overwrite=true -o naive_nsys ./matrix_product
```

查看统计结果：

```bash
nsys stats naive_nsys.nsys-rep
```

重点区分：

- `CUDA API`: CPU 侧调用 CUDA API 的耗时
- `CUDA HW -> Kernel`: GPU 上 kernel 的真实执行时间
- `CUDA HW -> Memory`: GPU 上 memcpy 的真实传输活动

`cudaMemcpy` 是同步 API，尤其是 device-to-host copy，可能会等待前面的 kernel 完成。因此 `CUDA API` 里的 `cudaMemcpy` 时间不一定等于纯拷贝时间。

### Nsight Compute

Nsight Compute 用于分析单个 kernel 的详细性能瓶颈。

```bash
ncu --set full --kernel-name matrixProduct --force-overwrite -o naive_ncu_full ./matrix_product
```

重点关注：

- `Duration`
- `Compute Throughput`
- `Memory Throughput`
- `DRAM Throughput`
- `Memory Workload Analysis`
- `Warp State Statistics`
- `Stall Long Scoreboard`
- `Source Counters`

当前朴素版本观察到的现象：

```text
Memory Throughput 高于 Compute Throughput
DRAM Throughput 不高
Warp 主要 stall 在 L1TEX memory dependency
```

这说明当前 kernel 的主要问题不是单纯打满显存带宽，而是 global memory/cache 路径上的数据依赖导致 warp 等待。

## Current Baseline

朴素版本 baseline：

```text
Kernel: matrixProduct
Block size: 16 x 16
Grid size: 256 x 256
Matrix size: 4096 x 4096

Nsight Compute:
Duration: 690.87 ms
Compute Throughput: 51.53%
Memory Throughput: 64.78%
DRAM Throughput: 11.12%
Stall reason:
  L1TEX scoreboard dependency accounts for most warp stall cycles.
```

以上数据可能会随 GPU 频率、系统负载、编译参数和 profiling 设置变化。

## Next Step

下一步计划实现 shared memory tiled matrix multiplication。

目标：

- 使用 shared memory 缓存 `A` 和 `B` 的 tile
- 减少重复 global memory load
- 降低 L1TEX memory dependency stall
- 对比朴素版本和 shared memory 版本的 kernel duration

计划使用的 tile size：

```text
16 x 16
```

优化后重点比较：

```text
Duration
Compute Throughput
Memory Throughput
Stall Long Scoreboard
Shared Memory Load/Store
```

## Git Workflow

建议每完成一个清晰阶段就提交一次。

查看状态：

```bash
git status
git status --short
```

查看修改内容：

```bash
git diff
```

查看暂存区内容：

```bash
git diff --cached
```

添加文件：

```bash
git add README.md
```

提交版本：

```bash
git commit -m "Add initial README"
```

查看提交历史：

```bash
git log --oneline
```

推荐提交节奏：

```text
1. Add vector addition example
2. Add naive matrix multiplication kernel
3. Add profiling commands and baseline results
4. Add shared memory tiled matrix multiplication
5. Compare naive and shared memory performance
```

## Notes

当前仓库忽略以下文件：

```text
matrix_product
vector_add
*.o
*.nsys-rep
*.ncu-rep
*.sqlite
*.ptx
*.cubin
*.fatbin
.codex
```

这些文件通常是编译产物、profiling 报告或本地临时文件，不适合提交到 Git。
