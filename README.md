# CUDA 学习记录

这个仓库用于记录 CUDA 编程学习过程。目前主要包含向量加法和矩阵乘法示例，并逐步使用 NVIDIA profiling 工具分析性能瓶颈。README 初稿由 GPT 辅助整理，内容会根据个人实验结果持续更新。

## 环境

当前开发环境：

- 操作系统: WSL2
- CUDA: 12.4
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- 编译器: `nvcc`
- 分析工具:
  - Nsight Systems
  - Nsight Compute

可以用以下命令检查环境：

```bash
nvcc --version
nvidia-smi
nsys --version
ncu --version
```

## 文件

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

## 矩阵乘法

当前 [matrix_product.cu](/home/ngsxz/cuda_study/matrix_product.cu) 实现的是朴素版本矩阵乘法。

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

## 编译

使用 Makefile 编译：

```bash
make
```

或者手动编译，并保留 profiling 需要的源码行号信息：

```bash
nvcc -O3 -lineinfo matrix_product.cu -o matrix_product
```

## 运行

运行矩阵乘法：

```bash
./matrix_product
```

正确输出：

```text
OK! Correct!
```

## 性能分析

### Nsight Systems 时间线分析

Nsight Systems 用于观察程序整体时间线，包括 CUDA API、kernel、memcpy 以及 CPU/GPU 活动。

```bash
nsys profile --stats=true --force-overwrite=true -o naive_nsys ./matrix_product
```

查看统计结果：

```bash
nsys stats naive_nsys.nsys-rep
```

重点关注：

- `CUDA API`: CPU 侧调用 CUDA API 的耗时
- `CUDA HW -> Kernel`: GPU 上 kernel 的真实执行时间
- `CUDA HW -> Memory`: GPU 上 memcpy 的真实传输活动

`cudaMemcpy` 是同步 API，尤其是 device-to-host copy，可能会等待前面的 kernel 完成。因此 `CUDA API` 里的 `cudaMemcpy` 时间不一定等于纯拷贝时间。

### Nsight Compute 内核分析

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

## 当前基线

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

## 下一步

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

## Git 工作流

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

## 备注

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

## 学习记录模板

下面这段只用于个人学习记录，不是课程报告格式。现在已经先填入了这次 `naive vs shared` 的结果，后续你可以继续往下追加。

### 这次改了什么

- 日期: `2026-04-15`
- 版本: `naive vs shared`
- 主要改动:
  - `把 global memory 版本改成 shared memory tiled 版本`
  - `A/B 初始化改为和下标相关，并使用 double 做正确性验证`
- 正确性结果: `OK! Correct!`

### 这次怎么测的

- Build:
  ```bash
  nvcc -O3 -lineinfo matrix_product_naive.cu -o mm_naive
  nvcc -O3 -lineinfo matrix_product.cu -o mm_shared
  ```
- Run:
  ```bash
  ./mm_naive
  ./mm_shared
  ```
- Profiling:
  ```bash
  ncu --set basic --kernel-name matrixProduct --force-overwrite -o naive_ncu ./mm_naive
  ncu --set basic --kernel-name matrixProduct --force-overwrite -o shared_ncu ./mm_shared
  ```

### 这次我关心的数据

| 版本 | Kernel Duration (ms) | Compute Throughput (%) | Memory Throughput (%) | DRAM Throughput (%) | Stall Long Scoreboard (%) | 结果 |
|---|---:|---:|---:|---:|---:|---|
| Naive | 992.30 | 93.49 | 77.45 | 19.00 | 66.6 | OK |
| Shared | 951.22 | 97.55 | 21.73 | 19.67 | 不再是主导瓶颈 | OK |

### 这次的结论

- 最大收获:
  - `Kernel duration 从 992.30 ms 降到 951.22 ms，提升约 4.1%`
- 当前主要瓶颈:
  - `naive 版本主要卡在 Long Scoreboard Stall；shared 版本中这个问题不再主导，新的瓶颈转向 Tex Throttle 和 Short Scoreboard`
- 关于 FP64 的额外观察:
  - `Nsight Compute 提示这张 RTX 4060 Laptop GPU 的 fp32:fp64 峰值比约为 64:1。当前 double 版本的 Compute Throughput 已经很高，说明双精度算力本身也在明显限制性能，这会压缩 shared memory 优化带来的收益。`
- 下一步可以尝试:
  - `调 tile size 到 32x8 或 32x16`
  - `检查 shared memory bank conflicts 和 Source Counters`
  - `调整测试数据，在保证测试强度的同时尝试让 float 版本也能稳定通过正确性校验`

### 公式备忘

```text
Speedup = T_naive / T_shared
Improvement(%) = (T_naive - T_shared) / T_naive * 100%
Relative Error = abs((C - Ref) / Ref)
```

本次实验：

```text
Speedup = 992.30 / 951.22 = 1.043
Improvement(%) = (992.30 - 951.22) / 992.30 * 100% = 4.14%
```

### 测试数据设计过程

这次为了比较 `naive` 和 `shared` 版本的性能，我专门调整了矩阵初始化方式。最开始使用的是更强的 `double` 测试数据：

```text
A[i][j] = i + 1
B[i][j] = j + 1
```

这种写法的优点是正确性检验能力很强，因为输出矩阵中几乎每个位置的理论值都不同，索引错位、tile 搬运错位和写回错误都更容易暴露。但它的缺点也很明显：在 RTX 4060 Laptop GPU 这类消费级显卡上，`double` 性能会明显受到 FP64 吞吐限制，shared memory 的收益容易被压缩；同时如果直接切回 `float`，数值范围又太大，误差会明显上升。

为了避免每次校验都在 CPU 上重新做一遍 `O(N^3)` 的矩阵乘法，我后来坚持了一个原则：

```text
A 只依赖 i
B 只依赖 j
```

这样理论值就始终可以直接写成：

```text
C[i][j] = M1 * f(i) * g(j)
```

最后采用的版本是：

```text
A[i][j] = (i % 13) / 10 + 0.1
B[i][j] = (j % 17) / 10 + 0.1
```

这个版本的考虑有几层：

- `float` 下数值范围比较温和，不容易因为累加 4096 次就直接炸精度
- `13 x 17 = 221` 种输出模式，明显比之前那种大块分段常数更有辨识度
- `13` 和 `17` 都不和 `blockSize = 16` 对齐，尽量避免某些 tile 边界恰好掩盖错误
- 理论值仍然能闭式计算，不需要 CPU 再跑一遍完整矩阵乘法

这版实测结果是：

```text
OK! Correct!
Max relative error = 5.60922e-05
obtained it when i = 0, j = 13
```

最大相对误差没有出现在数值最大的位置，这说明这里的误差主要还是由 `float` 对十进制小数的表示误差，以及 4096 次累加产生的舍入误差共同决定，而不是简单由输出值大小决定。当前 `eps = 1e-4` 可以稳定通过，因此这组数据已经比较适合作为后续 `float` 版本的 profiling 输入。

### float 测试数据版本

在将测试数据调整为更适合 `float` 的输入后，我重新对 `naive` 和 `shared` 版本做了一轮 profiling。这个版本使用的是：

```text
A[i][j] = (i % 13) / 10 + 0.1
B[i][j] = (j % 17) / 10 + 0.1
```

它的优点是：

- 理论值仍然可以直接写成闭式公式，不需要 CPU 再做一遍 `O(N^3)` 矩阵乘法
- 测试数据比大块分段常数更有辨识度
- `float` 下可以稳定通过正确性校验

这组数据下，两份程序的输出都是：

```text
OK! Correct!
Max relative error is 5.60922e-05
Obtain it when i = 0 and j = 13
```

对应的 Nsight Compute 结果如下：

| 版本 | Kernel Duration (ms) | Compute Throughput (%) | Memory Throughput (%) | DRAM Throughput (%) | 结果 |
|---|---:|---:|---:|---:|---|
| Naive | 661.11 | 52.59 | 61.13 | 11.37 | OK |
| Shared | 178.04 | 97.66 | 97.66 | 39.09 | OK |

这一轮的加速比为：

```text
Speedup = 661.11 / 178.04 = 3.71
Improvement(%) = (661.11 - 178.04) / 661.11 * 100% = 73.07%
```

这说明在当前 `float` 测试数据下，shared memory 优化的收益非常明显。相比 `naive` 版本，`shared` 版本不只是 kernel 时间大幅下降，而且 compute 与 memory 两侧的吞吐率都接近打满，说明这次优化真正把硬件资源利用起来了。

### block shape 参数扫描

在确认 `float` 测试数据可用之后，我继续对不同 block shape 做了参数扫描。为了保证对比公平，这一轮统一使用：

```bash
nvcc -O3 -lineinfo ...
```

并对每个版本做三联重测，最后使用中位数作为记录结果。

参与对比的版本包括：

- `Naive`
- `Shared 16x16`
- `Shared 32x8`
- `Shared 32x16`
- `Shared 32x32`

整理后的中位数结果如下：

| 版本 | Kernel Duration (ms) | Compute Throughput (%) | Memory Throughput (%) | DRAM Throughput (%) |
|---|---:|---:|---:|---:|
| Naive | 700.49 | 51.86 | 64.69 | 11.02 |
| Shared 16x16 | 190.96 | 95.45 | 95.45 | 38.33 |
| Shared 32x8 | 216.64 | 96.49 | 96.49 | 66.80 |
| Shared 32x16 | 205.42 | 89.69 | 89.69 | 36.01 |
| Shared 32x32 | 220.03 | 75.38 | 75.38 | 16.52 |

对应的加速比如下：

```text
Shared 16x16 vs Naive:
Speedup = 700.49 / 190.96 = 3.67
Improvement(%) = (700.49 - 190.96) / 700.49 * 100% = 72.74%

Shared 32x8 vs Naive:
Speedup = 700.49 / 216.64 = 3.23
Improvement(%) = (700.49 - 216.64) / 700.49 * 100% = 69.07%

Shared 32x16 vs Naive:
Speedup = 700.49 / 205.42 = 3.41
Improvement(%) = (700.49 - 205.42) / 700.49 * 100% = 70.67%

Shared 32x32 vs Naive:
Speedup = 700.49 / 220.03 = 3.18
Improvement(%) = (700.49 - 220.03) / 700.49 * 100% = 68.59%
```

这轮扫描可以得出几个比较明确的结论：

- 当前最优版本仍然是 `Shared 16x16`
- `Shared 32x16` 比 `32x8` 和 `32x32` 更均衡，但依然没有超过 `16x16`
- `Shared 32x8` 的 `Compute Throughput` 与 `Memory Throughput` 虽然很高，但 `DRAM Throughput` 明显更大，说明它更依赖外部带宽，最终没有转化为更短的 kernel 时间
- `Shared 32x32` 的表现最差，Nsight Compute 也提示其 occupancy 受到寄存器、shared memory 和 block 内 warp 数量的限制

因此，在当前硬件、当前测试数据和当前实现方式下，可以把 `16x16` 视为当前 shared memory 版本的最佳 block shape。继续单纯扫描 block shape 的收益已经不高，下一步更值得尝试的方向应该是：

- register blocking
- shared memory 访问模式优化
- bank conflict 分析
- source counters / warp state 的进一步定位
