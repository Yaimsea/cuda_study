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

### naive 版本在 `ncu basic` 下的统一基线补测

为了统一 profiling 口径，我后来又单独对 `naive` 版本重新使用：

```bash
ncu --set basic --kernel-name matrixProduct --force-overwrite -o naive_ncu ./mm_naive
```

这一轮重新补测得到的结果为：

```text
Kernel: matrixProduct
Block size: 16 x 16
Grid size: 256 x 256
Matrix size: 4096 x 4096

Nsight Compute (basic):
Duration: 680.92 ms
Compute Throughput: 51.73%
Memory Throughput: 64.91%
DRAM Throughput: 11.00%
```

这个结果和前面 block shape 扫描里记录的 `Naive = 700.49 ms` 非常接近，说明虽然单次 profiling 结果会有一定波动，但整体量级和结论是一致的：

- `naive` 版本依然明显慢于后续所有 shared memory / register block 版本
- 其主要特征仍然是 `Memory Throughput` 高于 `Compute Throughput`
- `DRAM Throughput` 仍然不高，说明问题并不是简单地“显存带宽被完全打满”

因此，后面关于 `naive -> shared -> register block` 这条优化主线的判断并没有改变；这次补测的意义主要在于把 `naive` 版本也统一到了 `ncu basic` 的 profiling 口径下。

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

### register block 参数扫描

在确定 `16x16` 是当前 shared memory 版本的最佳 block shape 之后，我继续在这个 tile 上做了 register block 参数扫描。

这里我采用的命名约定是：

- `x_registerSize` 对应列方向
- `y_registerSize` 对应行方向

因此像 `1x4` 的含义不是“横向拉长 4”，而是：

```text
一个 thread 负责 1 列 x 4 行的输出子块
```

这一轮我主要测试了这些组合：

- `1x1`
- `1x2`
- `1x4`
- `1x8`
- `1x16`
- `2x2`
- `2x4`
- `4x2`
- `4x4`

其中 `1x1` 可以看作 shared memory 版本但不使用 register block 的 baseline，其结果就是前面 `Shared 16x16` 的中位数。

所有版本同样统一使用：

```bash
nvcc -O3 -lineinfo ...
```

并尽量采用三联重测，中位数作为最终记录结果。整理后的结果如下：

| Register Block | Kernel Duration (ms) | Compute Throughput (%) | Memory Throughput (%) | DRAM Throughput (%) | 备注 |
|---|---:|---:|---:|---:|---|
| `1x1` | 190.96 | 95.45 | 95.45 | 38.33 | shared baseline |
| `1x2` | 130.92 | 92.99 | 92.99 | 52.63 | 明显优于 baseline |
| `1x4` | 100.03 | 93.25 | 93.25 | 67.66 | 明显优于 `1x2` |
| `1x8` | 91.45 | 89.68 | 89.68 | 73.76 | 当前最优 |
| `1x16` | 223.49 | 97.31 | 97.31 | 30.96 | 单次测试已明显退化 |
| `2x2` | 140.92 | 61.13 | 94.57 | 50.99 | 优于 baseline |
| `2x4` | 115.36 | 57.12 | 95.36 | 58.81 | 强于 `2x2` |
| `4x2` | 188.82 | 31.22 | 97.92 | 36.54 | 明显退化 |
| `4x4` | 178.76 | 56.89 | 95.75 | 38.38 | 明显退化 |

从这轮实验里可以看出几个非常明确的趋势：

1. `register block` 确实有效。  
   只要从 `1x1` 走到 `1x2`，kernel 时间就从 `190.96 ms` 下降到了 `130.92 ms`。

2. 这份 kernel 更偏好沿 `y` 方向增大 register block。  
   也就是说，在当前实现里，一个线程负责更多“行方向输出”比扩展成更大的二维子块更有效。

3. 最优点不是“越大越好”，而是存在明显甜点。  
   `1x2 -> 1x4 -> 1x8` 持续变快，但 `1x16` 直接明显退化，说明 block 内线程数太少后，warp 利用率和调度效率会变差。

4. 二维更大的 register block 并没有赢。  
   `2x4` 虽然表现不错，但仍然不如 `1x4` 和 `1x8`；`4x2`、`4x4` 都进一步退化。

在这一轮 register block 参数扫描中，最好的结果来自：

```text
Shared 16x16 + register block 1x8
```

它相对 `naive` 的加速比为：

```text
Speedup = 700.49 / 91.45 = 7.66
Improvement(%) = (700.49 - 91.45) / 700.49 * 100% = 86.95%
```

相对 shared memory 但不使用 register block 的 `1x1` baseline，则有：

```text
Speedup = 190.96 / 91.45 = 2.09
Improvement(%) = (190.96 - 91.45) / 190.96 * 100% = 52.11%
```

这说明在当前硬件和当前 kernel 结构下，register block 的收益甚至比我前面做 block shape 扫描时更大。也就是说，下一阶段真正值得继续深入的方向已经不再是简单换 block shape，而是围绕当前最优的 `16x16 + 1x8` 版本，进一步分析：

- shared memory 访问模式
- bank conflict
- source counters
- warp state / stall 原因

### 32x32 tile 上的 register block 复测

在前面的 block shape 扫描里，`Shared 32x32` 的表现并不理想，因此当时的判断更偏向于认为 `32x32` 不是当前实现的最佳选择。

但在继续引入 `register block` 之后，这个结论需要更新。新的实验结果表明：

```text
32x32 tile 这条线本身并不差；
真正决定表现的关键，在于它是否搭配了合适的 register block。
```

这一轮我继续围绕 `32x32 tile` 做了三组复测，主要比较：

- `1x4`
- `1x8`
- `1x16`

其中 `32x32 tile + 1x8 register block` 的三次结果为：

```text
63.38 ms, Compute 98.62%, Memory 98.62%, DRAM 53.90%
72.47 ms, Compute 98.21%, Memory 98.21%, DRAM 51.99%
64.38 ms, Compute 98.49%, Memory 98.49%, DRAM 53.75%
```

按之前一直采用的中位数记录方式，可记为：

```text
Duration = 64.38 ms
Compute Throughput = 98.49%
Memory Throughput = 98.49%
DRAM Throughput = 53.75%
```

`32x32 tile + 1x16 register block` 的三次结果为：

```text
86.27 ms
86.02 ms
84.95 ms
```

其中位数结果为：

```text
Duration = 86.02 ms
Compute Throughput = 98.28%
Memory Throughput = 98.28%
DRAM Throughput = 40.15%
```

`32x32 tile + 1x4 register block` 的结果为：

```text
Duration = 79.81 ms
Compute Throughput = 97.43%
Memory Throughput = 97.43%
DRAM Throughput = 44.57%
```

把这几组结果放在一起之后，可以得到当前 `32x32 tile` 这条线上的明确排序：

1. `1x8` 最好
2. `1x4` 次之
3. `1x16` 更差

这说明在 `32x32 tile` 下，`register block` 的最佳选择同样不是“越大越好”。  
从 `1x4` 增加到 `1x8` 时，kernel duration 继续下降；但再扩大到 `1x16`，性能反而明显退化，说明线程块内部的并行度、寄存器压力和调度效率之间依然存在一个甜点，而当前甜点更接近 `1x8`。

更重要的是，这轮结果已经足以推翻之前对 `32x32` 的简单判断。  
之前 `Shared 32x32` 表现不佳，只能说明“没有合适 register block 的 32x32”不理想；但现在的数据已经说明：

```text
32x32 tile + 合适的 register block
可以明显优于之前的 16x16 最优版本。
```

前面记录中的最佳版本是：

```text
Shared 16x16 + register block 1x8
Duration = 91.45 ms
```

而这次新的最好结果是：

```text
Shared 32x32 + register block 1x8
Duration = 64.38 ms
```

两者相比：

```text
Speedup = 91.45 / 64.38 = 1.42
Improvement(%) = (91.45 - 64.38) / 91.45 * 100% = 29.60%
```

如果相对最初记录中的 `Naive = 700.49 ms` 来看，则有：

```text
Speedup = 700.49 / 64.38 = 10.88
Improvement(%) = (700.49 - 64.38) / 700.49 * 100% = 90.81%
```

这说明在当时那一轮 `ncu basic` profiling 视角下，项目里的“阶段性最佳版本”已经发生了变化。  
至少在那一轮实现、那套测试数据和当时的 profiling 口径下，新的最优候选已经变成：

```text
Shared 32x32 + register block 1x8
```

这轮实验还有一个很重要的观察点：  
`32x32 + 1x8` 的 `Compute Throughput` 和 `Memory Throughput` 都已经接近 `98.5%`，但 `DRAM Throughput` 仍然没有被完全打满。这说明当前 kernel 的收益已经不再只是“多吃一点显存带宽”这么简单，而更可能来自：

- 更高效的片上数据复用
- 更合适的线程块组织方式
- 更合理的 register block 粒度

因此，下一步最值得深入分析的对象，也应该从之前的 `16x16 + 1x8` 切换为现在的：

- `32x32 + 1x8`

后续如果继续做 Nsight Compute 深挖，建议重点关注：

- occupancy
- register pressure
- shared memory 访问模式
- bank conflict
- source counters
- warp state / stall 原因

### 锁频后的实验规程调整

需要特别说明的是：从这一节开始，下面记录的时间数据不再是前面 `Nsight Compute Duration` 的 profiling 结果，而是我在统一锁频规程下，使用 `cudaEvent` 记录得到的 benchmark 时间。

也就是说，README 到这里开始分成两条并行线索：

- 前面的内容主要是 `ncu` 视角下的 profiling 记录，用来分析阶段性优化方向和瓶颈变化
- 从这一节开始，主要记录统一锁频条件下的 `cudaEvent` benchmark 结果，用来做手写 kernel 与 `cuBLAS` 的正式性能对比

因此，后面出现的 `36.397 ms`、`13.6834 ms` 这类数字，不应该和前文 `700.49 ms`、`91.45 ms`、`64.38 ms` 这些 `ncu Duration` 直接混在一起做绝对比较。

在继续做 benchmark 时，我发现单纯使用 `cudaEvent` 做多次测试仍然会遇到比较明显的波动。进一步排查后，确认主要问题不在 kernel 正确性，而在 GPU 的动态频率策略，尤其是 `memory clock` 的降档会直接拉高 kernel time。

因此，这一阶段我开始尝试在 Windows 侧使用 `nvidia-smi` 设置 GPU 核心锁频目标，并在 WSL 中运行程序。实验过程中逐步形成了下面这套更稳定的规程：

- 在 Windows 侧先设置核心锁频目标
- 在 WSL 中确认当前 `graphics clock` 和 `memory clock`
- 在正式测试前，先进行 `5` 次 warmup
- 正式 benchmark 采用 `9` 次测试并记录中位数

这一轮实验里，我最终确定的正式档位不是 `2550` 或 `2700`，而是：

```text
Lock target: 2850 MHz
```

需要说明的是，这个数值是“目标锁频值”，并不等于 GPU 实际运行时始终稳定在 `2850 MHz`。实测中可以看到，核心频率通常会因为功耗墙或平台限制，实际落在：

```text
2715 ~ 2730 MHz
```

但和 `2550`、`2700` 这两个目标档位相比，`2850` 对应的整体运行状态更稳定。结合观测结果来看，真正让 benchmark 稳定下来的关键，很可能不是核心频率本身更高，而是：

```text
2850 这个目标档位更容易把 memory clock 保持在高频状态
```

也就是说，`2850` 的价值不在于“GPU 真正跑到了 2850 MHz”，而在于它让整张卡在当前平台上更容易进入一个对 benchmark 更友好的稳定状态。

### 手写 kernel 在正式档位下的结果

在将锁频目标固定为 `2850 MHz`，并采用“锁频后直接进入测试、正式测试 `9` 次取中位数”这套规程后，主版本 `[matrix_product.cu](/home/ngsxz/cuda_study/matrix_product.cu)` 的结果稳定在：

```text
Median kernel time ≈ 36.4 ms
```

其中一轮代表性结果如下：

```text
Median kernel time: 36.397 ms
The kernel time for these tests:
35.922 ms
35.9794 ms
36.2692 ms
36.2906 ms
36.397 ms
36.4199 ms
36.4203 ms
36.4317 ms
36.4549 ms
```

这说明在当前实验条件下，手写 kernel 的主体分布已经相当集中，大多数样本都能稳定落在 `36.2 ~ 36.5 ms` 这一带。

### cuBLAS 对照结果

在同样的锁频目标 `2850 MHz` 下，我也对 `[cublasSgemm.cu](/home/ngsxz/cuda_study/cublasSgemm.cu)` 做了同样规程的 benchmark。该版本同样采用：

- 锁频后直接进入测试
- 正式测试 `9` 次
- 记录中位数

代表性结果如下：

```text
Median kernel time: 13.6834 ms
13.4063 ms
13.4511 ms
13.4814 ms
13.4832 ms
13.6834 ms
13.7177 ms
13.7932 ms
13.8423 ms
13.8459 ms
```

这说明在当前正式实验档位下，`cuBLAS SGEMM` 的稳定表现大约为：

```text
Median kernel time ≈ 13.7 ms
```

### 当前正式结论

在正式实验规程和统一锁频目标下：

```text
Handwritten kernel: 36.397 ms
cuBLAS SGEMM:      13.6834 ms
```

因此，当前手写版本相对 `cuBLAS` 的性能比例约为：

```text
13.6834 / 36.397 = 37.6%
```

也就是说，在目前这版实现、这套测试数据以及这套锁频规程下，我的手写矩阵乘法大约达到了 `cuBLAS` 的：

```text
37% ~ 38%
```

这一阶段最重要的收获不是继续把绝对时间压到最低，而是明确了下面几点：

- benchmark 的主要波动源来自 `memory clock` 降档，而不是单纯的 kernel 写法错误
- 锁频时应该关心“哪一个锁频目标能带来最稳定的整体运行状态”，而不是只看目标值本身
- 在当前机器上，`2850 MHz` 这个目标锁频值虽然实际跑不到 `2850`，但它对应的实验结果最稳定，因此适合作为正式对照实验档位
- 在统一实验条件下，当前手写 kernel 已经能够稳定达到 `cuBLAS` 的约 `37% ~ 38%`
