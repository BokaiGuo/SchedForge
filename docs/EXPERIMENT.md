# Experiment Methodology / 实验方法

[English](#english) · [简体中文](#简体中文)

## English

### Research Question

Can a target-aware compiler rank and select faster CPU schedules for the same
fused tensor computation before exhaustively benchmarking the search space?

### Workload

The primary workload is FP32 `MatMul + Bias + ReLU`. Additional BF16 and INT8
paths validate datatype plumbing but are not presented as optimized ISA kernels.

### Schedule Variables

- Loop order: `i-j-k` or `i-k-j`
- Outer tiles: `MC`, `NC`, `KC`
- Inner/cache tiles: `BM`, `BN`, `BK`
- Register block: `MR x NR`
- SIMD width and K-loop unroll factor
- PackA and PackB decisions
- Software-prefetch distance
- Thread count and affinity
- Fused or separate Bias/ReLU epilogue

### Measurements

- Median wall-clock execution time after warmup
- MatMul GFLOPS or datatype-appropriate GOPS
- Maximum absolute error against a scalar FP32 reference
- Simulated cache, DTLB, prefetch, bandwidth, and register-pressure metrics
- Linux `perf` counters when host permissions and PMU support permit them
- Spearman rank correlation and top-k recall between predicted and measured schedules

### Protocol

1. Generate deterministic random inputs.
2. Lower the same tensor computation into each scheduled Loop IR candidate.
3. Reject statically invalid schedules, including estimated register spills.
4. Rank remaining candidates using the target-aware simulator and cost model.
5. Benchmark only the selected top-k candidates unless running a prediction study.
6. Compare every executed candidate with the scalar reference.
7. Report medians and preserve negative results such as packing regressions.

### Claim Boundaries

- The simulator targets relative ranking, not cycle-accurate Intel execution.
- Recorded performance is host-specific and must not be generalized across CPUs.
- Simulated cache or TLB statistics are never reported as measured PMU counters.
- AVX-512, NEON, and multi-node NUMA behavior require validation on matching hardware.

## 简体中文

### 研究问题

对于同一个融合 Tensor 计算，目标感知编译器能否在穷举所有真机 Benchmark
之前，预测并选择更快的 CPU Schedule？

### 工作负载

主要工作负载为 FP32 `MatMul + Bias + ReLU`。BF16 和 INT8 路径用于验证
数据类型基础设施，不将其表述为已经完成 ISA 专用优化的高性能 Kernel。

### Schedule 变量

- 循环顺序：`i-j-k` 或 `i-k-j`
- 外层 tile：`MC`、`NC`、`KC`
- 内层/Cache tile：`BM`、`BN`、`BK`
- 寄存器分块：`MR x NR`
- SIMD 宽度和 K 循环展开因子
- PackA / PackB 决策
- 软件预取距离
- 线程数量和亲和性
- Bias/ReLU 融合或独立执行

### 测量指标

- Warmup 后多次执行的中位数时间
- MatMul GFLOPS 或对应数据类型 GOPS
- 相对标量 FP32 Reference 的最大绝对误差
- 模拟的 Cache、DTLB、Prefetch、带宽和寄存器压力指标
- 在权限与 PMU 支持允许时采集 Linux `perf` 计数器
- 预测 Schedule 与真机结果之间的 Spearman 相关性和 Top-k recall

### 实验流程

1. 使用固定随机种子生成输入。
2. 将同一个 Tensor 计算 Lower 为不同 Schedule 对应的 Loop IR。
3. 静态剔除不合法配置，包括预测会发生寄存器 Spill 的方案。
4. 使用目标感知 Simulator 和 Cost Model 排序剩余候选。
5. 除预测误差研究外，仅对 Top-k 候选执行真机 Benchmark。
6. 每个执行候选都与标量 Reference 比较正确性。
7. 报告中位数，并保留 Packing 退化等负结果。

### 声明边界

- Simulator 用于相对排序，不是 cycle-accurate Intel CPU 模拟器。
- 仓库记录的性能仅属于实验主机，不可直接推广到其他 CPU。
- 模拟 Cache/TLB 指标不得描述成真实 PMU 测量结果。
- AVX-512、NEON 和跨 NUMA Node 行为需要匹配硬件进一步验证。
