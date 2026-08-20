<div align="center">

# SchedForge

**SchedForge — A Model-to-Machine Target-Aware CPU AI Compiler**

**调度锻造器：把张量计算逐层编译成高效的 CPU 代码。**

[English](README.md) · [架构说明](docs/ARCHITECTURE.md) · [实验报告](results/FINAL_REPORT.md) · [参与贡献](CONTRIBUTING.zh-CN.md)

[![CI](https://github.com/BokaiGuo/SchedForge/actions/workflows/ci.yml/badge.svg)](https://github.com/BokaiGuo/SchedForge/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://isocpp.org/)
[![LLVM 18](https://img.shields.io/badge/LLVM-18-262D3A.svg)](https://llvm.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

SchedForge 是一个使用 C++20 编写、同时面向稠密、动态路由稀疏与 Attention
张量程序的
model-to-machine CPU AI 编译器。它能够
导入 StableHLO 子集，构建真正的多算子 Tensor SSA Graph，执行 Shape 推导、
图正规化、融合、Dispatch 形成、Layout 传播、Bufferization 和 Workspace
复用，再通过 Structured Tensor Compute、Transform IR、Loop IR、Tensorization
和 LLVM 18 ORC JIT、目标特化 ELF AOT 包或原生 AVX2 后端生成并执行 CPU 代码。

当前旗舰 Workload 是一层完整的 Llama/Mistral 风格 Transformer Decoder。
一个 StableHLO 输入会被导入为一张图，编译为一个 `DecoderExecutablePlan`，并由
一次 Runtime 调用完成 RMSNorm、融合 QKV、RoPE、GQA/MQA Flash-style Attention、
输出投影、残差，以及 Dense SwiGLU 或 Top-K MoE FFN。MatMul 继续作为 Kernel
性能与硬件自动调优实验室。

`RMSNorm → 融合 QKV → RoPE → GQA/MQA Attention → O 投影 → 残差 → RMSNorm → Dense SwiGLU / MoE → 残差`。

`Router → Softmax → TopK → Segmented Dispatch → Grouped Expert SwiGLU → Weighted Combine`。

`QKᵀ → Scale → Causal Mask → Online Softmax → PV`；IO-aware Lowering 不会物化
完整 Attention Matrix。

```text
StableHLO / AI Graph
  → Tensor SSA + Shape Inference
  → Decoder Fusion + Dispatch IR
  → Layout + Bufferization
  → Structured Compute + Transform IR
  → Scheduled Loop IR + Tensorization
  → LLVM IR / x86 Machine Code
  → ExecutablePlan + AI Runtime
```

同一份调度方案会同时交给循环变换、模拟器、原生后端、LLVM 后端和自动调优器，
避免出现“模拟器预测的是一种执行方式，真正运行的却是另一种实现”的问题。

```mermaid
flowchart LR
    A["张量计算图"] --> B["SSA Tensor IR"]
    B --> C["Dense Dispatch"]
    B --> R["MoE Router + TopK"]
    B --> A["SDPA Fusion + Attention Planner"]
    R --> S["Segmented Tensor + Grouped Expert IR"]
    S --> P["Routing-aware Strategy Planner"]
    C --> D["显式 Scheduled LoopIR"]
    P --> D
    A --> T["Materialized / IO-aware / Split-KV"]
    T --> D
    D --> E["多级分块"]
    D --> F["数据打包与微内核"]
    D --> G["向量化、多线程与算子融合"]
    E --> H["Cache、TLB 与寄存器模型"]
    F --> H
    G --> H
    H --> I["代价模型与自动调优"]
    I --> J["原生 AVX2 后端"]
    I --> K["LLVM JIT / ELF AOT 后端"]
    J --> L["真机测试与 perf"]
    K --> L
    L --> M["模型校准与误差分析"]
```

## 主要能力

- **编译器基础设施**：实现 SSA 类型系统、`Value`、use-def 关系、
  `Operation`、嵌套 `Block`、`Module`、`IRBuilder` 和分层 PassManager。
- **Graph Compiler**：多算子 Tensor SSA、静态/动态/符号维度、Shape Constraint、
  StableHLO 子集导入、图正规化、Fusion Legality/Profitability、Dispatch IR。
- **Decoder Layer Compiler**：一个 StableHLO Graph 编译成一个可执行计划，覆盖
  RMSNorm、编译期 QKV/Gate-Up 权重打包、RoPE、GQA/MQA Attention、残差、
  Dense SwiGLU 与可选 Top-K MoE。
- **真实规模 Decoder Suite**：覆盖 24 个 Tiny/Medium/Large Prefill、Decode 与
  MoE Profile，明确区分真机实测和 compile-only，并报告阶段占比、Peak Workspace、
  Compile/JIT 时间和等效 token/s。
- **Whole-Graph Planner**：`ExecutablePlanOptimizer` 联合选择 Attention 策略、
  Layout、Materialization、Workspace 复用、Schedule Family、线程数与放置策略，
  最终依据预算内的端到端真机测量选择计划。
- **MoE Compiler**：显式 Router/TopK/Histogram/Prefix/Dispatch/Combine IR、
  Segmented Tensor、Variable-M Grouped Expert GEMM、SwiGLU、Token Bucket、
  Routing Trace 与 Load-aware Expert Task Scheduler。
- **Attention Compiler**：SDPA 结构融合、MHA/GQA/MQA、Causal Tile、精确在线
  Softmax、TilePipelineIR、Materialized/IO-aware Prefill、可增长 KV Cache 与
  并行 Split-KV Decode，以及单函数可执行 LLVM Online-Softmax Attention。
- **Production LLVM 研究**：ORC 执行保留 Scheduled LoopIR 的线程语义；汇编
  报告覆盖指令、分支、地址计算、栈访问和 Spill，并提供可复现双后端矩阵。
- **AOT 部署**：LLVM `TargetMachine` 生成 PIC ELF Object；带版本的 `.sfe`
  保存目标、Shape、ABI Guard 与校验和，并可由新进程在不运行 LLVM 编译器的
  情况下通过 `dlopen` 直接执行。
- **内存与布局编译**：Layout 进入 Tensor Type；支持跨 Dispatch 布局传播、
  Bufferization、生命周期分析、64 字节对齐 Workspace 复用和 Guarded Specialization。
- **结构化 Kernel Compiler**：Iteration Domain、并行/归约 Iterator、Indexing Map、
  Transform IR 序列化、回放与直接生成 LoopIR，以及由实测 Schedule 生成的变换程序。
- **显式可执行 LoopIR**：Schedule 会真正 rewrite 出 `scf.for`、
  `scf.parallel`、Pack、Prefetch、Load、Accumulator、Vector FMA、Epilogue
  和 Store。原生执行、Simulator 与 LLVM 都消费 LoopIR，不再读取 Schedule 字段。
- **CPU 循环优化**：支持多级分块、MR/NR 寄存器分块、K 维展开、A/B 矩阵
  数据打包、软件预取、SIMD 向量化、尾部处理、算子融合和多线程执行。
- **生成式微内核后端**：原生 AVX2 与 LLVM ORC 都支持寄存器驻留 MR×NR
  微内核；LLVM 路径真实生成 Vector FMA，并检查正确性与寄存器 Spill。
- **硬件感知代价模型**：模拟组相联 L1/L2/L3 Cache、DTLB、软件预取、
  寄存器占用、潜在 Spill、数据打包开销和内存带宽。
- **真机优先自动调优**：所有去重后的合法候选都先在真实硬件运行，再对实测
  前列随机交错复测；测量数据库可用于 analytical + learned 混合模型，也可通过
  `schedforge-compile --measurement-db=records.csv` 直接复用实测赢家。
- **AI Runtime**：可序列化 `.sfe` ExecutablePlan，包含常量、Buffer、Dispatch、
  Shape Guard、Transform IR、LLVM Kernel Artifact 和 Workspace；MLP 特化会同时
  具体化动态 Tensor SSA、Buffer、LoopIR 与 LLVM Artifact。
- **Transformer/推理抽象**：可执行 Dense MLP、MoE、IO-aware Prefill 与
  KV-cache Decode；量化 Tensor 元数据与传播、BF16/INT8 参考路径和动态 Guard。
- **单一真源图 Epilogue**：原生执行与 LLVM ORC 都消费 Scheduled LoopIR 中显式的
  GELU/Residual 操作，不再依赖后端外层手写包装逻辑。

## 快速开始

### 1. 安装依赖

以下命令适用于 Ubuntu 24.04：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build \
  llvm-18-dev llvm-18-runtime llvm-18-tools clang-18
```

如果还需要采集硬件性能计数器，可以安装：

```bash
sudo apt-get install -y linux-tools-common linux-tools-generic
```

### 2. 编译并运行测试

```bash
git clone https://github.com/BokaiGuo/SchedForge.git
cd SchedForge

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

如果 CMake 没有自动找到 LLVM，可以显式指定配置目录：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
```

### 3. 体验完整工具链

```bash
# 查看 Tensor IR、调度后的 Loop IR 和生成的 LLVM IR
./build/schedforge-opt --M=129 --N=131 --K=127

# 查看 Cache、TLB、寄存器压力和预取效果的预测结果
./build/schedforge-sim --M=256 --N=256 --K=256

# 使用原生 AVX2 后端运行
./build/schedforge-run --backend=native --M=256 --N=256 --K=256

# 使用 LLVM 即时编译后端运行
./build/schedforge-run --backend=llvm --M=256 --N=256 --K=256

# 编译、检查并运行目标特化 AOT 包
./build/schedforge-aot compile --M=128 --N=128 --K=128 \
  --output=results/matmul_128.sfe
./build/schedforge-aot inspect --artifact=results/matmul_128.sfe
./build/schedforge-aot run --artifact=results/matmul_128.sfe --repetitions=10

# 校准硬件参数后执行自动调优
./build/schedforge-bench --M=192 --N=192 --K=192 \
  --threads=8 --autoschedule --top-k=12 --calibrate

# 将 StableHLO Transformer MLP 编译成可执行计划
./build/schedforge-compile examples/transformer_mlp.mlir \
  --target=native-cpu --batch=1 --sequence=16 \
  --hidden=64 --intermediate=128 --threads=4 \
  -o results/transformer_mlp.sfe

# 编译并真实执行一层完整 Dense Transformer Decoder
./build/schedforge-decoder examples/decoder_layer.mlir \
  --batch=1 --sequence=4 --hidden=16 --intermediate=32 \
  --q-heads=4 --kv-heads=2 --head-dim=4 --threads=2 \
  -o results/decoder_dense.sfe

# 使用 Top-2 MoE FFN 执行同一 Decoder Layer
./build/schedforge-decoder examples/decoder_layer_moe.mlir --moe \
  --batch=1 --sequence=4 --hidden=16 --intermediate=32 \
  --q-heads=4 --kv-heads=2 --head-dim=4 --experts=4 --top-k=2 \
  --threads=2 -o results/decoder_moe.sfe

# 运行真实规模 Decoder 矩阵；过重档位明确保留为 compile-only
./build/schedforge-decoder-bench --suite=realistic --threads=8 \
  --repetitions=5 --max-real-gflop=1.2 --max-weight-mib=256 \
  --output=results/decoder_realistic.csv

# 将全图候选与显式默认计划进行端到端真机对比
./build/schedforge-decoder-bench --suite=optimizer --threads=8 \
  --repetitions=9 --max-real-gflop=1.2 --max-weight-mib=256 \
  --output=results/decoder_plan_optimizer.csv

# 编译并真实运行动态路由 Top-2 MoE MLP
./build/schedforge-moe --tokens=128 --hidden=512 --intermediate=2048 \
  --experts=8 --top-k=2 --threads=8 --router-data \
  --strategy=auto -o results/moe_mlp.sfe

# 编译并真实运行 CPU Flash-style Causal Prefill
./build/schedforge-attention \
  --batch=1 --q-heads=8 --kv-heads=8 --sq=128 --sk=128 \
  --head-dim=64 --value-dim=64 --causal --threads=8 \
  --strategy=auto -o results/attention_prefill.sfe

# 使用 KV Cache 运行 GQA Split-KV Decode
./build/schedforge-attention \
  --batch=1 --q-heads=8 --kv-heads=2 --sq=1 --sk=1024 \
  --head-dim=64 --value-dim=64 --causal --threads=8 \
  --strategy=auto -o results/attention_decode.sfe

# 运行 Routing Skew、Grouped Execution 与 Scheduler 对比实验
./build/schedforge-moe --tokens=64 --hidden=64 --intermediate=128 \
  --experts=8 --top-k=2 --threads=8 --routing=heavy \
  --experiment-csv=results/moe_strategy_matrix.csv
```

## 调度描述语言

SchedForge 使用一段紧凑文本描述变换方案。Schedule 不是最终执行配置；应用
之后会生成显式 Scheduled LoopIR，执行后端不再接收 Schedule。

```text
order=ikj;outer=64,128,64;tile=32,64,32;micro=4,8;
vector=8;unroll=4;threads=8;pack=ab;prefetch=4;fuse=true;pin=true
```

重写后的 IR 会明确表达循环嵌套、并行、Packing、Prefetch、寄存器驻留归约、
向量宽度、Epilogue 作用域与 Store，进入目标后端前即可检查和验证。

| 字段 | 作用 |
|---|---|
| `order` | 指定循环顺序，例如 `ijk` 或 `ikj` |
| `outer` | 设置 `MC,NC,KC` 外层分块 |
| `tile` | 设置 `BM,BN,BK` 内层 Cache 分块 |
| `micro` | 设置 `MR,NR` 寄存器分块 |
| `vector` | 设置 FP32 SIMD 向量宽度 |
| `unroll` | 设置 K 循环展开倍数 |
| `pack` | 选择是否打包 A、B 矩阵 |
| `prefetch` | 设置软件预取距离 |
| `threads` | 设置工作线程数量 |
| `fuse` | 选择是否融合 Bias 和 ReLU |
| `pin` | 选择是否将工作线程绑定到固定 CPU |

## 命令行工具

| 工具 | 用途 |
|---|---|
| `schedforge-opt` | 查看各级中间表示和 LLVM IR |
| `schedforge-sim` | 运行硬件模型并输出代价预测 |
| `schedforge-run` | 运行原生、LLVM、BF16 或 INT8 后端 |
| `schedforge-bench` | 搜索调度方案并测试候选性能 |
| `schedforge-calibrate` | 测量内存带宽并校准代价模型 |
| `schedforge-search` | 对比不同调度搜索算法 |
| `schedforge-study` | 分析数据打包收益和预测误差 |
| `schedforge-resolution` | 测量调优噪声与候选区分能力 |
| `schedforge-compile` | 将 StableHLO Graph 编译为 `.sfe` ExecutablePlan |
| `schedforge-moe` | 编译、模拟、执行并对比 MoE Execution Plan |
| `schedforge-attention` | 编译、调优、模拟并执行 Attention Plan |
| `schedforge-decoder` | 编译并执行完整 Dense 或 MoE Decoder Layer |
| `schedforge-decoder-bench` | 运行真实规模 Decoder 与全图计划实验 |
| `schedforge-codegen-study` | 从相同 LoopIR 对比 Native 与 LLVM 代码质量 |
| `schedforge-aot` | 编译、检查并运行目标特化 `.sfe` AOT 包 |
| `schedforge-aot-study` | 测量 JIT 编译与 AOT 编译、加载、执行成本 |
| `schedforge-next-study` | 运行 Paged KV、INT8、Transfer 和 NEON 能力实验 |
| `schedforge-fuzz` | 对 LoopIR 与数值不变量执行确定性 Fuzzing |

## v0.12-v0.18 Runtime 里程碑线

这条里程碑线全部落成可执行、可测试的切片，而不是模拟器宣传：v0.12 增加
非连续物理页 KV 和直接分页 Decode 遍历；v0.13 增加 INT8 Weight-only
MatMul；v0.14 在真实主机内存拷贝上搜索 Chunk/Worker 调度；v0.15 生成有效
ARM NEON Intrinsic 并执行 AArch64 交叉语法编译；v0.16 执行 LoopIR Fuzzing；
v0.17 将 INT8 权重接入 Dense Decoder 的全部投影与 Decode 路径；v0.18 增加
MoE Expert W1/W3/W2 INT8 执行。

```bash
./build/schedforge-next-study
./build/schedforge-fuzz --iterations=1000 --seed=1
./scripts/run_neon_qemu.sh
```

当前主机是 x86_64，因此不声明 ARM 原生硬件性能；NEON 已通过 AArch64 交叉语法
编译，并通过 QEMU 执行 AArch64 ELF 做运行正确性检查。INT8 Dense Decoder 已实际执行并校验，Paged Decode 热路径直接遍历页表；
`gather_paged_kv` 仅保留为检查/调试 API。

## Decoder Layer Compiler 实测 Demo

SchedForge 0.7 将 `examples/decoder_layer.mlir` 或
`examples/decoder_layer_moe.mlir` 作为一张完整 Graph 编译，生成单一 `.sfe`
计划。计划包含导入与正规化后的 Tensor SSA、融合决策、打包常量、
Memory Plan、QKV/O/Gate-Up/Down Scheduled LoopIR、嵌入的 Attention 或 MoE Plan，
以及 LLVM Kernel Artifact。Q/K/V 权重只在编译期拼接一次；Dense Gate/Up 权重
复用同一套常量特化路径。

仓库内真实 CPU Smoke 使用 `B=1, S=4, H=16, I=32, Hq=4, Hkv=2, D=4`。
Dense 端到端记录为 **0.025 ms**，MoE 为 **0.099 ms**，最大打印误差均为
**0.000**。这些小尺寸结果用于验证单 Plan 真实执行，不是吞吐性能声明。详见
`results/decoder_dense_run.txt`、`results/decoder_moe_run.txt` 与
[Decoder Compiler 设计](docs/DECODER_COMPILER.md)。

## 真实规模 Decoder 与 Whole-Graph Planning

SchedForge 0.8/0.9 增加了 24 个架构 Profile。在当前 Intel Core i5-14600K
上，**12 个 Profile 完成真实硬件执行**，另有 **12 个 Profile 因超过配置的
1.2 GFLOP 或 256 MiB 权重预算而明确标记为 compile-only**。compile-only 行的
Runtime Latency 固定为 0，不是 Simulator 估算。

代表性实测包括 Tiny Prefill `S=128` **3.753 ms**（**34,110 token/s**）、
Tiny Decode `KV=512` **1.443 ms**（**693 token/s**），Medium Decode
`KV=4096` **12.817 ms**（**78 token/s**），以及 Tiny 8-Expert Top-2 MoE
Uniform Routing **8.509 ms**。所有实测行的验证误差均低于 `1e-3`。

Whole-Graph 实验会测量显式默认计划，以及分析模型优先选出的 6 个候选。所有
候选使用相同预热；若出现候选胜者，再执行 3 轮 Baseline/Winner 交错复测。
Tiny Prefill 仍由默认计划获胜（**1.000×**）；Tiny Decode `KV=512` 则选择
单线程、非物化的 Split-KV 计划，交错复测后相对默认计划为 **1.400×**。
完整证据见
`results/decoder_realistic.csv`、`results/decoder_plan_optimizer.csv` 与
`results/decoder_plan_optimizer_candidates.csv`。

## Production LLVM 与 Fused Attention CodeGen

SchedForge 0.10 将同一份 Scheduled LoopIR 中的 `threads` 决策带入 LLVM ORC
执行，并使用 MR 对齐的行分区。仓库记录的 `192/256/512³` 实验中，LLVM 达到
**103-153 GFLOPS**，Native 为 **238-367 GFLOPS**。相比此前数量级差距已经明显
缩小，但这些行上 LLVM 仍慢 **1.8-2.5×**，项目不宣称已经达到 Native Parity。

Attention 后端现在也能生成并执行单个 LLVM Function，在不物化 `Sq×Sk` 的
情况下完成 QK、精确在线 Max/Sum Rescale、PV Accumulation 与最终归一化。在
`B=1, Hq=8, D=64` 下，Fused LLVM 的 MHA Prefill `S=128` 为 **0.791 ms**，
GQA Prefill `S=128` 为 **0.786 ms**，GQA Decode `Sk=1024` 为 **0.214 ms**，
最大误差低于 `5e-8`。专门化 Native 路径仍快 **2.1-3.1×**，而且融合 LLVM
汇编仍存在 Vector Spill Pattern。这些负结果被明确保留。详见
`results/llvm_codegen_study.csv` 与 `results/fused_attention_llvm.csv`。

## AOT 可执行部署

SchedForge 0.11 会把同一份优化后 Scheduled LoopIR 交给 LLVM 18
`TargetMachine`，生成 PIC ELF Relocatable Object，再链接为可加载的
`kernel.so`，最终写入带版本的 `.sfe` 目录。Manifest 会检查 ABI、精确 MatMul
Shape、Target Triple、Host CPU，以及 LoopIR、Object 和 Shared Object 的校验和。
同一组不变量也会嵌入 `kernel.so` 并在调用前交叉核对。`schedforge-aot run`
随后直接执行 `dlopen`/`dlsym`，运行时不调用 LLVM。

在记录的 Intel Core i5-14600K 上，`64/128/256³` 的 AOT 加载延迟为
**0.064-0.081 ms**，最大误差均低于 `6e-6`；直接执行时间为
**0.007/0.137/0.844 ms**。ORC 对比路径即使单线程也包含当前宿主 Worker Thread
封装，因此这里证明的是部署与调用开销，而不是声称 Object Emission 本身改变了
机器码质量。详见 `results/aot_deployment.csv` 与
[AOT 部署说明](docs/AOT_DEPLOYMENT.md)。

## MoE Compiler 实测 Demo

SchedForge 0.4 将 MoE Lower 为 18 个 Tensor SSA Operation 和 11 个
Routing/Expert Operation。生成的 `.sfe` 包含 Segmented Tensor 元数据、动态
Token Guard、Execution Strategy IR、按 Token Bucket 专门化的 W1/W3/W2
LoopIR 与经过 LLVM ORC 编译验证的 Kernel Artifact。

附件指定的完整 FP32 MVP：`T=128, H=512, I=2048, E=8, TopK=2`，已在当前
Intel Core i5-14600K 上真实运行，P50 延迟 **35.843 ms**、P95 延迟
**37.789 ms**，验证误差为 0。多行 AVX2 专家微内核会在多个路由 Token 之间
复用已加载的专家权重向量。这是单机 correctness-first 实现，不宣称已经达到
生产级 MoE 吞吐。

在仓库记录的完整尺寸实验 `T=128, H=512, I=2048` 中，Heavy Skew 下固定
Grouped 执行的模拟不均衡度为 2.0；Load-aware splitting 将其降为 0，并把实测
Grouped P50 从 **26.780 ms** 降至 **20.113 ms**。完整 27 组结果见
`results/moe_strategy_matrix.csv`。

## Attention Compiler 实测 Demo

SchedForge 0.5 会识别完整 SDPA 链并形成 `attention.sdpa`，在 Materialized、
Tiled Materialized、IO-aware Prefill 与 Split-KV Decode 之间选择。IO-aware
路径使用 `BQ×BK` TilePipelineIR，维护在线 Row Max、分母、Rescale 与 Output
Numerator；它是精确 Attention，并且不会分配完整 `Sq×Sk` Score/Probability。

在当前 Intel Core i5-14600K 上，Causal MHA Prefill
`B=1, H=8, Sq=Sk=128, D=64` 的 P50 为 **0.262 ms**，打印精度下误差为 0；
GQA Decode `Hq=8, Hkv=2, Sq=1, Sk=1024, D=64` 自动选择并行 Split-KV，
P50 为 **0.112 ms**，打印精度下误差为 0。

在 `S=512` 的四策略实测中，Materialized Attention 的临时内存为 **16 MiB**，
Auto-scheduled IO-aware 仅为 **49 KiB**；P50 从 **18.929 ms** 降至
**2.947 ms**。96 组 Scaling Analysis 覆盖 Head `8/12`、D `64/128` 和
序列长度 `128–4096`。`S=512`、200 次执行的 Linux PMU 记录为 P-core IPC
**2.561**、L1D Miss Rate **1.066%**、Cache-reference Miss Rate **16.230%**；
这些进程级计数包含 Runtime/框架开销。

## Graph Compiler 实测 Demo

仓库中的 `examples/transformer_mlp.mlir` 当前会生成：

- 12 个正规化后的 Tensor SSA Operation
- 2 个融合 Dispatch：`MatMul + Bias + GELU` 与 `MatMul + Bias + Residual`
- 跨 Dispatch 的 `blocked<6x16>` Layout 传播
- 朴素中间张量 32,768 字节，规划后 Workspace 8,192 字节
- 2 个携带 Transform IR 与 AVX2 Tensor Intrinsic 的 LLVM Kernel Artifact
- 每个 Dispatch 从 16,200 个生成候选中完成 3,825 次真机测量
- 当前记录主机上的原生 Scheduled Loop Runtime 为 0.021 ms
- 端到端 MLP 执行验证，最大误差低于 `1e-3`

可复现输出见 `results/transformer_mlp_compile.txt` 和
`results/transformer_mlp.sfe`。

## 实测结果

下面的数据来自一台 Intel Core i5-14600K，仅用于展示当前实现的实验结果，
不代表其他 CPU 或其他运行环境能够得到相同性能。

| 测试项目 | 实测结果 |
|---|---:|
| 原生后端真机自动调优，融合 192³ 矩阵乘 | **374.402 GFLOPS** |
| 显式 LoopIR 新鲜重测，融合 192³ 矩阵乘 | **369.728 GFLOPS** |
| 原生后端真机自动调优，融合 256³ 矩阵乘 | **390.772 GFLOPS** |
| 原生后端真机自动调优，融合 512³ 矩阵乘 | **434.863 GFLOPS** |
| `LLVM ORC JIT` 后端，融合 192³ 矩阵乘 | **31.216 GFLOPS** |
| BF16 相对 FP32 的最大绝对误差，128³ | **0.0303** |
| INT8 相对 FP32 的最大绝对误差，128³ | **0.0805** |

在没有命中调优缓存时，自动调优器会先静态剔除非法 Schedule，再合并当前
Runtime 中执行路径完全相同的配置，然后把所有剩余候选都在真实 CPU 上运行
一轮。实测前列会以随机顺序进行多轮交错复测，最终选择完全依据真机中位数。
Simulator 只用于解释 Cache、TLB 和寄存器压力，不参与性能候选的晋级和选择。

完整实验方法、负结果、原始数据和结论的适用范围，请查看
[最终实验报告](results/FINAL_REPORT.md)。

## 项目结构

```text
include/schedforge/   对外公开的编译器、IR、运行时和模拟器接口
src/                  IR、循环变换、LLVM 后端、运行时、模拟器和调度器实现
tools/                命令行工具
tests/                单元测试与端到端测试
docs/                 架构说明和设计决策记录
results/              经过整理的实验结果与报告
scripts/              性能计数器辅助脚本
```

## 当前边界

- SchedForge 是研究型 CPU AI 编译器原型，不是 oneDNN、XLA、IREE、TVM
  或生产级推理 Runtime 的替代品。
- 完整 Dense/MoE Decoder Layer、Transformer MLP、FP32 Top-2 MoE、精确
  IO-aware Prefill Attention 与连续 KV Cache Split-KV Decode 已端到端执行并验证。
- Large Profile 在当前主机上属于 compile-feasibility 证据；被权重或 FLOP 预算
  拒绝的档位不声明真实 Runtime Latency。
- 当前只在真实硬件上验证了 AVX2 后端。NEON 已通过 AArch64 交叉语法编译，
  但 AVX-512 与 ARM 运行时性能尚未在当前主机验证。
- MoE 的 P-core/E-core 放置、NUMA-aware 执行、量化 Router、Block-sparse
  Lowering 与分布式 Expert Parallelism 尚未实现；Expert W1/W3/W2
  Weight-only INT8 已实现。
- Attention 是面向 CPU Cache Hierarchy 的 Flash-style 精确 Attention，不是
  GPU FlashAttention-2 的实现或性能声明。
- BF16/INT8 Attention、Dropout/Backward、分布式 Attention，以及无 Spill
  且达到 Native Parity 的 LLVM Fused Attention 尚未实现。
- AOT Format v1 仅支持同目标、Shape 特化 FP32 MatMul 和单 Runtime Thread；
  Whole-Graph 常量重定位与多 Kernel Dispatch 仍属于后续工作。
- v0.17 已将 Weight-only INT8 接入 Dense Decoder 全部投影与 MoE Expert
  W1/W3/W2；低精度 Attention 仍是后续工作。
- v0.17 已交叉编译并通过 QEMU 执行 NEON AArch64 ELF；本 x86_64 主机仍不能代表 ARM 原生硬件性能。
- Paged Decode 已直接遍历非连续页并执行在线 Softmax；`gather_paged_kv`
  仅用于检查与调试。
- 模拟器只提供诊断信息，既不是性能选择裁判，也不会逐周期复现 Intel
  处理器的乱序执行。
- 性能会受到 CPU 型号、频率策略、编译器版本、输入规模和后台负载影响，
  请在自己的机器上重新运行实验。

## 延伸阅读

- [架构说明](docs/ARCHITECTURE.md)
- [Graph Compiler 与 `.sfe` 格式](docs/GRAPH_COMPILER.md)
- [显式 Scheduled LoopIR](docs/LOOP_IR.md)
- [MoE Compiler Pipeline](docs/MOE_COMPILER.md)
- [Attention Compiler Pipeline](docs/ATTENTION_COMPILER.md)
- [Decoder Layer Compiler Pipeline](docs/DECODER_COMPILER.md)
- [AOT 可执行部署](docs/AOT_DEPLOYMENT.md)
- [实验方法](docs/EXPERIMENT.md)
- [最终实验报告](results/FINAL_REPORT.md)
- [架构决策记录](docs/decisions/)
- [中文贡献指南](CONTRIBUTING.zh-CN.md)
- [安全策略](SECURITY.md)

## 参与贡献

欢迎提交 Issue 和 Pull Request。开始之前请先阅读
[中文贡献指南](CONTRIBUTING.zh-CN.md)，并确保完整测试已经通过。

## 引用

如果 SchedForge 对课程、研究或工程实验有帮助，可以按照
[CITATION.cff](CITATION.cff) 中的信息引用本项目。

## 开源许可证

SchedForge 使用 [MIT License](LICENSE) 发布。
