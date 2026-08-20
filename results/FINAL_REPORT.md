# SchedForge Performance and Validation Report

**Experiment date:** August 13, 2026

## Verdict

SchedForge now uses real CPU execution as the authority for auto-tuning. The
simulator remains available for cache, TLB, bandwidth, and register-pressure
diagnostics, but it does not rank finalists or select the winning schedule.

The native FP32 path was upgraded from a memory-updating `i-k-j` loop to AVX2
register-resident micro-kernels, a reusable worker team, and topology-aware CPU
affinity. These changes improve both throughput and small-matrix launch latency.

## Hardware Auto-Tuning Protocol

For a fresh tuning run:

1. Generate the schedule space.
2. Statically reject illegal schedules and predicted register spills.
3. Deduplicate schedules that map to the same current runtime execution path.
4. Warm reusable worker teams for each candidate thread count.
5. Execute every remaining candidate once on the host CPU.
6. Check measured candidates against the scalar FP32 reference.
7. Sort by measured screening time.
8. Re-run the measured top-k finalists in randomized, interleaved rounds.
9. Select the correct candidate with the lowest median time.

The tuning cache key includes a runtime implementation version. This prevents a
schedule selected for an older micro-kernel from being silently reused after a
runtime performance rewrite. CLI output explicitly reports cache hit or miss.

## Final Native Results

Host: Intel Core i5-14600K, AVX2/FMA, one NUMA node. Workload: fused FP32
`MatMul + Bias + ReLU`. Search used up to 12 threads, top-16 repeated finalists,
2 warmups, and 11 interleaved repetitions.

| Shape | Fresh-search candidates measured | Selected micro-kernel | Threads | Median time | Throughput |
|---|---:|---:|---:|---:|---:|
| 192³ | 5,355 | 6 x 16 | 6 | 0.038 ms | **374.402 GFLOPS** |
| 256³ | 5,355 | 6 x 16 | 6 | 0.086 ms | **390.772 GFLOPS** |
| 512³ | 5,355 | 6 x 16 | 6 | 0.617 ms | **434.863 GFLOPS** |

All three winners used unpacked row-major inputs, fused Bias/ReLU, AVX2 width 8,
and topology-aware thread pinning. The exact tile shape remains workload-specific
and is selected by real execution rather than simulator rank.

The previous recorded 192³ result was 84.584 GFLOPS. The new 374.402 GFLOPS
measurement is a **4.43x throughput increase** on the same host and workload
shape. This comparison is host-specific and not a portable speedup claim.

Raw result snapshots:

- `results/performance_192_autotune.txt`
- `results/performance_256_autotune.txt`
- `results/performance_512_autotune.txt`

After the explicit LoopIR migration, a fresh 192³ search measured all 5,355
runtime-distinct candidates and reached **369.728 GFLOPS** with a 6 x 16
micro-kernel and six threads. This is 1.25% below the earlier 374.402 GFLOPS
snapshot, so the architectural rewrite preserves nearly all recorded native
throughput but does not claim a new peak. The exact output is stored in
`results/loopir_performance_192.txt`.

## Runtime Optimizations

- AVX2 8-column and 16-column register-resident micro-kernels
- 4x16 and 6x16 kernels that use available YMM registers without spilling
- Bias and ReLU applied in registers before the final output stores
- Full-K accumulation without loading and storing C on every K iteration
- Reusable thread teams instead of per-invocation thread construction
- Linux topology-aware affinity that uses distinct physical cores before SMT siblings
- Explicit affinity reset for unpinned schedules
- Scalar tail handling for non-vector-divisible N and partial M blocks

## Correctness and Safety

- Release CTest suite passes.
- ASan and UBSan CTest suite passes.
- Non-divisible shapes 31x29x27, 127x131x113, and 193x197x181 pass against the
  scalar reference.
- Packed and scalar fallback paths remain covered.
- Fused and non-fused epilogues remain covered.
- Maximum absolute error for the final tuned square workloads is below `1e-3`.

## LLVM and Data Types

LLVM ORC now has a generated register-resident MR×NR path for legal static
shapes. The backend builds vector accumulator PHIs across K, emits vector FMA,
applies the fused epilogue before final stores, and retains the prior safe
vector-plus-tail implementation for irregular shapes. Unit tests validate
correctness, FMA presence, and the absence of vector stack-spill patterns.

BF16 and INT8 paths remain correctness-oriented reference kernels rather than
ISA-specialized AVX-512 BF16 or VNNI implementations.

## Graph Compiler Validation

SchedForge 0.6 closes the explicit executable-IR backbone for the model-to-machine
Transformer MLP path. The checked-in
StableHLO example is canonicalized to 12 Tensor SSA operations and compiled into
two dispatches: `MatMul + Bias + GELU` and `MatMul + Bias + Residual`.

The v0.6 compiler path now uses `TransformProgram::apply` to generate verified
Scheduled LoopIR, can select exact-shape schedules from a measurement database,
and can concretize symbolic `B*S` plans into new buffer plans, dispatch problems,
LoopIR, guards, and LLVM artifacts. StableHLO constants are preserved in Tensor
SSA rather than discarded during import.

For the recorded `batch=1, sequence=16, hidden=64, intermediate=128` run:

- imported/canonical operations: 12
- fused dispatches: 2
- propagated inter-dispatch layout: `blocked<6x16>`
- naive intermediate memory: 32,768 bytes
- planned workspace: 8,192 bytes
- native and LLVM graph epilogues: explicit GELU and residual LoopIR operations
- LLVM validation: final graph LoopIR JIT output remains below `1e-3` max error

The non-autotuned v0.6 architecture smoke on the same host used four threads,
compiled two LLVM kernels, executed the native Scheduled LoopIR MLP in **0.038
ms**, and reported zero at the displayed three-decimal error precision. This is
a functional architecture snapshot, not a replacement for the earlier autotuned
0.021 ms performance record. Raw v0.6 evidence is stored in:

- `results/transformer_mlp_v06_compile.txt`
- `results/transformer_mlp_v06.sfe`

The earlier autotuned graph-compiler record remains:

- generated LLVM kernels: 2
- compiled Scheduled LoopIR programs: 2
- generated schedule candidates per dispatch: 16,200
- hardware measurements per dispatch: 3,825
- LLVM JIT compilation time: recorded in `results/transformer_mlp_compile.txt`
- native scheduled-loop end-to-end execution: 0.021 ms
- maximum absolute error: below `1e-3`

The `.sfe` artifact contains Tensor SSA, Structured Compute, Dispatch IR,
Transform IR, explicit Scheduled LoopIR, tensor intrinsics, buffer plans, shape
guards, and LLVM kernel IR.
The recorded MLP runtime uses the native scheduled-loop dispatch path selected
by hardware auto-tuning; LLVM ORC is compiled and validated separately and its
IR is embedded as an executable-plan artifact.

## Full Decoder Layer Validation

SchedForge 0.7 validates a complete Llama/Mistral-style Decoder Layer rather
than invoking Attention, Dense MLP, and MoE as disconnected demos. One
StableHLO file is imported, structural QKV/Gate-Up/RoPE patterns are recognized,
one `DecoderExecutablePlan` is serialized, and one runtime call evaluates the
full residual graph.

The checked-in Dense and MoE integration runs use
`B=1, S=4, H=16, I=32, Hq=4, Hkv=2, D=4` on the recorded Intel Core i5-14600K:

- imported StableHLO operations: 25 Dense, 21 MoE
- compile-time packed QKV constant: 2,048 bytes
- compile-time packed Dense Gate-Up constant: 4,096 bytes
- Dense naive activation bytes / planned workspace: 4,096 / 1,536
- MoE naive activation bytes / planned workspace: 7,028 / 3,444
- Dense end-to-end execution: **0.025 ms**
- MoE end-to-end execution: **0.099 ms**
- maximum printed end-to-end error: **0.000** for both branches

These are real executions, not simulator estimates. They are intentionally
small CI/integration shapes and therefore are not production-model throughput
claims. The raw console records and plans are `results/decoder_dense_run.txt`,
`results/decoder_dense.sfe`, `results/decoder_moe_run.txt`, and
`results/decoder_moe.sfe`.

## Realistic Decoder Matrix and Whole-Graph Planning

SchedForge 0.8/0.9 adds a 24-profile architecture matrix spanning Tiny,
Medium, and Large Decoder dimensions, Prefill lengths 128/512/2048, Decode KV
lengths 128/512/2048/4096, and Top-2 MoE routing with 8/16 experts. The checked-
in run uses a 1.2 GFLOP and 256 MiB weight budget: 12 rows execute on the real
CPU and 12 remain explicitly `compile-only`. Compile-only rows report zero
runtime latency and are not simulator estimates.

Representative real CPU results on the recorded Intel Core i5-14600K are:

- Tiny Prefill `S=128`: **3.753 ms**, **34,110 token/s**
- Tiny Decode `KV=512`: **1.443 ms**, **693 token/s**
- Medium Decode `KV=4096`: **12.817 ms**, **78 token/s**
- Tiny 8-expert Top-2 MoE, uniform routing: **8.509 ms**, **3,761 token/s**
- Tiny 16-expert Top-2 MoE, heavy skew: **13.445 ms**, **2,380 token/s**

All measured matrix rows remain below `1e-3` maximum absolute error. The MoE
Decoder path passes normalized activations separately from immutable expert
parameters; removing the previous per-call expert-weight copy reduces the
uniform 8-expert layer from roughly 32.6 ms to 8.5 ms in the recorded runs.

`ExecutablePlanOptimizer` jointly searches Attention strategy, intermediate
layout, materialization, workspace reuse, schedule family, thread count, and
core placement. Analytical scores prioritize the measurement budget, but only
full Decoder execution selects a winner. All candidates receive equal warmup;
an apparent winner is checked in three interleaved baseline/winner rounds.

- Tiny Prefill `S=128`: default plan retained, **1.000x**
- Tiny Decode `KV=512`: one-thread non-materializing Split-KV plan selected,
  confirmed **1.400x** over the default plan

The raw matrix, winner summary, and complete candidate table are stored in
`results/decoder_realistic.csv`, `results/decoder_plan_optimizer.csv`, and
`results/decoder_plan_optimizer_candidates.csv`.

## Production LLVM and Fused Attention CodeGen

SchedForge 0.10 makes the same-LoopIR backend comparison fairer by carrying the
Scheduled LoopIR thread count into LLVM ORC execution. Workers receive disjoint,
MR-aligned row partitions; irregular tails use a separately compiled safe
specialization. The assembly report now includes total/vector/FMA instructions,
loads, stores, branches, address-generation proxies, stack accesses, and vector
spill detection.

The recorded 8-thread cubic matrix is:

| Shape | Native GFLOPS | LLVM GFLOPS | Native / LLVM | Error |
|---|---:|---:|---:|---:|
| 192³ | 238.172 | 111.525 | 2.136x | 0.000 |
| 256³ | 265.845 | 146.711 | 1.812x | 0.000 |
| 512³ | 367.069 | 144.773 | 2.535x | 0.000 |

This is a substantial improvement over the previous 31.216 GFLOPS LLVM record,
but it is not parity. The raw per-run values and assembly categories are stored
in `results/llvm_codegen_study.csv`.

Version 0.10 also lowers the complete exact online-softmax Attention algorithm
into one LLVM function. The function performs QK, causal bounds, running max and
denominator rescaling, PV numerator accumulation, and final normalization. It
does not materialize the score/probability matrix and executes MHA/GQA rows in
parallel.

| Profile | Native P50 | Fused LLVM P50 | LLVM speed fraction | Error |
|---|---:|---:|---:|---:|
| MHA Prefill S=128 | 0.340 ms | 0.791 ms | 0.430 | <5e-8 |
| GQA Prefill S=128 | 0.255 ms | 0.786 ms | 0.324 | <5e-8 |
| GQA Decode KV=1024 | 0.099 ms | 0.214 ms | 0.464 | <5e-8 |

The fused function is correct and executable, but it remains 2.1-3.1x slower
than the specialized native path and its emitted assembly still shows a vector
spill pattern. This negative result is retained in
`results/fused_attention_llvm.csv` and is the primary remaining v0.10 code-
quality boundary.

## Target-Specific AOT Deployment

SchedForge 0.11 adds a real machine-code deployment path for Scheduled LoopIR.
The optimized LLVM module is emitted as a PIC ELF `kernel.o`, linked into
`kernel.so`, and packaged with `manifest.sfe`, canonical LoopIR, LLVM IR, and
assembly in an inspectable `.sfe` directory. A separate process validates the
format, ABI, exact shape, target triple, target CPU, and checksums before loading
`schedforge_matmul_v1` with `dlopen`/`dlsym`. No LLVM compilation occurs in the
runtime load or execution path.

The recorded one-thread cubic study is:

| Shape | JIT compile ms | AOT compile ms | AOT link ms | AOT load ms | AOT run ms | Error |
|---|---:|---:|---:|---:|---:|---:|
| 64³ | 264.371 | 268.110 | 38.063 | 0.073 | 0.007 | 1.43e-6 |
| 128³ | 12.388 | 10.608 | 44.346 | 0.064 | 0.137 | 3.34e-6 |
| 256³ | 12.208 | 11.657 | 41.420 | 0.081 | 0.844 | 5.72e-6 |

The first LLVM initialization dominates the 64³ compile row. The ORC execution
measurement also includes its current host worker-thread wrapper even when the
LoopIR requests one thread, while AOT invokes the loaded function directly.
Therefore the execution difference measures current deployment/runtime overhead
as well as code, and is not attributed solely to object emission. Raw values are
stored in `results/aot_deployment.csv`.

## v0.12-v0.17 Runtime Milestones

The runtime line is implemented as executable slices and checked by the same
Release and sanitizer suites:

| Slice | Implemented evidence | Current boundary |
|---|---|---|
| v0.12 Paged KV | Page table, physical pages, append/recycle, direct page-aware online-softmax Decode, error check | Gather remains inspection-only |
| v0.13 INT8 | Per-channel weight-only INT8 MatMul, FP32 accumulation, bias/ReLU, error check | Dense Decoder integration is covered; MoE expert weights remain FP32 |
| v0.14 Transfer | Real memcpy chunk/worker search with destination validation and bandwidth | Host-memory copy tuning, not NUMA/RDMA |
| v0.15 NEON | ARM NEON intrinsic source and compile-time capability report | Current x86_64 host cannot execute ARM code |
| v0.16 Fuzzing | Random LoopIR generation, rejection accounting, numerical invariants, CTest CLI | libFuzzer corpus minimization remains future work |
| v0.17 Closure | Dense Decoder INT8 Q/K/V/O/Gate/Up/Down, prefill and KV-cache Decode; direct paged traversal; AArch64 syntax compile | ARM runtime performance still requires an ARM host |

The checked-in `results/next_milestones.csv` records 16 logical KV tokens in
two physical pages, direct paged-attention numerical error, INT8 error below
`5e-7` on the study shape, positive measured transfer bandwidth, and successful
AArch64 NEON syntax compilation from the x86_64 host. The deterministic fuzz
smoke completes with zero invariant failures.

## MoE Compiler Validation

SchedForge 0.4 adds a single-host FP32 Top-2 MoE MLP compiler/runtime path. MoE
is decomposed into Router MatMul, Softmax, TopK, Histogram, Prefix Sum, stable
token dispatch, Segmented Tensor IR, variable-M grouped W1/W3 GEMM, SwiGLU,
variable-M W2 GEMM, and weighted combine.

The requested full MVP configuration, `T=128, H=512, I=2048, E=8, TopK=2`,
ran on the recorded Intel Core i5-14600K with:

- Tensor SSA operations: 18
- MoE Routing/Expert IR operations: 11
- routed assignments: 256
- compiled token buckets: `M <= 4`, `M <= 16`, `M <= 64`, `M <= 128`
- P50 end-to-end execution: **35.843 ms**
- P95 end-to-end execution: **37.789 ms**
- maximum absolute error: **0.000** at printed precision

The checked-in 27-case full-shape experiment covers three routing
distributions, three expert execution strategies, and three task schedulers.
For `T=128, H=512, I=2048` under heavy skew:

- fixed grouped simulated imbalance: 2.0
- load-aware split simulated imbalance: 0.0
- fixed grouped P50: 26.780 ms
- load-aware grouped P50: 20.113 ms

This is a host-specific systems result, not a universal speedup claim. The full
matrix is stored in `results/moe_strategy_matrix.csv`; the executable artifact
and representative run are `results/moe_mlp.sfe` and
`results/moe_mlp_run.txt`.

## CPU Flash-Style Attention Validation

SchedForge 0.5 adds structural SDPA fusion and four exact FP32 CPU algorithms:
Materialized, Tiled Materialized, IO-aware online-softmax prefill, and Split-KV
decode. Attention is represented by explicit TilePipelineIR operations for QK,
scale, causal mask, row max/sum reductions, vector exp, online rescaling, PV,
and final division. QK/PV scheduled LoopIR and LLVM ORC artifacts are embedded
in each `.sfe` plan.

Representative real executions on the recorded Intel Core i5-14600K:

- causal MHA prefill `B=1, H=8, Sq=Sk=128, D=64`: **0.262 ms P50**
- causal GQA decode `Hq=8, Hkv=2, Sq=1, Sk=1024, D=64`: **0.112 ms P50**
- maximum printed validation error for both: **0.000**

The checked-in 12-case measured matrix compares four algorithms at sequence
lengths 128, 256, and 512. At `B=1, H=8, S=512, D=64`:

- Materialized temporary footprint: **16 MiB**
- Auto-scheduled IO-aware temporary footprint: **49 KiB**
- Materialized P50: **18.929 ms**
- Auto-scheduled IO-aware P50: **2.947 ms**

The 96-case analytical matrix covers heads `8/12`, dimensions `64/128`, and
sequence lengths `128, 256, 512, 1024, 2048, 4096`. At the largest
`H=12, S=4096, D=128` point, the materialized intermediate is 1.5 GiB while the
fixed IO-aware concurrent tile state is 194 KiB. These long-sequence rows are simulator/
IO-analysis results, not claimed hardware latencies.

A Linux `perf` snapshot over 200 measured `S=512` executions records P-core IPC
2.561, L1D miss rate 1.066%, and cache-reference miss rate 16.230%. These are
process-level counters and include runtime/framework overhead. Raw evidence is
stored in `results/attention_pmu.txt`; plans and runs are in
`results/attention_prefill.*` and `results/attention_decode.*`.

## Simulator Boundary

The simulator now consumes explicit LoopIR and traverses the full problem by
default. Bounded studies must request sampling explicitly and record the sampled
extent. It remains useful for explaining predicted cache misses, DTLB
behavior, bandwidth traffic, and register pressure. It is not used to decide
which candidates receive hardware execution, and it is not used to select the
winner. Prediction studies remain research diagnostics only.

The refreshed 192³ resolution study measured 12 simulator-leading candidates
over 20 randomized rounds. All 11 adjacent predicted pairs were tied in the
model. Confidence intervals overlapped for 86.4% of candidate pairs with
topology pinning and 51.5% without it; the predicted first candidate had 192.9%
and 92.7% median regret respectively. These are negative results for simulator-
only selection and are preserved in `results/top_resolution_*`.

## Claim Boundaries

- Results belong to this Intel Core i5-14600K host and current frequency/load state.
- SchedForge is a compiler prototype, not a replacement for BLIS or oneDNN.
- AVX2 is runtime-validated; NEON is AArch64 syntax-validated, while AVX-512
  and ARM runtime performance remain unvalidated on this host.
- The machine has one NUMA node, so cross-node performance is not validated.
- MoE quantization, heterogeneous-core placement, NUMA scheduling, block-sparse
  lowering, and distributed expert parallelism are outside this release.
- Attention is a CPU cache-hierarchy Flash-style lowering, not GPU
  FlashAttention-2. Paged KV direct traversal is implemented, while
  reduced-precision attention, backward/dropout, distributed
  attention, and native-parity spill-free fused LLVM code remain outside this release.
- Dense Decoder Layer execution supports FP32 and weight-only INT8 projections
  with FP32 accumulation. BF16 and MoE expert quantization remain future work.
- AOT format v1 is same-target, shape-specialized FP32 MatMul with one runtime
  thread. Whole-graph constant relocation, multi-kernel AOT dispatch, and
  cross-CPU feature compatibility remain future work.
- Large Decoder profiles and expensive Prefill profiles are compile-only under
  the checked-in 1.2 GFLOP/256 MiB execution budget; no latency is claimed for
  those rows.
- Full fresh tuning is deliberately expensive because every deduplicated candidate
  is executed on hardware; cached runs are much faster and are labeled as cache hits.
