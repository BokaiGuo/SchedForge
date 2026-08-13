# Changelog

All notable changes to SchedForge are documented here.

## [0.5.0] - 2026-08-13

### Added

- SDPA graph construction and structural `AttentionFusionPass` recognition.
- Materialized, tiled-materialized, IO-aware online-softmax, and Split-KV
  decode lowerings for FP32 CPU attention.
- `TilePipelineIR` for fused QK, scale, causal mask, reductions, online
  softmax rescaling, PV, and normalization.
- MHA, GQA, and MQA layouts with dynamic `Sq`/`Sk` guards.
- Growable contiguous KV cache and parallel Split-KV partial-state merging.
- Cache-aware attention schedule selection and real-hardware tile tuning.
- Attention IO/arithmetic-intensity simulator, 12-case measured strategy
  matrix, 96-case scaling analysis, and Linux PMU snapshot.
- `schedforge-attention` compiler/runtime CLI and serialized attention `.sfe`
  plans containing QK/PV LoopIR and LLVM ORC artifacts.

### Changed

- SchedForge now has three executable flagship paths: Dense MLP, sparse MoE,
  and exact CPU Flash-style attention.
- The cost-model scope now includes algorithm selection and intermediate
  materialization cost in addition to GEMM schedule ranking.

## [0.4.0] - 2026-08-13

### Added

- Decomposed FP32 MoE Tensor SSA and Routing IR for Router, Softmax, TopK,
  Histogram, Prefix Sum, Dispatch, Grouped GEMM, SwiGLU, and Weighted Combine.
- Segmented Tensor IR for runtime-variable expert batches.
- Token-bucketed variable-M expert LoopIR and LLVM artifacts.
- Independent, grouped, and bucketed-grouped execution strategies.
- Fixed, work-stealing, and load-aware split expert task schedulers.
- Routing-trace and target-aware automatic execution-strategy selection.
- Routing-trace simulator and uniform/moderate/heavy skew workloads.
- `schedforge-moe` compiler/runtime CLI and 27-case strategy experiment matrix.

### Changed

- SchedForge now supports dense and dynamically routed sparse tensor programs.
- Runtime plans can guard dynamic token counts and specialize LoopIR by expert
  token bucket without reconstructing Schedule objects.

## [0.3.0] - 2026-08-13

### Added

- Explicit executable LoopIR operations for loops, parallel regions, packing,
  prefetching, scalar/vector loads, accumulator initialization, broadcast, FMA,
  graph epilogues, and scalar/vector stores.
- LoopIR verification and cached read-only execution analysis.
- Full-size simulation by default with explicit, labeled sampling controls.
- Compiled Scheduled LoopIR embedded per dispatch in `.sfe` executable plans.

### Changed

- Schedule application now performs an IR rewrite instead of storing Schedule
  fields inside LoopIR.
- Native execution, the cache/TLB simulator, LLVM textual lowering, and LLVM
  ORC accept the same explicit LoopIR as their executable input.
- Auto-tuning precompiles candidate LoopIR before hardware timing, so rewrite
  overhead is excluded from kernel latency.
- Transformer MLP GELU and residual epilogues are explicit runtime IR operations.

## [0.2.0] - 2026-08-13

### Added

- StableHLO subset importer for tensor graph programs.
- Multi-operation Tensor SSA graph, symbolic dimensions, shape inference, and guards.
- Structured Tensor Compute, FusionPlanner, Dispatch IR, graph layout propagation,
  bufferization, lifetime-based workspace reuse, and ExecutablePlan artifacts.
- Transformer MLP compilation and runtime validation, plus a mini-attention graph path.
- Transform IR serialization/replay and AVX2 tensor-intrinsic matching.
- Quantized tensor metadata and graph-level quantization propagation.
- Hardware measurement database and hybrid analytical/learned cost model.
- `schedforge-compile` model-to-machine CLI and `.sfe` executable-plan output.
- LLVM ORC register-resident MR×NR micro-kernel generation for legal static shapes.

### Changed

- Project positioning expands from a tensor kernel compiler to a target-aware CPU AI
  compiler with separate graph, kernel, target, and runtime layers.

## [0.1.0] - 2026-08-12

### Added

- SSA Tensor IR, nested Loop IR, IRBuilder, and layered pass managers.
- Multi-level tiling, packing, generated micro-kernel schedules, AVX2, fusion,
  prefetching, threading, affinity, and scalar tails.
- LLVM 18 IR generation, O3 optimization, ORC JIT execution, and assembly analysis.
- Cache/TLB/register/bandwidth model and calibrated auto-scheduling.
- Grid, random, greedy, and evolutionary search strategies.
- Dynamic shapes, kernel cache, layout propagation, memory planning, BF16, and INT8.
- Bilingual project documentation and GitHub Actions CI.
