# Changelog

All notable changes to SchedForge are documented here.

## [0.12.0-0.16.0] - 2026-08-20

### Added

- v0.12 Paged KV page tables, physical-page allocation, boundary-safe append,
  truncate/recycle, logical gather, active-page release guards, and exact decode integration.
- v0.13 per-output-channel INT8 weight-only MatMul with FP32 accumulation,
  bias/ReLU epilogues, reference validation, and measured study output.
- v0.14 real host-memory transfer tuning over chunk sizes and worker counts,
  with copy validation and bandwidth reporting.
- v0.15 ARM NEON intrinsic source generation and compile-time capability probe.
  Non-ARM hosts report unavailable rather than claiming NEON execution.
- v0.16 deterministic LoopIR fuzzing with rejection/failure accounting,
  numerical invariants, standalone CLI, and CTest coverage.

### Evidence Boundaries

- The current host is x86_64: NEON source generation is implemented, but ARM
  execution is not claimed without an ARM build/host.
- INT8 currently covers weight-only MatMul, not the complete Decoder graph.
- Paged decode currently gathers pages into the existing exact attention kernel;
  direct page-aware tiled traversal remains a future optimization.

## [0.11.0] - 2026-08-16

### Added

- LLVM 18 `TargetMachine` ELF object emission from the same optimized Scheduled
  LoopIR module used by ORC JIT and assembly analysis.
- Versioned target-specific `.sfe` deployment packages containing a manifest,
  shape and ABI guards, LoopIR, LLVM IR, assembly, `kernel.o`, and `kernel.so`.
- `schedforge-aot compile|inspect|run` with checksum validation, host target
  checks, POSIX `dlopen`/`dlsym`, and numerical verification without runtime
  LLVM compilation.
- Separate-process CTest coverage plus `schedforge-aot-study` measurements for
  cold JIT compilation, AOT compilation/linking, load latency, execution, and
  correctness.

### Changed

- `.sfe` now has a real machine-code deployment form in addition to the older
  text ExecutablePlan serializers; the AOT form is a directory with a `.sfe`
  suffix so artifacts remain inspectable.
- AOT format v1 is deliberately shape-specialized, same-target, FP32 MatMul,
  and single-threaded. Unsupported thread dispatch and target mismatches fail
  explicitly rather than silently changing semantics.

## [0.10.0] - 2026-08-13

### Added

- Parallel LLVM ORC execution that preserves Scheduled LoopIR thread counts and
  uses MR-aligned row partitions with correct irregular tails.
- Expanded machine-code quality reports covering total/vector/FMA instructions,
  loads, stores, branches, address-generation proxies, stack accesses, and
  vector spill patterns.
- `schedforge-codegen-study` for measured same-LoopIR Native-versus-LLVM and
  native-versus-fused-Attention studies with checked-in CSV evidence.
- One executable LLVM Attention function containing QK, causal masking through
  legal key limits, exact online max/denominator rescaling, PV accumulation,
  and final normalization for MHA and GQA.

### Changed

- LLVM MatMul now consumes LoopIR parallel semantics instead of silently
  benchmarking the same scheduled program with one worker.
- The recorded 192/256/512 cubic study reaches 103-153 GFLOPS in LLVM versus
  238-367 GFLOPS native; the remaining 1.8-2.5x gap is preserved explicitly.
- Fused LLVM Attention validates below `5e-8` error but remains 2.1-3.1x slower
  than the specialized native IO-aware/Split-KV runtime and still shows a
  vector spill pattern in emitted assembly.

## [0.9.0] - 2026-08-13

### Added

- `ExecutablePlanOptimizer` for cross-dispatch Decoder planning across
  attention algorithms, intermediate layouts, materialization, workspace
  reuse, schedule families, thread counts, and core placement.
- Measurement-budgeted end-to-end candidate selection with an explicit default
  plan baseline, equal candidate warmup, and three-round interleaved
  baseline/winner confirmation.
- Serializable Decoder plan policy and plan-cost metadata in `.sfe` artifacts.

### Changed

- Plan selection is evaluated by full Decoder latency rather than the sum of
  independently selected kernel scores.
- The checked-in optimizer study preserves the Tiny Prefill negative result and
  records a confirmed 1.400x Tiny Decode improvement from a one-thread,
  non-materializing Split-KV plan.
- Decoder-to-MoE execution passes normalized activations separately from expert
  parameters, eliminating per-invocation copies of 67-132 MiB expert weights.

## [0.8.0] - 2026-08-13

### Added

- A 24-profile realistic Decoder suite spanning Tiny, Medium, and Large model
  dimensions, Prefill lengths 128/512/2048, Decode KV lengths
  128/512/2048/4096, and 8/16-expert Top-2 MoE routing skew.
- Real KV-cache Decoder execution and per-stage timing for Attention,
  projections, RMSNorm/RoPE, FFN/MoE, residuals, and dispatch overhead.
- Peak workspace, compile time, LLVM JIT time, memory-planning time, stage
  percentages, equivalent tokens/s, estimated FLOPs, and weight bytes.
- `schedforge-decoder-bench` with explicit `measured`, `compile-only`, and
  `skipped` evidence classes.

### Changed

- Nested Attention and MoE correctness references no longer contaminate
  measured Decoder latency; end-to-end validation remains at the Decoder seam.
- Large and expensive Prefill rows are retained as compile-feasibility evidence
  instead of being reported as simulated or fabricated runtime latency.

## [0.7.0] - 2026-08-13

### Added

- Full Llama/Mistral-style Decoder Layer compilation from one StableHLO graph
  into one serializable `DecoderExecutablePlan` and one runtime invocation.
- Tensor SSA operations for RMSNorm, RoPE, SiLU, split/concat, fused QKV
  projection, and fused Gate-Up projection.
- Structural QKV, Gate-Up, and RoPE recognition independent of temporary names.
- Compile-time QKV and Gate-Up constant concatenation/packing with serialized
  constant-specialization metadata.
- Dense SwiGLU and Top-K MoE FFN branches behind the same Decoder Layer API.
- `schedforge-decoder`, `examples/decoder_layer.mlir`, end-to-end Dense/MoE
  validation, ASan coverage, and CI artifact checks.

### Changed

- The StableHLO importer now parses multiline function arguments correctly.
- SchedForge's flagship workload is now a complete executable Transformer
  Decoder Layer rather than disconnected MLP, MoE, and Attention demos.

## [0.6.0] - 2026-08-13

### Added

- Executable MLP shape specialization that rewrites Tensor SSA dimensions,
  buffer plans, Dispatch problems, Scheduled LoopIR, guards, and LLVM artifacts.
- Direct `TransformProgram::apply` lowering from replayable Transform IR to
  verified executable LoopIR.
- Measurement-database schedule selection through `--measurement-db`, with the
  selected tuning source serialized into Dispatch IR.
- StableHLO `constant` import with literal preservation in Tensor SSA.

### Changed

- Graph compilation now lowers kernels from Transform IR instead of bypassing
  the transformation program with a parallel schedule-to-loop call.
- LLVM ORC kernels consume the final Scheduled LoopIR graph epilogues, including
  GELU and residual addition, through the same executable IR used by native code.
- Dynamic-shape guards are no longer metadata-only for Transformer MLP plans;
  runtime specialization produces a concrete executable plan.

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
