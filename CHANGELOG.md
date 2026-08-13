# Changelog

All notable changes to SchedForge are documented here.

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
