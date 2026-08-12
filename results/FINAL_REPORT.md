# SchedForge Final Experiment Report

**Experiment date:** August 12, 2026

## Verdict

The requested CPU Tensor Compiler architecture is implemented end to end rather
than as a collection of independent MatMul kernels. Tensor/Loop IR, scheduling,
packing, micro-kernel execution, LLVM ORC JIT, hardware simulation, runtime,
auto-tuning, dynamic shapes, data types, calibration, and research evaluation
all have executable code paths and tests.

## Final Native Auto-Schedule

Workload: fused FP32 `MatMul + Bias + ReLU`, `192 x 192 x 192`.

- Calibrated target bandwidth during this run: 12.579 GB/s
- Generated schedules: 10,368
- Statically valid schedules: 3,564
- Hardware-benchmarked candidates: 12
- Selected schedule: `i-k-j`
- Outer tile: `64 x 128 x 64`
- L1 tile: `16 x 64 x 64`
- Register block: `8 x 8`
- Vector width: 8 FP32 lanes
- Threads: 8
- Packing: disabled
- Median execution: 0.167 ms
- Throughput: **84.584 GFLOPS**
- Maximum absolute error: below `1e-3`
- Simulator rank of hardware winner: 2

## LLVM ORC JIT

For the recorded 192³ run:

- Compile time: 13.906 ms
- Median execution: 0.453 ms
- Throughput: **31.216 GFLOPS**
- Maximum absolute error: `4.77e-6`
- Assembly contains AVX2 vector FMA (`vfmadd213ps`)
- No vector stack-spill pattern detected
- The second identical compilation uses the process-local JIT cache

## Search and Prediction

The final crossover/prediction study reports:

- Spearman schedule-ranking correlation: **0.968**
- Top-5 recall: **0.40**
- Top-10 recall: **0.70**

With 12 hardware measurements on the 128³ comparison workload:

| Strategy | Best GFLOPS |
|---|---:|
| Cost-model grid ranking | 39.84 |
| Random | 28.54 |
| Greedy neighborhood | 37.26 |
| Evolutionary | 36.52 |

The simulator is useful for ranking but not perfect. The Top-5 result remains a
clear limitation and motivates calibration/prefetch/parallel-model refinement.

## Packing Result

Packing is fully implemented with MR/NR-oriented panels and multi-level tiling,
but it is not automatically claimed as a speedup. On this workload family the
scheduler generally chose no packing; copy and panel traversal costs exceeded
reuse benefits. At 32³ PackB occasionally matched or slightly exceeded the
unpacked path, while larger tested squares favored the contiguous unpacked
`i-k-j` kernel. This negative result is preserved because packing must remain a
compiler decision rather than a hard-coded optimization.

## Data Types

Recorded 128³ runs:

| Data type | Time | Throughput | Error vs FP32 |
|---|---:|---:|---:|
| BF16 inputs / FP32 accumulation | 0.824 ms | 5.09 GOPS | 0.0303 |
| INT8 inputs / INT32 accumulation | 0.786 ms | 5.33 GOPS | 0.0805 |

These are correctness-oriented reference kernels, not ISA-specialized VNNI or
AVX-512 BF16 implementations.

## Hardware Counters

Linux `perf` was enabled for user-space counters. The recorded LLVM 256³ run
contains core/atom cycles, instructions, cache references/misses, branches, and
branch misses in `final_perf_llvm.csv`. Hybrid P-core/E-core aggregation makes a
single IPC or miss-rate number potentially misleading, so raw per-PMU values
are retained instead of presenting an oversimplified combined metric.

## Requirements Matrix

| Requirement | Status |
|---|---|
| SSA Value/Operation/Block/Module/IRBuilder | Implemented and tested |
| Tensor IR / Loop IR separation | Implemented |
| Tensor and Loop PassManagers | Implemented |
| Canonical lowering and schedule transforms | Implemented |
| Multi-level tiling | Implemented in native backend |
| PackA / PackB decisions | Implemented and searched |
| Generated MR/NR micro-kernel shapes | Implemented |
| Register-pressure and spill model | Implemented and assembly-checked |
| AVX2 explicit vectorization and tails | Implemented |
| LLVM 18 IRBuilder/O3/ORC JIT | Implemented and executed |
| Assembly generation and analysis | Implemented |
| Cache associativity / LRU | Implemented |
| TLB and prefetch simulation | Implemented |
| Bandwidth and hardware calibration | Implemented |
| Grid/random/greedy/evolutionary search | Implemented and compared |
| Schedule DSL | Implemented |
| Bias/ReLU fusion | Implemented in execution and IR |
| Layout propagation | Implemented |
| Lifetime memory planning/reuse | Implemented |
| Dynamic-shape specialization | Implemented |
| Persistent kernel/schedule/IR cache | Implemented |
| FP32/BF16/INT8 paths | Implemented |
| ISA abstraction | Scalar, AVX2 implemented; AVX-512/NEON represented |
| Thread affinity / NUMA topology | Implemented; single-node validation only |
| Performance portability | Architecture supported; one physical CPU validated |
| Prediction error analysis | Implemented with correlation and top-k recall |
| Linux perf validation | Implemented and recorded |

## Validation

- Release build completes without compiler warnings.
- All CTest tests pass.
- ASan, UBSan, and leak detection pass.
- Tests cover non-tile and non-vector-divisible dimensions.
- Every hardware candidate is checked against a scalar FP32 reference.
- Real LLVM ORC execution and generated assembly are validated.

## Claim Boundaries

- This is a CPU Tensor Compiler prototype, not a replacement for oneDNN/BLIS.
- The native micro-kernel is generated from schedule parameters but not yet an
  LLVM-generated packed MR-by-NR kernel.
- AVX-512 and NEON are target abstractions, not validated backends on this host.
- NUMA-aware execution is implemented at topology/partition/affinity level but
  cross-node performance cannot be measured on a single NUMA node.
- Performance portability across CPUs requires additional machines.
