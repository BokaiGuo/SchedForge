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

SchedForge 0.3 adds an explicit executable-IR backbone to the model-to-machine
Transformer MLP path. The checked-in
StableHLO example is canonicalized to 12 Tensor SSA operations and compiled into
two dispatches: `MatMul + Bias + GELU` and `MatMul + Bias + Residual`.

For the recorded `batch=1, sequence=16, hidden=64, intermediate=128` run:

- imported/canonical operations: 12
- fused dispatches: 2
- propagated inter-dispatch layout: `blocked<6x16>`
- naive intermediate memory: 32,768 bytes
- planned workspace: 8,192 bytes
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
- AVX2 is validated; AVX-512 and NEON remain target abstractions on this host.
- The machine has one NUMA node, so cross-node performance is not validated.
- Full fresh tuning is deliberately expensive because every deduplicated candidate
  is executed on hardware; cached runs are much faster and are labeled as cache hits.
