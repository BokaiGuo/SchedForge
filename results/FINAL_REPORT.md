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

The existing LLVM ORC JIT result for fused 192³ remains 31.216 GFLOPS. It is a
separate code-generation path and did not receive the native intrinsic
micro-kernel rewrite in this increment.

BF16 and INT8 paths remain correctness-oriented reference kernels rather than
ISA-specialized AVX-512 BF16 or VNNI implementations.

## Simulator Boundary

The simulator is still useful for explaining predicted cache misses, DTLB
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
