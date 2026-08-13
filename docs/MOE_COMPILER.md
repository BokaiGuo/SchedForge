# MoE Compiler Pipeline

SchedForge 0.4 treats Mixture-of-Experts as a dynamically routed tensor program,
not as one opaque operator. The implemented FP32 single-host path is:

```text
Tensor SSA
  -> Router MatMul
  -> Softmax + TopK
  -> Histogram + Prefix Sum
  -> Stable Token Dispatch
  -> Segmented Tensor
  -> Variable-M Grouped Expert GEMM (W1/W3)
  -> SwiGLU
  -> Variable-M Grouped Expert GEMM (W2)
  -> Weighted Combine
```

## Segmented Tensor IR

Dispatched activations use:

```text
!sfg.segmented_tensor<values=tensor<?xHxf32>, segments=E>
```

The physical form contains contiguous values, `E + 1` offsets, original token
IDs, route slots, and routing weights. Segment `e` spans
`offsets[e]..offsets[e + 1]`, so its batch dimension is a runtime value.

The runtime allocates W1/SwiGLU and W3 activation arenas once per invocation and
hands disjoint segment slices to expert tasks. This makes the reported workspace
plan correspond to real execution instead of per-task hidden allocations.

## Expert Kernel Specialization

The compiler emits LoopIR variants for token-count buckets. The current default
is `M <= 4`, `M <= 16`, `M <= 64`, plus the configured maximum token count.
Each bucket contains W1/W3 and W2 LoopIR plus LLVM ORC-compiled vector-FMA
artifacts that are executed and checked against the scalar reference during
plan compilation. Runtime specialization changes only dynamic `M`; it does not
reconstruct a Schedule.

## Execution Strategy IR

`MoeExecutionSchedule` records model-level choices:

- independent, grouped, or bucketed-grouped expert execution
- fixed expert placement, work stealing, or load-aware task splitting
- token-count buckets and split threshold
- router/TopK, histogram/prefix, and combine-weight fusion decisions
- runtime worker count

`select_moe_schedule` evaluates the strategy/scheduler cross product against the
current routing trace and target worker limit. This provides an automatic
compiler/runtime policy in addition to explicit CLI overrides.

Large expert segments can be split into multiple tasks. Grouped strategies run
those tasks concurrently; independent execution remains a serial baseline.

## Simulation and Experiments

`simulate_moe` consumes a `RoutingTrace` and reports expert counts, task count,
load imbalance, worker utilization, dispatch traffic, weight traffic, and an
estimated critical-path makespan. Traces can come from real router data or
uniform, moderate-skew, and heavy-skew generators.

`schedforge-moe --experiment-csv=...` runs the full 3 x 3 x 3 matrix across:

- three routing distributions
- independent, grouped, and bucketed-grouped execution
- fixed, work-stealing, and load-aware scheduling

The checked-in experiment demonstrates that the best policy is workload
specific. On the recorded host and full `T=128, H=512, I=2048` workload,
load-aware splitting materially improves heavy-skew grouped execution. The
matrix records measured P50/P95 latency as well as the simulator's explanatory
load metrics; scheduling decisions are validated by real execution rather than
simulator-only scores.

## Claim Boundaries

Implemented and validated:

- FP32, single-host Top-2 MoE MLP
- dynamic token count guarded by a compiled maximum
- Router, Softmax, TopK, Histogram, Prefix Sum, Dispatch, Segmented Tensor,
  variable-M expert GEMM, SwiGLU, and Weighted Combine
- real CPU execution and numerical validation
- strategy and routing-skew experiments with P50/P95 latency

Not yet claimed:

- BF16/INT8 expert kernels
- P-core/E-core-specific placement or NUMA-aware execution
- distributed expert parallelism and communication IR
- block-sparse MegaBlocks-style lowering
- production-grade fused router/TopK or fused grouped-GEMM machine code
- sparse-to-dense MoE densification decisions
