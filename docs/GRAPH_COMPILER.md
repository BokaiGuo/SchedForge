# Graph Compiler and Executable Format

## Tensor SSA

`TensorGraph` owns immutable tensor values, producer/user links, operation
attributes, tensor types, symbolic dimensions, layouts, and quantization
metadata. The importer recognizes the StableHLO operations needed for MLP and
early Transformer graphs, including an explicit `custom_call @Gelu` boundary.

## Fusion and Dispatch

Fusion is not a boolean schedule flag at graph level. `FusionPlanner` requires a
single-use producer-consumer edge and supported epilogue semantics. It records a
`FusionCost` containing saved materialization bytes, estimated register
pressure, and a profitability score. A Transformer MLP becomes two dispatches:

```text
dispatch #0: MatMul + Bias + GELU
dispatch #1: MatMul + Bias + Residual
```

## Bufferization

Internal values inside a fused dispatch remain virtual tensors. Only graph
inputs/constants, dispatch boundaries, and returned values become physical
buffers. The planner assigns aligned workspace offsets from lifetime intervals.
For the checked-in 1×16×64 MLP example, naive intermediates require 32 KiB while
the executable workspace requires 8 KiB.

## Transform IR

Every selected schedule is converted into a replayable transformation program:

```text
transform.sequence {
  match tensor.matmul
  tile [32,64,32]
  interchange [0,2,1]
  register_tile [4,8]
  vectorize width=8
  unroll factor=1
  parallelize threads=4
  tensorize avx2_f32_m4n8
}
```

`TransformProgram::parse` and `replay` reconstruct the payload schedule, while
`TransformProgram::apply` directly lowers that program into verified executable
LoopIR. Graph compilation uses this path, so Transform IR is no longer only a
serialized description beside an independently-created kernel.

## Guarded Specialization

`ExecutablePlan::specializeMLP` resolves the symbolic `B*S` dimension and then
recomputes shape inference, buffer sizes, Dispatch kernel problems, Scheduled
LoopIR, exact guards, and LLVM artifacts. The resulting plan is independently
executable; the guard is not metadata attached to a static fallback kernel.

## Measurement Database

`MeasurementDatabase` stores real hardware latency records keyed by problem and
schedule, including fused Bias/ReLU semantics. `GraphCompiler` can load this
database and select the fastest exact workload match before Transform IR lowering. The Dispatch records whether its
schedule came from defaults, a measurement database, the auto-tuning cache, or a
new hardware auto-tuning run.

## `.sfe` ExecutablePlan

The text `.sfe` artifact contains:

- canonical Tensor SSA graph
- Structured Tensor Compute descriptions
- fused Dispatch IR
- Transform IR and tensor intrinsic selection
- explicit Scheduled LoopIR per dispatch
- layout and buffer plans
- dynamic shape guards
- LLVM IR kernel artifacts

The current runtime executes the Transformer MLP plan through the compiled
Scheduled LoopIR selected by hardware auto-tuning or the measurement database.
GELU and residual are explicit LoopIR epilogues rather than hand-written wrapper
steps. LLVM ORC kernels now lower those same graph epilogues from the final
Scheduled LoopIR and are embedded in the plan, but the graph runtime does not yet
load embedded kernels back from a parsed `.sfe` file.
Other imported graphs can already be compiled and serialized even when a
graph-specific runtime executor is not yet available.
