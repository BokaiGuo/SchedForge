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

`TransformProgram::parse` and `replay` reconstruct the payload schedule. This is
also the serialization boundary used by the executable plan and future schedule
mutation/search work.

## `.sfe` ExecutablePlan

The text `.sfe` artifact contains:

- canonical Tensor SSA graph
- Structured Tensor Compute descriptions
- fused Dispatch IR
- Transform IR and tensor intrinsic selection
- layout and buffer plans
- dynamic shape guards
- LLVM IR kernel artifacts

The current runtime executes the Transformer MLP plan through the native
scheduled-loop dispatch path selected by hardware auto-tuning. LLVM ORC kernels
are generated and validated during compilation and embedded in the plan, but
the graph runtime does not yet load those embedded kernels back from `.sfe`.
Other imported graphs can already be compiled and serialized even when a
graph-specific runtime executor is not yet available.
