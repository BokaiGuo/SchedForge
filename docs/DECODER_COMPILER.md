# Decoder Layer Compiler

SchedForge 0.9 promotes the complete Transformer Decoder Layer to the primary
model-to-machine compilation unit. The public contract is deliberately larger
than an operator benchmark: one StableHLO module is imported into one Tensor SSA
graph, compiled into one `DecoderExecutablePlan`, and executed through one
runtime API.

## Supported Architecture

```text
Input
  → RMSNorm
  → Fused QKV Projection
  → Q/K/V Split
  → RoPE(Q, K)
  → exact causal MHA/GQA/MQA attention
  → Output Projection
  → Residual Add
  → RMSNorm
  → Fused Gate-Up + SwiGLU + Down Projection
      or Router + TopK + Grouped Expert SwiGLU + Combine
  → Residual Add
  → Output
```

The Dense and MoE branches share the attention and residual prefix. They differ
only at the second normalized activation, so callers use the same
`DecoderConfig`, `DecoderData`, compiler, plan serialization, and execution API.

## Graph and Fusion IR

The Tensor SSA vocabulary adds `tensor.rms_norm`, `tensor.rope`, `tensor.silu`,
`tensor.concat`, `tensor.split`, `tensor.fused_qkv`, and
`tensor.fused_gate_up`. `DecoderFusionPass` recognizes QKV projections by a
shared use-def source rather than temporary names, recognizes Gate/Up matmuls
feeding SwiGLU, and records RoPE presence. The canonical graph then makes the
selected fused projections explicit.

StableHLO custom calls currently map RMSNorm, RoPE, GELU, SiLU, and SwiGLU into
this graph vocabulary. Multiline function arguments are supported, which is
required by realistic exported modules.

## Constant Specialization

Decoder weights are immutable compile inputs. The compiler concatenates Q, K,
and V columns into one packed projection weight and, for Dense FFN, concatenates
Gate and Up columns into one packed weight. The `.sfe` artifact records each
specialized constant, its shape, and byte count. Runtime execution consumes the
packed vectors directly and does not rebuild them for each invocation.

This release implements compile-time in-process specialization. A portable
object-file AOT cache, weight-file relocation, BF16/INT8 packed kernels, and
cross-process constant deduplication remain future work.

## Lowering and Runtime

Each matrix projection is converted to Structured Compute, scheduled LoopIR,
and an LLVM ORC artifact. Attention is delegated to the exact online-softmax
Attention compiler, including MHA/GQA/MQA layouts and its strategy planner. The
MoE branch delegates routing and expert execution to the MoE compiler. The
Decoder runtime joins these executable components with RMSNorm, RoPE, split/
merge, SwiGLU, and residual semantics in one call.

`reference_decoder_layer` independently evaluates the same architecture with
scalar reference projections plus reference Attention/MoE. End-to-end tests
compare the complete output rather than testing kernels in isolation.

The MoE call boundary passes the second normalized activation separately from
the immutable expert parameter object. This preserves explicit ownership while
avoiding a 67-132 MiB expert-weight copy on every Decoder invocation in the
realistic 8/16-expert profiles.

## Whole-Graph Executable Planning

`ExecutablePlanOptimizer` searches typed cross-dispatch policies rather than
concatenating independently optimal kernels. A policy jointly controls:

- Attention lowering strategy and intermediate layout;
- Attention-output materialization;
- latency- or throughput-oriented schedules;
- workspace reuse;
- thread count and compact/spread placement.

Candidates are ranked analytically, but the winner is selected only by real
end-to-end Decoder latency. All measured candidates receive equal warmup. If a
candidate initially beats the explicit default plan, the optimizer performs
three interleaved baseline/winner confirmations and compares their medians.
Simulator output never selects the measured winner.

The checked-in Tiny study retains the default Prefill plan at `S=128` and
selects a one-thread, non-materializing Split-KV Decode plan at `KV=512`, with a
confirmed 1.400x speedup over the default plan on the recorded host.

## Realistic Benchmark Matrix

`schedforge-decoder-bench` covers 24 architecture profiles:

- Tiny `H=512, I=1376, D=64`;
- Medium `H=1024, I=2816, D=64`;
- Large `H=4096, I=11008, D=128`;
- Prefill `S=128/512/2048`;
- Decode `Sq=1`, `Sk=128/512/2048/4096`;
- Top-2 MoE with 8/16 experts and uniform/moderate/heavy routing skew.

Every row records an evidence class. `measured` rows contain real end-to-end and
stage latency, tokens/s, validation error, workspace, compile time, LLVM JIT
time, and memory-planning time. `compile-only` rows contain zero runtime latency
and retain only feasibility evidence because they exceeded the configured FLOP
or weight-memory budget.

## Command Line

```bash
./build/schedforge-decoder examples/decoder_layer.mlir \
  --batch=1 --sequence=4 --hidden=16 --intermediate=32 \
  --q-heads=4 --kv-heads=2 --head-dim=4 --threads=2 \
  -o results/decoder_dense.sfe

./build/schedforge-decoder examples/decoder_layer_moe.mlir --moe \
  --batch=1 --sequence=4 --hidden=16 --intermediate=32 \
  --q-heads=4 --kv-heads=2 --head-dim=4 \
  --experts=4 --top-k=2 --threads=2 \
  -o results/decoder_moe.sfe

./build/schedforge-decoder-bench --suite=realistic --threads=8 \
  --repetitions=7 --max-real-gflop=1.2 --max-weight-mib=256 \
  --output=results/decoder_realistic.csv

./build/schedforge-decoder-bench --suite=optimizer --threads=8 \
  --repetitions=11 --filter=tiny --optimize \
  --output=results/decoder_plan_optimizer.csv
```

The CLI prints imported operation count, fusion decisions, specialized
constants, memory plan, total/attention/FFN latency, validation error, and target
information. The emitted plan contains both imported and canonical graphs so
fusion decisions remain inspectable.

## Current Boundaries

- Execution is FP32; BF16/INT8 Decoder kernels are not claimed in v0.9.
- RoPE and RMSNorm semantics execute correctly but are not yet one monolithic
  LLVM function with adjacent projections.
- Attention is exact CPU Flash-style online softmax, not GPU FlashAttention-2.
- The MoE branch is single-host and does not implement distributed expert
  parallelism, NUMA placement, or P-core/E-core specialization.
- Large profiles and expensive Prefill rows are compile-only under the checked-in
  resource budget; they are not production-model latency claims.
- JIT time is process-local: repeated shapes may report zero LLVM compile time
  after a cache hit.
