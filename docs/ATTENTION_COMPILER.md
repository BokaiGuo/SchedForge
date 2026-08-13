# CPU Flash-Style Attention Compiler

SchedForge 0.5 implements an exact, IO-aware scaled dot-product attention
lowering for the CPU cache hierarchy. It adopts the core FlashAttention idea—
tiling plus online softmax without a persistent `Sq x Sk` matrix—but does not
claim to implement GPU FlashAttention-2.

## Semantic IR and Fusion

The frontend first represents standard attention as:

```text
tensor.matmul(Q, transpose(K))
  -> tensor.multiply(scale)
  -> tensor.mask(causal)
  -> tensor.softmax
  -> tensor.matmul(V)
```

`AttentionFusionPass` structurally verifies this producer chain and creates an
`attention.sdpa` operation. The public shape contract is:

```text
Q : [B, Hq, Sq, D]
K : [B, Hkv, Sk, D]
V : [B, Hkv, Sk, Dv]
O : [B, Hq, Sq, Dv]
Hq % Hkv == 0
```

The interface and runtime support MHA, GQA, and MQA. The first release executes
FP32 with arbitrary positive `Sq/Sk` values bounded by compiled guards.

## Algorithms

One `attention.sdpa` can select four concrete algorithms:

- `materialized`: creates score and probability matrices.
- `tiled-materialized`: tiles QK/PV but retains full intermediates.
- `io-aware`: consumes `BQ x BK` score tiles immediately with exact online softmax.
- `split-kv-decode`: splits the KV range, computes partial online states in
  parallel, and merges them with numerically correct max/denominator rescaling.

The automatic planner uses the execution regime and target cache sizes:
small attention selects materialization, prefill selects measured IO-aware tile
candidates, and `Sq <= 2` selects Split-KV decode.

## TilePipelineIR

The serialized `.sfe` plan contains an explicit fused tile pipeline:

```text
tile.pipeline @attention {
  attention.qk
  attention.scale
  attention.causal_mask
  reduce.max
  vector.max
  vector.exp
  reduce.sum
  attention.rescale
  attention.online_softmax
  attention.pv
  vector.div
}
```

QK and PV also lower into ordinary scheduled LoopIR and LLVM 18 ORC artifacts.
The native attention runtime currently uses specialized AVX2 dot products and
vectorized weighted-value accumulation; the embedded LLVM kernels are compiled
and numerically validated building blocks, not yet one monolithic fused LLVM
attention function.

## Exact Online Softmax

For each query row, the IO-aware path maintains running maximum `m`, denominator
`l`, and output numerator `O`. For a new score tile with maximum `m_tile`:

```text
m_new = max(m, m_tile)
alpha = exp(m - m_new)
l_new = alpha * l + sum(exp(score - m_new))
O_new = alpha * O + exp(score - m_new) * V_tile
```

The final output is `O / l`. Causal tiles wholly beyond the legal frontier are
skipped; diagonal tiles apply element-level masking before online updates.

## KV Cache and Decode

`KVCache` is a growable contiguous cache with explicit capacity and current
length. `append_kv` preserves `[B, Hkv, sequence, D]` head-major layout across
multiple appends. Decode accepts any cache length within the compiled `Sk`
guard. Split-KV computes partial `(m, l, O)` states in parallel and merges them
exactly, enabling GQA and MQA head sharing.

Paged KV is intentionally separate: Flash-style attention is a compute
algorithm, while paged KV is a storage/allocation strategy.

## Simulation and Measurement

The attention simulator compares algorithm-level FLOPs, bytes read/written,
temporary footprint, arithmetic intensity, tile count, causal skips,
exponential/reduction work, and estimated L2/LLC traffic.

Checked-in evidence includes:

- `results/attention_strategy_matrix.csv`: 12 real executions across sequence
  `128/256/512` and four algorithms.
- `results/attention_scaling_analysis.csv`: 96 analytical cases across heads
  `8/12`, dimensions `64/128`, sequence `128-4096`, and four algorithms.
- `results/attention_pmu.txt`: Linux `perf` process counters for 200 executions.
- `results/attention_prefill.sfe` and `results/attention_decode.sfe`: serialized
  prefill and decode compiler artifacts.

On the recorded Intel Core i5-14600K at `B=1, H=8, S=512, D=64`, measured
materialized attention uses 16 MiB of temporary storage and 18.929 ms P50;
auto-scheduled IO-aware attention uses 49 KiB and 2.947 ms P50. The IO-aware
peak accounts for all concurrently active worker tile states. These numbers
are host-specific and not a universal speedup claim.

The checked-in PMU snapshot reports P-core IPC 2.561, L1D miss rate 1.066%, and
cache-reference miss rate 16.230%. Counters cover the process and therefore
include runtime/framework overhead around the 200 measured executions.

## Claim Boundaries

Implemented and validated:

- FP32 exact causal and non-causal attention
- structural SDPA fusion and Attention IR
- Materialized, Tiled Materialized, IO-aware, and Split-KV algorithms
- dynamic `Sq/Sk`, MHA, GQA, and MQA
- contiguous growable KV cache
- real hardware attention tile tuning and PMU capture
- QK/PV scheduled LoopIR plus LLVM ORC artifacts

Not yet claimed:

- GPU FlashAttention-2 compatibility or performance
- paged KV cache allocation
- BF16/INT8 attention
- dropout, backward, or training kernels
- distributed or NUMA-aware attention
- one fully fused LLVM-generated attention machine-code function
