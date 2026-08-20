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
The native attention runtime uses specialized AVX2 dot products and vectorized
weighted-value accumulation. Version 0.10 additionally lowers the complete
online-softmax pipeline into one executable LLVM 18 ORC function.

## Fused LLVM Function

`execute_fused_attention_llvm` specializes a complete Attention configuration
and emits one function with four pointer operands plus a query-row range. The
function computes QK, applies the causal legal-key bound, updates exact online
maximum and denominator state, rescales and accumulates the PV numerator, and
normalizes the final output. Independent query rows are split across the
Attention plan's worker count. No `Sq x Sk` score or probability matrix is
allocated.

The recorded Intel Core i5-14600K results are:

- MHA causal Prefill `S=128`: native 0.340 ms, fused LLVM 0.791 ms
- GQA causal Prefill `S=128`: native 0.255 ms, fused LLVM 0.786 ms
- GQA Decode `Sk=1024`: native 0.099 ms, fused LLVM 0.214 ms

Maximum error is below `5e-8`. The emitted function is real executable machine
code, but it remains 2.1-3.1x slower than the native specialized algorithms and
the current assembly report detects vector stack spills. Thus v0.10 establishes
the fused codegen seam and a reproducible quality diagnosis; it does not claim
native parity.

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

Paged KV is implemented as a storage/allocation strategy in
`next_milestones.h`. It maintains a logical page table over physical pages,
supports append, truncate/recycle, and active-page guards. Decode follows the
logical page table directly and performs exact online softmax without a
contiguous gather. `gather_paged_kv` remains available only for inspection and
debug/reference workflows.

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
- spill-free or native-parity fused LLVM Attention
