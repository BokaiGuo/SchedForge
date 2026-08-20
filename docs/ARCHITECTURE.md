# SchedForge Architecture

SchedForge 0.16 is organized as a model-to-machine CPU AI compiler. The graph,
kernel, target, and runtime layers have separate IR objects and explicit
boundaries; the existing MatMul performance path remains the kernel laboratory,
while a complete Transformer Decoder Layer is the end-to-end graph workload.

<style>
.sf-arch{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0b1220;color:#e5edf8;border:1px solid #26334a;border-radius:16px;padding:18px;margin:18px 0}.sf-title{text-align:center;font-size:22px;font-weight:800;margin-bottom:14px}.sf-layer{border-radius:12px;padding:12px;margin:10px 0;border:1px solid}.sf-layer-title{font-weight:800;margin-bottom:9px}.sf-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.sf-box{background:#111c30;border:1px solid #334766;border-radius:9px;padding:9px;text-align:center;font-size:13px}.sf-box small{display:block;color:#a8b8cf;margin-top:4px}.sf-front{border-color:#4f8cff;background:#101b35}.sf-graph{border-color:#9b7cff;background:#191632}.sf-kernel{border-color:#35c59a;background:#102a29}.sf-target{border-color:#ffae57;background:#302115}.sf-runtime{border-color:#ef6f8f;background:#301621}.sf-arrow{text-align:center;color:#8fa8c9;font-size:20px;font-weight:700}@media(max-width:900px){.sf-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
<div class="sf-arch"><div class="sf-title">SchedForge 0.16 — Model-to-Machine CPU AI Compiler</div><div class="sf-layer sf-front"><div class="sf-layer-title">Frontend</div><div class="sf-grid"><div class="sf-box">StableHLO Subset<small>constants, dot_general, RMSNorm, RoPE, SiLU, SwiGLU</small></div><div class="sf-box">Tensor SSA Graph<small>multi-op use-def graph</small></div><div class="sf-box">Shape System<small>static, dynamic, symbolic</small></div><div class="sf-box">Decoder Canonicalization<small>QKV, Gate-Up, SDPA, MoE</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-graph"><div class="sf-layer-title">Graph Compiler</div><div class="sf-grid"><div class="sf-box">Shape Inference<small>constraints and runtime guards</small></div><div class="sf-box">Decoder Fusion<small>RMSNorm, QKV, RoPE, Gate-Up</small></div><div class="sf-box">Attention / MoE Plans<small>online softmax or routed experts</small></div><div class="sf-box">Constant Specialization<small>packed QKV and Gate-Up</small></div><div class="sf-box">Bufferization<small>tensor to physical buffers</small></div><div class="sf-box">Memory Planner<small>lifetime and workspace reuse</small></div><div class="sf-box">Dispatch IR<small>kernel boundary formation</small></div><div class="sf-box">DecoderExecutablePlan<small>one graph, one plan, one call</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-kernel"><div class="sf-layer-title">Kernel Compiler</div><div class="sf-grid"><div class="sf-box">Structured Compute<small>iteration domain and indexing maps</small></div><div class="sf-box">Transform IR<small>serialize, replay, apply</small></div><div class="sf-box">AutoScheduler<small>real hardware measurement</small></div><div class="sf-box">Scheduled Loop IR<small>tiling, packing, vectorization</small></div><div class="sf-box">Tensorization<small>AVX2 tensor intrinsic matching</small></div><div class="sf-box">Native Microkernel<small>register-resident MR×NR</small></div><div class="sf-box">Measurement Database<small>hardware labels</small></div><div class="sf-box">Hybrid Cost Model<small>analytical + learned</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-target"><div class="sf-layer-title">Target Compiler</div><div class="sf-grid"><div class="sf-box">LLVM Vector IR<small>vector FMA and graph epilogues</small></div><div class="sf-box">LLVM O3<small>target-aware optimization</small></div><div class="sf-box">ORC LLJIT<small>process-local kernel cache</small></div><div class="sf-box">x86 Machine Code<small>AVX2/FMA validated</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-runtime"><div class="sf-layer-title">SchedForge Runtime</div><div class="sf-grid"><div class="sf-box">Decoder Runtime<small>Dense or MoE full-layer call</small></div><div class="sf-box">Executable Runtime<small>constants, buffers, dispatches</small></div><div class="sf-box">Worker Teams<small>topology-aware CPU affinity</small></div><div class="sf-box">Hardware Feedback<small>latency, perf/PMU, validation</small></div></div></div></div>

## Decoder Layer Compilation Unit

The primary 0.10 unit is `RMSNorm → fused QKV → RoPE → exact GQA/MQA attention
→ output projection → residual → RMSNorm → Dense SwiGLU or MoE → residual`.
The compiler preserves the imported StableHLO graph, creates an explicit
canonical Decoder graph, specializes immutable projection weights, embeds the
Attention/MoE subplans, and serializes all executable LoopIR and LLVM artifacts
into one `DecoderExecutablePlan`.

## Whole-Graph Plan Optimization

SchedForge 0.9 adds a planning layer above individual dispatch selection.
`ExecutablePlanOptimizer` enumerates typed policies over Attention algorithm,
intermediate layout, materialization boundary, workspace reuse, schedule family,
thread count, and compact/spread placement. The analytical model only controls
which candidates receive the measurement budget; the selected plan is decided
by complete Decoder execution.

Candidate comparison uses equal warmup. Any preliminary challenger is then
compared against the default plan in three interleaved confirmation rounds.
This prevents cold-start order from being narrated as a compiler speedup. The
recorded Tiny Prefill case retains the baseline, while Tiny Decode `KV=512`
confirms a 1.400x improvement from the one-thread Split-KV policy.

The plan owns the selected policy, specialized constants, nested Attention/MoE
plans, physical buffers, workspace reuse decision, and executable dispatches.
MoE activations cross the runtime seam independently of expert weights, so
whole-layer execution does not copy the expert parameter store per invocation.

## End-to-End Pipeline

Dense, MoE, and Attention graphs share Tensor SSA and target lowering. MoE branches after
graph canonicalization into Router/TopK, histogram/prefix dispatch, Segmented
Tensor IR, grouped expert compute, and a routing-aware strategy planner before
rejoining dispatch bufferization and LoopIR lowering. Attention branches through
structural SDPA fusion and an algorithm planner: Materialized is the correctness
baseline, prefill lowers to an exact online-softmax TilePipelineIR, and decode
lowers to contiguous-KV Split-KV tasks with exact partial-state merging.

```mermaid
flowchart LR
    T["Tensor SSA"] --> D["Dense MLP"]
    T --> M["MoE Routing + Segmented Experts"]
    T --> A["SDPA Fusion"]
    A --> P["Attention Strategy Planner"]
    P --> AM["Materialized"]
    P --> AF["IO-aware Prefill"]
    P --> AD["Split-KV Decode"]
    D --> L["Structured / Loop IR"]
    M --> L
    AM --> L
    AF --> L
    AD --> L
    L --> C["LLVM / Native CPU Runtime"]
```

1. `StableHLOImporter` imports the supported StableHLO subset, including literal
   constants, into `TensorGraph`.
2. `GraphCanonicalizationPass` normalizes framework-level patterns such as the
   Transformer MLP GELU expansion.
3. `ShapeInferencePass` propagates tensor shapes and emits runtime constraints.
4. `FusionPlanner` evaluates use-def legality and materialization profitability,
   then forms `Dispatch` regions.
5. `GraphLayoutPlanner` propagates blocked layouts across compatible dispatches.
6. `GraphBufferizer` materializes only dispatch-boundary tensors and reuses an
   aligned workspace according to lifetime intervals.
7. `StructuredComputeLowering` preserves iteration domains, indexing maps, and
   reduction semantics before loop lowering.
8. `TransformProgram` records tiling, interchange, register tiling,
   vectorization, unrolling, packing, parallelization, and tensorization, then
   directly applies the recipe to create verified LoopIR.
9. Applying the winning transform rewrites an explicit Scheduled LoopIR with
   loops, parallel regions, pack/prefetch, vector reduction, epilogue, and store
   operations. Schedule is no longer an executable backend input.
10. Native execution, the cache/TLB simulator, and LLVM lowering consume that
    same verified LoopIR. Full-size simulation is the default; bounded sampling
    is explicit and recorded in the result.
11. Auto-tuning executes all deduplicated legal candidates on hardware and
   remeasures finalists in randomized rounds.
12. `TensorizationPass` maps structured contractions to target intrinsics such
    as `avx2_f32_m4n8`.
13. LLVM ORC generates register-resident MR×NR vector-FMA kernels for legal
    static shapes; irregular shapes use the safe vector-plus-tail path. ORC
    execution preserves LoopIR thread counts using MR-aligned disjoint row
    partitions rather than silently benchmarking one worker.
14. `ExecutablePlan` serializes graph IR, dispatches, memory plans, guards,
    Transform IR, Scheduled LoopIR, tensor intrinsics, and LLVM kernel artifacts
    into `.sfe`.

## Production LLVM Code Quality

Version 0.10 treats LLVM quality as an explicit compiler object. The same
Scheduled LoopIR is executed by native AVX2 and LLVM ORC, while generated
assembly is classified by total/vector/FMA instructions, loads, stores,
branches, address-generation proxies, stack accesses, and vector spill
patterns. `schedforge-codegen-study` stores the comparison as measured CSV.

Attention has a separate algorithm-level lowering from `TilePipelineIR` into
one ORC function. That function contains QK, legal causal key limits, exact
online maximum/denominator rescaling, PV numerator accumulation, and final
normalization. It supports MHA and GQA and partitions independent query rows
across the plan's worker count.

The checked-in results are deliberately not described as parity: LLVM MatMul is
1.8-2.5x slower than native across the recorded cubic rows, and fused LLVM
Attention is 2.1-3.1x slower than the specialized native runtime with a vector
spill pattern still present. These gaps define the next code-generation work.

## AOT Deployment Boundary

Version 0.11 reuses the optimized LoopIR LLVM module for a second target path:
LLVM `TargetMachine` emits a PIC ELF relocatable object, the system linker forms
`kernel.so`, and a versioned `.sfe` directory records the exact ABI, shape,
target triple, CPU, and checksums. The deployment runtime validates the package
and resolves `schedforge_matmul_v1` through `dlopen`/`dlsym`; LLVM compilation
is not present on the load or execution path.

The v1 AOT unit is intentionally narrower than the graph compiler: it is one
shape-specialized FP32 MatMul kernel, one target CPU, and one runtime thread.
Whole-graph constant relocation and multi-kernel dispatch are future extensions,
not implied by the existence of object emission.

## Runtime Milestone Line

The runtime now exposes Paged KV storage, INT8 weight-only MatMul, measured
transfer scheduling, NEON source/capability inspection, and deterministic
LoopIR fuzzing. These are explicit APIs in `next_milestones.h`, covered by
CTest and `schedforge-next-study`/`schedforge-fuzz`. The current x86_64 host
validates the first, second, third, and fifth slices directly; NEON is source
and capability validation only until an ARM host is available.

## Flagship Workloads

- **Kernel benchmark:** fused FP32 MatMul epilogues for schedule and
  microarchitecture research.
- **Dense graph benchmark:** Transformer MLP
  `Linear → Bias → GELU → Linear → Bias → Residual`, executed end to end.
- **Dynamic sparse benchmark:** Top-2 MoE MLP with Router, TopK, segmented token
  dispatch, variable-M grouped expert SwiGLU, weighted combine, and load-aware
  expert task scheduling, executed end to end.
- **Attention benchmark:** exact causal/non-causal MHA, GQA, and MQA with
  Materialized, Tiled Materialized, IO-aware prefill, and Split-KV decode
  algorithms, executed end to end.

## Claim Boundaries

- Dense MLP, FP32 Top-2 MoE, IO-aware prefill attention, and Split-KV decode are compiled, executed, and validated end to end.
- GELU and residual are explicit LoopIR epilogues executed by the native graph
  runtime. LLVM ORC currently lowers and validates the MatMul/bias kernel portion
  embedded in each dispatch artifact.
- LLVM-generated register microkernels are validated for legal static shapes;
  the native backend remains the highest-throughput path on the current host.
- Attention is a CPU Flash-style exact lowering, not a GPU FlashAttention-2
  implementation. Paged KV storage is implemented; direct page-aware tiled
  traversal and spill-free native-parity fused LLVM code remain future work.
- StableHLO support is deliberately a subset, not a complete specification
  implementation.
- AVX2 is validated on one Intel CPU; AVX-512, NEON, and multi-node NUMA require
  matching hardware validation.
