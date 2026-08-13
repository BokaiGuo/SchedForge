# SchedForge Architecture

SchedForge 0.3 is organized as a model-to-machine CPU AI compiler. The graph,
kernel, target, and runtime layers have separate IR objects and explicit
boundaries; the existing MatMul performance path remains the kernel laboratory,
while Transformer MLP is the end-to-end graph workload.

<style>
.sf-arch{font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0b1220;color:#e5edf8;border:1px solid #26334a;border-radius:16px;padding:18px;margin:18px 0}.sf-title{text-align:center;font-size:22px;font-weight:800;margin-bottom:14px}.sf-layer{border-radius:12px;padding:12px;margin:10px 0;border:1px solid}.sf-layer-title{font-weight:800;margin-bottom:9px}.sf-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.sf-box{background:#111c30;border:1px solid #334766;border-radius:9px;padding:9px;text-align:center;font-size:13px}.sf-box small{display:block;color:#a8b8cf;margin-top:4px}.sf-front{border-color:#4f8cff;background:#101b35}.sf-graph{border-color:#9b7cff;background:#191632}.sf-kernel{border-color:#35c59a;background:#102a29}.sf-target{border-color:#ffae57;background:#302115}.sf-runtime{border-color:#ef6f8f;background:#301621}.sf-arrow{text-align:center;color:#8fa8c9;font-size:20px;font-weight:700}@media(max-width:900px){.sf-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
<div class="sf-arch"><div class="sf-title">SchedForge 0.3 — Model-to-Machine CPU AI Compiler</div><div class="sf-layer sf-front"><div class="sf-layer-title">Frontend</div><div class="sf-grid"><div class="sf-box">StableHLO Subset<small>dot_general, add, multiply, maximum, broadcast, reshape, transpose, reduce, exp, rsqrt, convert, Gelu custom_call</small></div><div class="sf-box">Tensor SSA Graph<small>multi-op use-def graph</small></div><div class="sf-box">Shape System<small>static, dynamic, symbolic</small></div><div class="sf-box">Canonicalization<small>MLP/GELU normalization</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-graph"><div class="sf-layer-title">Graph Compiler</div><div class="sf-grid"><div class="sf-box">Shape Inference<small>constraints and runtime guards</small></div><div class="sf-box">Fusion Planner<small>legality and profitability</small></div><div class="sf-box">Dispatch IR<small>kernel boundary formation</small></div><div class="sf-box">Layout Planner<small>blocked layout propagation</small></div><div class="sf-box">Bufferization<small>tensor to physical buffers</small></div><div class="sf-box">Memory Planner<small>lifetime and workspace reuse</small></div><div class="sf-box">Quantization Propagation<small>scale, zero-point, axis</small></div><div class="sf-box">ExecutablePlan<small>dispatches, buffers, guards, artifacts</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-kernel"><div class="sf-layer-title">Kernel Compiler</div><div class="sf-grid"><div class="sf-box">Structured Compute<small>iteration domain and indexing maps</small></div><div class="sf-box">Transform IR<small>serialize, replay, mutate</small></div><div class="sf-box">AutoScheduler<small>real hardware measurement</small></div><div class="sf-box">Scheduled Loop IR<small>tiling, packing, vectorization</small></div><div class="sf-box">Tensorization<small>AVX2 tensor intrinsic matching</small></div><div class="sf-box">Native Microkernel<small>register-resident MR×NR</small></div><div class="sf-box">Measurement Database<small>hardware labels</small></div><div class="sf-box">Hybrid Cost Model<small>analytical + learned</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-target"><div class="sf-layer-title">Target Compiler</div><div class="sf-grid"><div class="sf-box">LLVM Vector IR<small>vector FMA and register PHIs</small></div><div class="sf-box">LLVM O3<small>target-aware optimization</small></div><div class="sf-box">ORC LLJIT<small>process-local kernel cache</small></div><div class="sf-box">x86 Machine Code<small>AVX2/FMA validated</small></div></div></div><div class="sf-arrow">↓</div><div class="sf-layer sf-runtime"><div class="sf-layer-title">SchedForge Runtime</div><div class="sf-grid"><div class="sf-box">Executable Runtime<small>constants, buffers, dispatches</small></div><div class="sf-box">Shape Dispatch<small>guarded specialization</small></div><div class="sf-box">Worker Teams<small>topology-aware CPU affinity</small></div><div class="sf-box">Hardware Feedback<small>latency, perf/PMU, validation</small></div></div></div></div>

## End-to-End Pipeline

1. `StableHLOImporter` imports the supported StableHLO subset into `TensorGraph`.
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
   vectorization, unrolling, packing, parallelization, and tensorization.
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
    static shapes; irregular shapes use the safe vector-plus-tail path.
14. `ExecutablePlan` serializes graph IR, dispatches, memory plans, guards,
    Transform IR, Scheduled LoopIR, tensor intrinsics, and LLVM kernel artifacts
    into `.sfe`.

## Flagship Workloads

- **Kernel benchmark:** fused FP32 MatMul epilogues for schedule and
  microarchitecture research.
- **Graph benchmark:** Transformer MLP
  `Linear → Bias → GELU → Linear → Bias → Residual`, executed end to end.
- **Next graph path:** mini attention with Q/K/V projections, transpose,
  score MatMul, exponential/reduction normalization, and value MatMul. The graph
  abstraction and shape analysis are implemented; optimized softmax/attention
  execution remains future backend work.

## Claim Boundaries

- The MLP path is imported, compiled, executed, and validated end to end.
- GELU and residual are explicit LoopIR epilogues executed by the native graph
  runtime. LLVM ORC currently lowers and validates the MatMul/bias kernel portion
  embedded in each dispatch artifact.
- LLVM-generated register microkernels are validated for legal static shapes;
  the native backend remains the highest-throughput path on the current host.
- Attention, graph quantization propagation, and the learned cost model have
  concrete APIs and tests but are not yet production-optimized inference paths.
- StableHLO support is deliberately a subset, not a complete specification
  implementation.
- AVX2 is validated on one Intel CPU; AVX-512, NEON, and multi-node NUMA require
  matching hardware validation.
