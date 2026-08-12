# SchedForge Architecture

## Separation of Concerns

Tensor IR represents **what** is computed. `Schedule` represents **how** it
should execute. Loop IR is the concrete scheduled execution plan. Both the
simulator and code generators consume that same plan, preventing prediction and
execution from silently referring to different schedules.

## Compiler Pipeline

1. `IRBuilder` creates SSA Tensor IR with use-def tracking.
2. Tensor passes infer shapes/layouts and describe fused epilogues.
3. `TensorToLoopLowering` creates canonical `i-j-k` Loop IR.
4. Loop passes apply schedule decisions without changing tensor semantics.
5. The simulator interprets loop/memory behavior for candidate ranking.
6. Native AVX2 or LLVM ORC lowers the selected plan to executable code.
7. Runtime validation compares every candidate with the scalar reference.
8. Hardware measurements and calibration feed back into schedule selection.

## Native Execution Hierarchy

```text
NC/MC/KC outer tiles
  -> BN/BM/BK cache tiles
    -> NR/MR register tiles
      -> AVX2 FP32 vectors
        -> scalar tails
```

PackB uses NR-oriented panels and PackA uses MR-oriented panels. Packing is an
explicit schedule decision because the copy cost can exceed its benefit.

## LLVM Backend

The LLVM backend constructs a real LLVM `Module` with dynamic `M/N/K`, applies
LLVM's O3 default pipeline, executes through ORC `LLJIT`, and emits assembly for
inspection. Explicit `<8 x float>` operations use alignment 1 because runtime
buffers are not guaranteed to be 32-byte aligned. This produces safe `vmovups`
loads/stores and AVX2 FMA instructions.

## Hardware Model

The model includes:

- Set-associative L1/L2/L3 caches with LRU replacement
- DTLB entries and miss penalties
- Software-prefetch issuance and usefulness
- Register pressure and spill penalties
- Packing traffic and compute cost
- Thread parallelism and calibrated memory bandwidth

It is designed for relative schedule ranking, not cycle-accurate reproduction
of Intel out-of-order execution.

## Runtime and Dynamic Shapes

Dynamic dimensions are specialized into runtime buckets. Schedule and emitted
IR artifacts are persisted by problem, target, and schedule key. ORC function
pointers are cached in process. Thread affinity and NUMA topology detection are
implemented; multi-node NUMA placement cannot be empirically compared on the
current single-node workstation.
