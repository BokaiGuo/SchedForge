# Explicit Scheduled Loop IR

SchedForge schedules are transformation inputs, not executable programs. Applying
one produces an explicit `LoopIR` operation tree. Native execution, the cache/TLB
simulator, textual lowering, and LLVM ORC all receive that same tree.

## Operations

The current executable dialect contains:

- `scf.for` and `scf.parallel`
- `buffer.alloc` and `buffer.pack`
- `memref.load`, `vector.load`, and `memref.prefetch`
- `vector.accumulator.init`, `vector.broadcast`, and `vector.fma`
- fused bias/ReLU/GELU/residual epilogues
- scalar and vector stores

A vectorized reduction has the following semantic scope:

```text
vector.accumulator.init
for kk
  for k
    load A
    vector.load B
    vector.broadcast A
    vector.fma
add_bias
relu
vector.store
```

The epilogue and store are outside the reduction loop. `verify_loop_ir` rejects
empty programs, non-positive loop steps, and programs without explicit FMA and
store operations.

Graph-level elementwise epilogues are inserted immediately before each store,
so their position in the operation tree matches their execution semantics.

## Single Source of Truth

`apply_schedule(problem, schedule)` performs the rewrite. Once the rewrite is
complete, the three executable consumers do not accept `Schedule`:

```text
LoopIR ──► native runtime
      ├──► cache/TLB simulator
      └──► LLVM ORC backend
```

`analyze_loop_ir` extracts a read-only execution plan from operations. This is a
lowering analysis, not a second schedule object and not a user-facing execution
input.

## Simulation Size

Simulation now traverses the complete problem by default. Research scripts that
need bounded sampling must request it explicitly with `SimulationOptions`, and
the result records `sampled_m`, `sampled_n`, and `sampled_k` so sampled counters
cannot be mistaken for full-workload counters.
