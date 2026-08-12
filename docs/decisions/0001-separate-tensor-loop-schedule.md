# ADR-0001: Separate Tensor IR, Loop IR, and Schedule

## Status

Accepted

## Date

2026-08-12

## Context

SchedForge needs to represent tensor semantics, apply CPU-specific execution
choices, predict hardware behavior, and generate both native and LLVM code. If
these concerns are encoded in one kernel implementation, the simulator and the
executable backend can silently evaluate different programs.

## Decision

Keep three distinct objects:

- Tensor IR describes what is computed.
- Schedule describes target-dependent execution decisions.
- Loop IR describes the concrete scheduled execution plan.

The simulator and both executable backends consume the same Loop IR and
Schedule representation.

## Alternatives Considered

### Direct Tensor-to-LLVM lowering

Rejected because tiling, packing, vector-axis selection, and schedule search
would become LLVM-specific and difficult to inspect or simulate.

### Separate handwritten kernels per configuration

Rejected because it turns the project into a kernel benchmark collection rather
than a compiler and makes search-space growth unmaintainable.

## Consequences

- Schedule transformations are testable without invoking LLVM.
- Simulator predictions and execution plans share a common source of truth.
- Backends can evolve independently, but IR cloning and validation remain
  necessary when evaluating multiple candidates.
