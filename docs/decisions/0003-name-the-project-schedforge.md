# ADR-0003: Name the Project SchedForge

## Status

Accepted

## Date

2026-08-12

## Context

The original working name described the tensor domain but did not identify the
project's primary contribution and overlapped with existing tensor software
names. The public name must work for GitHub, papers, resumes, command-line
tools, C++ APIs, and future operators beyond MatMul.

## Decision

Use **SchedForge** as the project name and **调度锻造器** as its Chinese name.
The full positioning is **SchedForge — A Target-Aware CPU Tensor Compiler**.

The name combines:

- **Sched**: schedule search, tiling, packing, vectorization, prefetching, and
  thread scheduling.
- **Forge**: progressively lowering and shaping Tensor IR into LLVM IR and CPU
  machine code.

Public identifiers use the same name consistently:

- GitHub repository: `BokaiGuo/SchedForge`
- C++ namespace: `schedforge`
- Header directory: `include/schedforge`
- CMake package and target namespace: `SchedForge`
- Command-line tools: `schedforge-opt`, `schedforge-run`, and related tools

## Alternatives Considered

### Keep the working name

Rejected because it was less schedule-specific and had a higher risk of name
collision with existing tensor projects.

### Retain compatibility aliases

Rejected for version 0.1.0 because the project had not yet been published. A
clean rename avoids carrying deprecated names into the first public API.

## Consequences

- The first public release has one coherent repository and API identity.
- Existing local build directories and caches must be regenerated.
- Future documentation uses the tagline:
  **Forge tensor schedules into efficient CPU code.**
