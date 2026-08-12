# ADR-0002: Use LLVM 18 for the JIT Backend

## Status

Accepted

## Date

2026-08-12

## Context

SchedForge requires a reproducible LLVM API surface for `IRBuilder`, the new
pass manager, assembly emission, and ORC `LLJIT`. Supporting many LLVM releases
would add compatibility branches before the compiler architecture stabilizes.

## Decision

Require LLVM 18 for the initial public release and fail CMake configuration
when a different major version is selected.

## Alternatives Considered

### Support LLVM 17-20

Rejected for version 0.1 because ORC and pass-manager API compatibility would
increase maintenance and CI cost without improving the core experiment.

### Make LLVM optional

Rejected because real ORC JIT execution is a central project feature rather
than an optional demonstration.

## Consequences

- Ubuntu 24.04 has directly installable LLVM 18 packages.
- Builds are reproducible against one known API version.
- Contributors on other distributions may need to provide `LLVM_DIR` manually.
