# Contributing to SchedForge

[简体中文](CONTRIBUTING.zh-CN.md)

Thank you for helping improve SchedForge. Contributions should preserve the
project's central invariant: Tensor semantics, schedule decisions, simulation,
and executable code generation must remain explicitly separated but connected
through the same Loop IR.

## Development Setup

```bash
sudo apt-get install -y build-essential cmake ninja-build \
  llvm-18-dev llvm-18-runtime llvm-18-tools clang-18

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Pull Requests

1. Open an issue for significant architecture or behavior changes.
2. Keep changes focused; do not combine unrelated cleanup with a feature.
3. Add tests at the IR, simulator, backend, or end-to-end seam you changed.
4. Preserve negative results and claim boundaries in experiment reports.
5. Run Release tests and, for memory-sensitive changes, ASan/UBSan.
6. Update both English and Chinese README content when changing public usage.

## Coding Guidelines

- Use C++20 and follow the existing naming and ownership patterns.
- Prefer RAII and explicit ownership; avoid unmanaged lifetime coupling.
- Keep Tensor IR, Loop IR, schedule policy, simulation, and runtime execution
  in separate modules.
- Do not hard-code a schedule as universally optimal.
- Do not report simulated counters as measured hardware counters.
- Avoid committing build directories, generated caches, or machine-specific
  artifacts unless they are curated experiment evidence.

## Validation

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/schedforge-run --backend=native --M=65 --N=67 --K=63
./build/schedforge-run --backend=llvm --M=64 --N=64 --K=64
```

## Commit Messages

Use a concise imperative subject, for example:

```text
Add DTLB-aware schedule cost
Fix LLVM scalar-tail lowering
Document packing crossover result
```
