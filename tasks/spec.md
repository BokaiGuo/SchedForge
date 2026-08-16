# Spec: SchedForge v0.11 AOT Executable Deployment

## Objective
Turn Scheduled LoopIR into a portable-on-the-same-target deployment artifact
instead of requiring process-local ORC JIT compilation. A `.sfe` package must
contain a versioned manifest, target and shape guards, the canonical schedule,
LLVM object code, a loadable shared library, and enough metadata for an
independent runtime process to validate and invoke the exported kernel.

## Tech Stack
C++20, CMake, LLVM 18 IRBuilder/O3/TargetMachine object emission, ELF shared
objects, POSIX `dlopen`/`dlsym`, existing Scheduled LoopIR and TensorData.

## Commands
Build: `cmake -S . -B build-v011 -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-v011 --parallel`
Test: `ctest --test-dir build-v011 --output-on-failure`
Compile: `./build-v011/schedforge-aot compile --M=128 --N=128 --K=128 --output=results/matmul_128.sfe`
Inspect: `./build-v011/schedforge-aot inspect --artifact=results/matmul_128.sfe`
Run: `./build-v011/schedforge-aot run --artifact=results/matmul_128.sfe --repetitions=10`
Study: `./build-v011/schedforge-aot-study --output=results/aot_deployment.csv`

## Project Structure
`include/schedforge/compiler.h`: public AOT artifact, manifest, compiler, and runtime APIs.
`src/llvm_backend.cpp`: shared LoopIR-to-LLVM lowering plus ELF object emission.
`src/aot_runtime.cpp`: `.sfe` package persistence, target guards, loading, execution.
`tools/schedforge-aot.cpp`: compile, inspect, and run deployment CLI.
`tools/schedforge-aot-study.cpp`: measured JIT-versus-AOT evidence.
`tests/test_schedforge.cpp`: object, package, guard, load, and correctness tests.

## Code Style
Use typed value objects, RAII for dynamic-library handles, deterministic
line-oriented manifests, atomic package replacement, and explicit errors for
ABI, target, shape, or artifact mismatches.

## Testing Strategy
Unit-test ELF magic, manifest round trips, target/shape guards, exported symbol
lookup, numerical correctness, and repeated loads. Run the CLI in a separate
process through CTest so deployment evidence cannot accidentally reuse the ORC
cache. Run Release, ASan/UBSan, install, and the host AOT study.

## Boundaries
- Always: object bytes come from LLVM TargetMachine object emission.
- Always: runtime loads prebuilt code without invoking LLVM compilation.
- Always: validate format version, ABI, host target, problem shape, and checksums.
- Never: describe text-only `.sfe` plans as AOT machine-code deployment.
- Never: claim cross-ISA portability; v0.11 artifacts are target-specific.
- Never: accept GELU/residual LoopIR until the manifest encodes their data ABI.
- Future: Paged KV, quantized Decoder, transfer tuning, NEON validation, and
  fuzzing remain v0.12-v0.16 work.

## Success Criteria
1. `schedforge-aot compile` emits a `.sfe` directory containing `manifest.sfe`,
   `kernel.o`, and `kernel.so` from one Scheduled LoopIR.
2. `schedforge-aot run` loads `kernel.so` with `dlopen`, resolves the versioned
   C ABI symbol, executes real inputs, and validates error below `1e-3`.
3. Corrupted packages and incompatible target, ABI, checksum, or shape metadata
   fail before kernel invocation.
4. A separate-process CTest compiles then runs an artifact successfully.
5. Checked-in measurements compare cold JIT compilation with AOT load and run.
6. Release, sanitizers, install, documentation, CI, commit, and push pass.

## Open Questions
None for v0.11. The package is intentionally a directory with a `.sfe` suffix;
single-file archive transport can be added later without changing its manifest.
