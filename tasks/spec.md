# Spec: SchedForge v0.12-v0.16 Runtime and Validation Line

## Objective
Complete the next five runtime/compiler milestones as tested vertical slices:
Paged KV storage and decode, quantized FP32-output MatMul, measured transfer
tuning, ARM NEON codegen inspection with a real intrinsic source path, and
deterministic compiler fuzzing. Every slice must execute real code or clearly
report unavailable host hardware; proxy-only results cannot be labeled as
hardware validation.

## Commands
Build: `cmake -S . -B build-v016 -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-v016 --parallel`
Test: `ctest --test-dir build-v016 --output-on-failure`
Study: `./build-v016/schedforge-next-study`
Fuzz: `./build-v016/schedforge-fuzz --iterations=1000 --seed=1`
Sanitizers: `cmake -S . -B build-v016-asan -DSCHEDFORGE_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug`

## Implemented Slices
- v0.12: page-table KV allocation, non-contiguous physical pages, append,
  truncate/recycle, active-page guards, gather, and decode execution through the existing exact
  attention runtime.
- v0.13: per-channel INT8 weight quantization and real FP32-output MatMul with
  reference validation and measured latency/GFLOPS.
- v0.14: chunk/worker transfer schedule search over real memory copies with
  destination validation and measured bandwidth.
- v0.15: NEON intrinsic source generation and compile-time ARM detection. The
  x86 host reports NEON unavailable instead of fabricating hardware results.
- v0.16: deterministic randomized LoopIR generation, rejection accounting,
  numerical invariants, CTest integration, and standalone fuzz CLI.

## Boundaries
- Always: separate measured hardware facts, generated source, and unavailable targets.
- Never: claim NEON machine-code execution on this x86 host.
- Never: call transfer tuning a NUMA benchmark; it is host-memory copy tuning.
- Never: call INT8 weight-only MatMul a complete quantized Decoder until all
  Decoder projections and attention paths consume quantized constants.
- Future: full quantized Decoder graph lowering, cross-target NEON execution,
  Paged KV direct tiled traversal without gather, and libFuzzer corpus minimization.

## Success Criteria
1. All five APIs compile, execute, and have regression tests.
2. Paged KV preserves logical token order under page boundaries.
3. INT8 output error is below `1e-2` on the checked-in test shape.
4. Transfer tuning reports positive measured bandwidth and validates bytes.
5. NEON report contains intrinsic source and honest host capability status.
6. 128+ deterministic fuzz cases pass with zero invariant failures.
7. Release, ASan/UBSan, install, CI, README, report, and changelog are updated.
