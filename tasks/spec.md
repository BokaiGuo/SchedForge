# Spec: SchedForge v0.12-v0.17 Runtime and Validation Line

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
  truncate/recycle, active-page guards, and direct page-aware decode traversal.
- v0.13: per-channel INT8 weight quantization and real FP32-output MatMul with
  reference validation and measured latency/GFLOPS.
- v0.14: chunk/worker transfer schedule search over real memory copies with
  destination validation and measured bandwidth.
- v0.15: valid NEON intrinsic source generation and AArch64 freestanding syntax
  cross-compilation. The x86 host reports runtime NEON unavailable.
- v0.16: deterministic randomized LoopIR generation, rejection accounting,
  numerical invariants, CTest integration, and standalone fuzz CLI.
- v0.17: Dense Decoder INT8 Q/K/V/O/Gate/Up/Down projections for prefill and
  KV-cache Decode, direct paged traversal, and NEON cross-target syntax gates.

## Boundaries
- Always: separate measured hardware facts, generated source, and unavailable targets.
- Never: claim NEON machine-code execution on this x86 host.
- Never: call transfer tuning a NUMA benchmark; it is host-memory copy tuning.
- Never: call Dense Decoder INT8 validation MoE expert quantization or ARM runtime
  performance validation.
- Future: MoE expert INT8 weights, cross-target NEON execution, and libFuzzer
  corpus minimization.

## Success Criteria
1. All five APIs compile, execute, and have regression tests.
2. Paged KV preserves logical token order under page boundaries.
3. INT8 output error is below `1e-2` on the checked-in test shape.
4. Transfer tuning reports positive measured bandwidth and validates bytes.
5. NEON report contains intrinsic source and honest host capability status.
6. 128+ deterministic fuzz cases pass with zero invariant failures.
7. Release, ASan/UBSan, install, CI, README, report, and changelog are updated.
