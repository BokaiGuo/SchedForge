# Spec: SchedForge v0.10 Production LLVM and Fused CodeGen

## Objective
Close the largest remaining backend gap after the v0.8 realistic Decoder suite
and v0.9 whole-graph planner. The same Scheduled LoopIR must carry its thread
semantics into native and LLVM execution, expose comparable machine-code quality
metrics, and lower exact IO-aware Attention into one executable LLVM function.

## Tech Stack
C++20, CMake, LLVM 18 IRBuilder/O3/ORC LLJIT, existing Scheduled LoopIR,
Attention TilePipelineIR, AVX2 native runtime, and CSV evidence artifacts.

## Commands
Build: `cmake -S . -B build-v010 -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-v010 --parallel`
Test: `ctest --test-dir build-v010 --output-on-failure`
Codegen study: `./build-v010/schedforge-codegen-study --output=results/llvm_codegen_study.csv`
Attention study: `./build-v010/schedforge-attention --fused-llvm --sequence=128 --repetitions=5`
Sanitizers: `cmake -S . -B build-v010-asan -DSCHEDFORGE_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug`

## Project Structure
`src/llvm_backend.cpp`: LoopIR LLVM compilation, parallel invocation, assembly analysis.
`src/attention_llvm_backend.cpp`: fused exact online-softmax Attention LLVM backend.
`include/schedforge/compiler.h`: backend quality result APIs.
`include/schedforge/attention_compiler.h`: fused Attention LLVM result API.
`tools/schedforge-codegen-study.cpp`: same-LoopIR native/LLVM comparison.
`tests/test_schedforge.cpp`: backend semantics, quality metrics, fused Attention correctness.
`results/`: host-specific measured codegen evidence.

## Code Style
Use typed value objects and RAII. Generated code decisions must derive from
LoopIR or TilePipelineIR, not duplicated display strings. Preserve explicit
measured versus analytical evidence boundaries.

## Testing Strategy
Unit-test parallel LLVM correctness, cache behavior, assembly metrics, and fused
Attention correctness at small shapes. Run Release and ASan/UBSan. Run measured
native/LLVM and Attention studies on the current host. Do not claim performance
for paths that only emit text.

## Boundaries
- Always: compare native and LLVM from the same Scheduled LoopIR.
- Always: measure complete execution and validate against independent references.
- Always: report thread count, instruction metrics, spills, compile time, and speed ratio.
- Never: call emitted-but-unexecuted LLVM IR a production kernel.
- Never: claim LLVM parity if the checked-in measurements do not show it.
- Future: portable object-file AOT, Paged KV, BF16/INT8 Decoder, transfer tuning,
  NEON hardware validation, and compiler fuzzing remain v0.11-v0.16 work.

## Success Criteria
1. LLVM execution honors LoopIR thread count with disjoint row partitions.
2. Assembly reports include total instructions, branches, loads/stores, address
   generation proxies, stack accesses, FMA count, and spill detection.
3. A reproducible CLI compares native and LLVM on identical LoopIR and writes CSV.
4. IO-aware Attention can compile and execute as one LLVM function containing
   QK, online max/sum rescaling, PV accumulation, and final normalization.
5. All new paths validate below `1e-3`; Release, sanitizers, install, and CI pass.
6. README, architecture, changelog, report, version, raw results, and claim
   boundaries are updated and pushed to GitHub.
