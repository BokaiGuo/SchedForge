# Spec: SchedForge Realistic Decoder and Whole-Graph Planning

## Objective
Upgrade SchedForge from tiny Decoder integration validation to a realistic-shape
AI compiler benchmark and a whole-graph execution planner. One StableHLO Decoder
must still produce one inspectable plan and one runtime call. Results must label
real execution, compile-only feasibility, and analytical estimates separately.

## Tech Stack
C++20, CMake, LLVM 18 ORC, existing Tensor SSA/Transform IR/Scheduled LoopIR,
Attention compiler, MoE compiler, and native AVX2 runtime.

## Commands
Build: `cmake -S . -B build-v09 -DCMAKE_BUILD_TYPE=Release && cmake --build build-v09 --parallel`
Test: `ctest --test-dir build-v09 --output-on-failure`
Benchmark: `./build-v09/schedforge-decoder-bench --suite=realistic --output=results/decoder_realistic.csv`
Plan study: `./build-v09/schedforge-decoder-bench --suite=optimizer --output=results/decoder_plan_optimizer.csv`

## Project Structure
`include/schedforge/decoder_compiler.h`: Decoder plan, metrics, optimizer API.
`src/decoder_compiler.cpp`: plan generation, execution, stage timing, optimization.
`tools/schedforge-decoder-bench.cpp`: benchmark matrix and CSV/report emission.
`tests/test_schedforge.cpp`: optimizer and benchmark-schema regression tests.
`docs/`: architecture and benchmark methodology.
`results/`: curated host-specific measured evidence.

## Code Style
Use existing C++20 value types and explicit ownership. Planner decisions are
typed enums/structs, serialized in `.sfe`, and never inferred from display names.

## Testing Strategy
Unit-test candidate generation, cost accounting, winner selection, stage timing,
and CSV schema at small shapes. Run Release and ASan/UBSan. Execute feasible
realistic shapes on hardware. Compile-only rows must never carry measured latency.

## Boundaries
- Always: distinguish measured, compile-only, and analytical evidence.
- Always: optimize end-to-end Decoder latency, not independent kernel scores.
- Always: report workspace, compile/JIT time, stage percentages, and tokens/s.
- Never: claim Large prefill latency without completing the real execution.
- Never: use simulator output as the optimizer's measured winner.
- Future: LLVM quality, AOT, reduced precision, Paged KV, heterogeneous cores,
  NEON, and active tuning remain roadmap items after this release.

## Success Criteria
1. Benchmark profiles cover Tiny, Medium, and Large architecture shapes plus
   prefill/decode scenarios and Dense/MoE metadata.
2. Feasible rows run on the CPU and record end-to-end/stage latency; infeasible
   rows are explicitly compile-only with memory/FLOP/JIT feasibility data.
3. `ExecutablePlanOptimizer` enumerates cross-dispatch plan candidates covering
   attention strategy, intermediate layout/materialization, schedule family,
   workspace reuse, and thread placement.
4. Candidate selection can use real end-to-end measurements and records the
   winner, baseline, speedup, and number of hardware measurements.
5. `.sfe`, CLI, README, architecture docs, tests, results, changelog, and version
   metadata are updated; Release, sanitizers, install audit, and GitHub CI pass.
