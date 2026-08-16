# Tasks

- [x] Shared LLVM AOT compiler
  - Acceptance: one LoopIR lowering feeds ORC JIT, assembly, and ELF object emission.
  - Verify: unit test sees ELF magic and non-empty exported symbol metadata.
- [x] Versioned `.sfe` package
  - Acceptance: manifest, object, shared library, target guards, and checksums round-trip.
  - Verify: load succeeds for valid packages and rejects tampered artifacts.
- [x] Independent AOT runtime and CLI
  - Acceptance: compile, inspect, and run commands execute without runtime LLVM compilation.
  - Verify: separate-process CTest compiles and runs a package below `1e-3` error.
- [x] Host deployment study
  - Acceptance: CSV records JIT compile, AOT compile, AOT load, execution, GFLOPS, and error.
  - Verify: all measured timings are nonnegative and validation passes.
- [x] v0.11 release validation
  - Acceptance: Release, ASan/UBSan, install, docs, version, CI, commit, and push pass.
  - Verify: clean git status and green GitHub Actions.
