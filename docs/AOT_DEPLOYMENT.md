# AOT Executable Deployment

SchedForge 0.11 adds a real target-specific machine-code form of `.sfe`. The
package is an inspectable directory with a `.sfe` suffix:

```text
matmul.sfe/
  manifest.sfe  version, ABI, target, shape, checksums
  loop.ir       canonical Scheduled LoopIR
  kernel.ll     optimized LLVM IR
  kernel.s      emitted target assembly
  kernel.o      PIC ELF relocatable object
  kernel.so     loadable shared object
```

## Compilation and Loading

```text
Scheduled LoopIR
  -> verified LLVM Module
  -> LLVM O3
  -> TargetMachine(ObjectFile, PIC)
  -> kernel.o
  -> system linker
  -> kernel.so
  -> manifest and checksums
  -> dlopen + dlsym + C ABI invocation
```

The runtime validates format version 1, `schedforge_matmul_v1`, exact problem
shape, target triple, target CPU, and checksums, then compares those invariant
fields with `schedforge_aot_metadata_v1` embedded in `kernel.so` before invoking
the kernel. Loading and execution do not instantiate ORC or invoke LLVM compilation.

## Commands

```bash
./build/schedforge-aot compile --M=128 --N=128 --K=128 \
  --output=results/matmul_128.sfe
./build/schedforge-aot inspect --artifact=results/matmul_128.sfe
./build/schedforge-aot run --artifact=results/matmul_128.sfe --repetitions=10
./build/schedforge-aot-study --output=results/aot_deployment.csv
```

## Evidence Boundary

Format v1 is a vertical deployment slice, not a universal binary format. It is
same-target, shape-specialized, FP32 MatMul, and single-threaded. A mismatch is
rejected rather than falling back silently. Whole-graph packaging, packed
constant relocation, multi-kernel dispatch, cross-CPU feature compatibility,
GELU/residual data contracts, and archive compression remain future work.
