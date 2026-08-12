#include "schedforge/compiler.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    schedforge::Problem problem;
    std::string schedule_text = "order=ikj;outer=64,128,32;tile=32,64,32;micro=4,8;vector=8;unroll=4;threads=1;pack=b;prefetch=4;fuse=true";
    bool llvm_jit = false;
    bool bf16 = false;
    bool int8 = false;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg.starts_with("--M=")) problem.m = std::stoi(arg.substr(4));
        else if (arg.starts_with("--N=")) problem.n = std::stoi(arg.substr(4));
        else if (arg.starts_with("--K=")) problem.k = std::stoi(arg.substr(4));
        else if (arg.starts_with("--schedule=")) schedule_text = arg.substr(11);
        else if (arg == "--backend=llvm") llvm_jit = true;
        else if (arg == "--backend=bf16") bf16 = true;
        else if (arg == "--backend=int8") int8 = true;
    }
    const auto schedule = schedforge::ScheduleDSL::parse(schedule_text);
    const auto data = schedforge::make_data(problem);
    if (llvm_jit) {
        const auto measured = schedforge::LLVMJITBackend{}.benchmark({problem}, data, schedule, 1, 5);
        std::cout << "backend: llvm-orc-jit\n"
                  << "compile_ms: " << measured.compile_milliseconds << '\n'
                  << "time_ms: " << measured.execution_milliseconds << '\n'
                  << "gflops: " << measured.gflops << '\n'
                  << "max_error: " << measured.max_error << '\n'
                  << "vector_instructions: " << measured.assembly_report.vector_instructions << '\n'
                  << "fma_instructions: " << measured.assembly_report.fma_instructions << '\n'
                  << "spill_pattern: " << measured.assembly_report.has_spill_pattern << '\n';
        return measured.max_error <= 1.0e-3 ? 0 : 1;
    }
    if (bf16 || int8) {
        const auto measured = bf16
            ? schedforge::benchmark_bf16(problem, data, 3)
            : schedforge::benchmark_int8(problem, data, 3);
        std::cout << "backend: " << (bf16 ? "bf16" : "int8") << '\n'
                  << "time_ms: " << measured.milliseconds << '\n'
                  << "gops: " << measured.gflops << '\n'
                  << "max_error_vs_fp32: " << measured.max_error << '\n';
        return 0;
    }
    const auto measured = schedforge::benchmark({problem, schedule}, data, 1, 5);
    std::cout << "schedule: " << schedforge::ScheduleDSL::print(schedule) << '\n'
              << "time_ms: " << measured.milliseconds << '\n'
              << "gflops: " << measured.gflops << '\n'
              << "max_error: " << measured.max_error << '\n';
    return measured.max_error <= 1.0e-3 ? 0 : 1;
}
