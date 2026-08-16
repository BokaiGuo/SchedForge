#include "schedforge/compiler.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::filesystem::path output = "results/aot_deployment.csv";
        int repetitions = 5;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument.starts_with("--output=")) output = argument.substr(9);
            else if (argument.starts_with("--repetitions="))
                repetitions = std::stoi(argument.substr(14));
            else throw std::invalid_argument("unknown option: " + argument);
        }
        std::filesystem::create_directories(output.parent_path());
        std::ofstream csv(output);
        if (!csv) throw std::runtime_error("cannot open output CSV");
        csv << "m,n,k,jit_compile_ms,jit_run_ms,aot_compile_ms,aot_link_ms,aot_load_ms,aot_run_ms,aot_gflops,max_error\n";
        schedforge::Schedule schedule;
        schedule.mr = 4;
        schedule.nr = 8;
        schedule.vector_width = 8;
        schedule.threads = 1;
        schedule.pack_b = true;
        for (const int extent : std::vector<int>{64, 128, 256}) {
            const schedforge::Problem problem{extent, extent, extent, true, true};
            const auto loop = schedforge::apply_schedule(problem, schedule);
            const auto data = schedforge::make_data(problem, static_cast<std::uint32_t>(extent));
            const auto jit = schedforge::LLVMJITBackend{}.benchmark(loop, data, 1, repetitions);
            const auto package = output.parent_path() /
                ("aot_matmul_" + std::to_string(extent) + ".sfe");
            const auto compiled = schedforge::create_aot_package(loop, package);
            const auto aot = schedforge::benchmark_aot_package(
                package, data, 1, repetitions);
            csv << extent << ',' << extent << ',' << extent << ','
                << jit.compile_milliseconds << ',' << jit.execution_milliseconds << ','
                << compiled.compile_milliseconds << ',' << compiled.link_milliseconds << ','
                << aot.load_milliseconds << ',' << aot.execution_milliseconds << ','
                << aot.gflops << ',' << aot.max_error << '\n';
            std::cout << extent << "^3 jit=" << jit.execution_milliseconds
                      << " ms aot=" << aot.execution_milliseconds
                      << " ms error=" << aot.max_error << '\n';
            std::filesystem::remove_all(package);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
