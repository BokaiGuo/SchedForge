#include "schedforge/compiler.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string command;
    std::filesystem::path artifact = "results/matmul.sfe";
    schedforge::Problem problem;
    std::string schedule = "order=ikj;outer=64,128,32;tile=32,64,32;micro=4,8;vector=8;unroll=4;threads=1;pack=b;prefetch=4;fuse=true";
    int warmup = 1;
    int repetitions = 5;
};

Options parse(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("usage: schedforge-aot <compile|inspect|run> [options]");
    Options options;
    options.command = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--artifact=")) options.artifact = argument.substr(11);
        else if (argument.starts_with("--output=")) options.artifact = argument.substr(9);
        else if (argument.starts_with("--M=")) options.problem.m = std::stoi(argument.substr(4));
        else if (argument.starts_with("--N=")) options.problem.n = std::stoi(argument.substr(4));
        else if (argument.starts_with("--K=")) options.problem.k = std::stoi(argument.substr(4));
        else if (argument.starts_with("--schedule=")) options.schedule = argument.substr(11);
        else if (argument.starts_with("--warmup=")) options.warmup = std::stoi(argument.substr(9));
        else if (argument.starts_with("--repetitions=")) options.repetitions = std::stoi(argument.substr(14));
        else if (argument == "--no-bias") options.problem.bias = false;
        else if (argument == "--no-relu") options.problem.relu = false;
        else throw std::invalid_argument("unknown option: " + argument);
    }
    return options;
}

void print_manifest(const schedforge::AOTManifest& manifest) {
    std::cout << manifest.dump();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.command == "compile") {
            const auto schedule = schedforge::ScheduleDSL::parse(options.schedule);
            const auto result = schedforge::create_aot_package(
                schedforge::apply_schedule(options.problem, schedule), options.artifact);
            std::cout << "artifact=" << result.path << '\n'
                      << "compile_ms=" << result.compile_milliseconds << '\n'
                      << "link_ms=" << result.link_milliseconds << '\n';
            print_manifest(result.manifest);
            return 0;
        }
        if (options.command == "inspect") {
            print_manifest(schedforge::inspect_aot_package(options.artifact));
            return 0;
        }
        if (options.command == "run") {
            const auto manifest = schedforge::inspect_aot_package(options.artifact);
            const auto measured = schedforge::benchmark_aot_package(
                options.artifact, schedforge::make_data(manifest.problem, 101),
                options.warmup, options.repetitions);
            std::cout << "backend=aot-dlopen\n"
                      << "load_ms=" << measured.load_milliseconds << '\n'
                      << "time_ms=" << measured.execution_milliseconds << '\n'
                      << "gflops=" << measured.gflops << '\n'
                      << "max_error=" << measured.max_error << '\n';
            return measured.max_error <= 1.0e-3 ? 0 : 1;
        }
        throw std::invalid_argument("unknown command: " + options.command);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
