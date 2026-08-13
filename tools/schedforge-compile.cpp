#include "schedforge/graph_compiler.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct Options {
    std::string input;
    std::string output = "model.sfe";
    std::string target = "native-cpu";
    bool autotune = false;
    bool run = true;
    schedforge::MLPConfig mlp;
    int threads = 8;
    std::size_t top_k = 8;
    std::filesystem::path measurement_database;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--target=")) options.target = argument.substr(9);
        else if (argument == "--autotune") options.autotune = true;
        else if (argument == "--no-run") options.run = false;
        else if (argument.starts_with("--threads=")) options.threads = std::stoi(argument.substr(10));
        else if (argument.starts_with("--top-k=")) options.top_k = static_cast<std::size_t>(std::stoul(argument.substr(8)));
        else if (argument.starts_with("--measurement-db=")) options.measurement_database = argument.substr(17);
        else if (argument.starts_with("--batch=")) options.mlp.batch = std::stoi(argument.substr(8));
        else if (argument.starts_with("--sequence=")) options.mlp.sequence = std::stoi(argument.substr(11));
        else if (argument.starts_with("--hidden=")) options.mlp.hidden = std::stoi(argument.substr(9));
        else if (argument.starts_with("--intermediate=")) options.mlp.intermediate = std::stoi(argument.substr(15));
        else if (argument == "-o" && index + 1 < argc) options.output = argv[++index];
        else if (argument == "--help") {
            std::cout << "schedforge-compile MODEL.mlir [--target=native-cpu] [--autotune] [-o model.sfe]\n"
                         "                   [--batch=1 --sequence=16 --hidden=64 --intermediate=128]\n"
                         "                   [--threads=8 --top-k=8 --measurement-db=records.csv --no-run]\n";
            std::exit(0);
        } else if (!argument.starts_with('-')) options.input = argument;
        else throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.input.empty()) throw std::invalid_argument("missing StableHLO input file");
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        auto graph = schedforge::StableHLOImporter{}.importFile(options.input);
        const auto data = schedforge::make_mlp_data(options.mlp);
        schedforge::GraphCompileOptions compile_options;
        compile_options.target = options.target;
        compile_options.autotune = options.autotune;
        compile_options.max_threads = options.threads;
        compile_options.top_k = options.top_k;
        compile_options.measurement_database = options.measurement_database;
        const auto plan = schedforge::GraphCompiler{}.compile(
            std::move(graph), compile_options, &options.mlp, &data);
        plan.save(options.output);

        std::cout << "[Import]\nStableHLO operations: " << plan.graph.operations().size() << "\n\n"
                  << "[Shape Inference]\ninput: " << plan.graph.values().front().type.str() << "\n\n"
                  << "[Graph Optimization]\n";
        for (std::size_t index = 0; index < plan.dispatches.size(); ++index)
            std::cout << "fused dispatch #" << index << ": " << plan.dispatches[index].epilogue << '\n';
        std::cout << "\n[Layout Planning]\nremoved layout conversions: "
                  << plan.layout_conversions_removed << "\n\n"
                  << "[Memory Planning]\nnaive temporary memory: " << plan.memory.naive_bytes << " bytes\n"
                  << "planned workspace: " << plan.memory.workspace_bytes << " bytes\n\n"
                  << "[Kernel Compiler]\n";
        for (std::size_t index = 0; index < plan.dispatches.size(); ++index) {
            const auto& dispatch = plan.dispatches[index];
            std::cout << "dispatch #" << index << ": " << schedforge::schedule_name(dispatch.schedule)
                      << " micro=" << dispatch.schedule.mr << 'x' << dispatch.schedule.nr
                      << " threads=" << dispatch.schedule.threads << '\n';
        }
        std::cout << "\n[AutoScheduler]\n";
        for (std::size_t index = 0; index < plan.dispatches.size(); ++index) {
            const auto& dispatch = plan.dispatches[index];
            std::cout << "dispatch #" << index << ": candidates="
                      << dispatch.tuning_search_space << " hardware measured="
                      << dispatch.hardware_measurements << " cache="
                      << (dispatch.tuning_cache_hit ? "hit" : "miss")
                      << " source=" << dispatch.tuning_source << '\n';
        }
        std::cout << "\n[LLVM]\nkernels generated: " << plan.llvm_ir.size()
                  << "\nJIT compile: " << std::fixed << std::setprecision(3)
                  << plan.llvm_compile_milliseconds << " ms\n";
        if (options.run && plan.dispatches.size() == 2) {
            const auto measured = schedforge::execute_mlp(plan, options.mlp, data, 1, 5);
            std::cout << "\n[Runtime]\nbackend: native scheduled-loop dispatch\nexecution: "
                      << std::fixed << std::setprecision(3)
                      << measured.milliseconds << " ms\n\n[Validation]\nmax error: "
                      << measured.max_error << "\n\n[Hardware]\n" << plan.hardware << '\n';
            return measured.max_error <= 1.0e-3 ? 0 : 1;
        }
        if (options.run) {
            std::cout << "\n[Runtime]\ngeneric graph execution is not enabled for this imported graph; "
                         "the executable plan and LLVM kernels were emitted.\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
