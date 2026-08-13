#include "schedforge/decoder_compiler.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string input;
    std::string output = "decoder_layer.sfe";
    schedforge::DecoderConfig config;
    schedforge::DecoderCompileOptions compile;
    int repetitions = 3;
};

Options parse(int argc, char** argv) {
    Options options;
    options.compile.max_threads = 8;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--batch=")) options.config.batch = std::stoi(argument.substr(8));
        else if (argument.starts_with("--sequence=")) options.config.sequence = std::stoi(argument.substr(11));
        else if (argument.starts_with("--hidden=")) options.config.hidden = std::stoi(argument.substr(9));
        else if (argument.starts_with("--intermediate=")) options.config.intermediate = std::stoi(argument.substr(15));
        else if (argument.starts_with("--q-heads=")) options.config.query_heads = std::stoi(argument.substr(10));
        else if (argument.starts_with("--kv-heads=")) options.config.kv_heads = std::stoi(argument.substr(11));
        else if (argument.starts_with("--head-dim=")) options.config.head_dim = std::stoi(argument.substr(11));
        else if (argument.starts_with("--experts=")) options.config.experts = std::stoi(argument.substr(10));
        else if (argument.starts_with("--top-k=")) options.config.top_k = std::stoi(argument.substr(8));
        else if (argument.starts_with("--threads=")) options.compile.max_threads = std::stoi(argument.substr(10));
        else if (argument.starts_with("--repetitions=")) options.repetitions = std::stoi(argument.substr(14));
        else if (argument == "--moe") options.config.ffn = schedforge::DecoderFFNKind::MoE;
        else if (argument == "--tune-attention") options.compile.tune_attention = true;
        else if (argument == "--non-causal") options.config.causal = false;
        else if (argument == "-o" && index + 1 < argc) options.output = argv[++index];
        else if (argument == "--help") {
            std::cout << "schedforge-decoder MODEL.mlir [--batch=1 --sequence=16 --hidden=64]\n"
                         "  [--intermediate=128 --q-heads=4 --kv-heads=2 --head-dim=16]\n"
                         "  [--moe --experts=4 --top-k=2] [--threads=8 --tune-attention]\n"
                         "  [--repetitions=3 --non-causal] [-o decoder.sfe]\n";
            std::exit(0);
        } else if (!argument.starts_with('-')) options.input = argument;
        else throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.input.empty()) throw std::invalid_argument("missing StableHLO decoder input");
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto imported = schedforge::StableHLOImporter{}.importFile(options.input);
        const auto data = schedforge::make_decoder_data(options.config, 71);
        const auto plan = schedforge::DecoderCompiler{}.compile(
            imported, options.config, data, options.compile);
        plan.save(options.output);
        const auto measured = schedforge::execute_decoder_layer(
            plan, data, 1, options.repetitions);
        std::cout << "[Import]\nStableHLO operations: " << plan.imported_graph.operations().size()
                  << "\n\n[Fusion]\n" << plan.fusion.dump()
                  << "\n\n[Constants]\n";
        for (const auto& constant : plan.constants) std::cout << constant.dump() << '\n';
        std::cout << "\n[Memory]\nnaive: " << plan.memory.naive_bytes
                  << " bytes\nworkspace: " << plan.memory.workspace_bytes
                  << " bytes\nmemory planning: " << plan.memory_planning_milliseconds
                  << " ms\n\n[Plan Policy]\n" << plan.policy.dump()
                  << "\n\n[Runtime]\nFFN: "
                  << schedforge::decoder_ffn_name(options.config.ffn)
                  << "\nexecution: " << std::fixed << std::setprecision(3)
                  << measured.milliseconds << " ms\nattention: "
                  << measured.attention_milliseconds << " ms\nprojections: "
                  << measured.projection_milliseconds << " ms\nnorm+rope: "
                  << measured.norm_rope_milliseconds << " ms\nffn: "
                  << measured.ffn_milliseconds << " ms\ndispatch overhead: "
                  << measured.dispatch_overhead_milliseconds << " ms\ntokens/s: "
                  << measured.tokens_per_second << "\ncompile: " << plan.compile_milliseconds
                  << " ms\nLLVM JIT: " << plan.llvm_compile_milliseconds
                  << " ms\n\n[Validation]\nmax error: "
                  << measured.max_error << "\n\n[Hardware]\n" << plan.hardware << '\n';
        return measured.max_error <= 1.0e-3 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
