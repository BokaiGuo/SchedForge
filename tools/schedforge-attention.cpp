#include "schedforge/attention_compiler.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

schedforge::AttentionLoweringStrategy parse_strategy(const std::string& value) {
    if (value == "materialized") return schedforge::AttentionLoweringStrategy::Materialized;
    if (value == "tiled") return schedforge::AttentionLoweringStrategy::TiledMaterialized;
    if (value == "io-aware" || value == "flash")
        return schedforge::AttentionLoweringStrategy::IOAware;
    if (value == "auto-io-aware")
        return schedforge::AttentionLoweringStrategy::AutoScheduledIOAware;
    if (value == "decode" || value == "split-kv")
        return schedforge::AttentionLoweringStrategy::SplitKVDecode;
    throw std::invalid_argument("unknown attention strategy: " + value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        schedforge::AttentionConfig config;
        schedforge::AttentionPlanOptions options;
        options.schedule.threads = 8;
        bool automatic = true;
        int repetitions = 5;
        std::string output = "results/attention_prefill.sfe";
        std::string experiment_csv;
        std::string scaling_csv;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument.starts_with("--batch=")) config.batch = std::stoi(argument.substr(8));
            else if (argument.starts_with("--q-heads=")) config.query_heads = std::stoi(argument.substr(10));
            else if (argument.starts_with("--kv-heads=")) config.kv_heads = std::stoi(argument.substr(11));
            else if (argument.starts_with("--sq=")) config.sequence_query = std::stoi(argument.substr(5));
            else if (argument.starts_with("--sk=")) config.sequence_kv = std::stoi(argument.substr(5));
            else if (argument.starts_with("--head-dim=")) config.head_dim = std::stoi(argument.substr(11));
            else if (argument.starts_with("--value-dim=")) config.head_dim_value = std::stoi(argument.substr(12));
            else if (argument.starts_with("--threads=")) options.schedule.threads = std::stoi(argument.substr(10));
            else if (argument.starts_with("--qtile=")) options.schedule.query_tile = std::stoi(argument.substr(8));
            else if (argument.starts_with("--kvtile=")) options.schedule.kv_tile = std::stoi(argument.substr(9));
            else if (argument.starts_with("--split-kv=")) options.split_kv = std::stoi(argument.substr(11));
            else if (argument.starts_with("--repetitions=")) repetitions = std::stoi(argument.substr(14));
            else if (argument.starts_with("--experiment-csv=")) experiment_csv = argument.substr(17);
            else if (argument.starts_with("--scaling-csv=")) scaling_csv = argument.substr(14);
            else if (argument == "--causal") config.causal = true;
            else if (argument == "--strategy=auto") automatic = true;
            else if (argument.starts_with("--strategy=")) {
                options.strategy = parse_strategy(argument.substr(11));
                automatic = false;
            } else if (argument == "-o" && index + 1 < argc) output = argv[++index];
            else if (argument == "--help") {
                std::cout << "schedforge-attention [--batch=1 --q-heads=8 --kv-heads=8]\n"
                             "  [--sq=128 --sk=128 --head-dim=64 --value-dim=64]\n"
                             "  [--causal --threads=8 --qtile=32 --kvtile=64]\n"
                             "  [--strategy=auto|materialized|tiled|io-aware|auto-io-aware|decode]\n"
                             "  [--experiment-csv=results/attention.csv]\n"
                             "  [--scaling-csv=results/attention_scaling.csv] [-o plan.sfe]\n";
                return 0;
            } else throw std::invalid_argument("unknown option: " + argument);
        }
        const auto data = schedforge::make_attention_data(config, 17);
        if (automatic) options = schedforge::tune_attention_plan(
            config, data, schedforge::TargetInfo::detect(), options.schedule.threads, 1, 2);
        const auto plan = schedforge::AttentionCompiler{}.compile(config, options);
        plan.save(output);
        schedforge::AttentionBenchmarkResult measured;
        if (options.strategy == schedforge::AttentionLoweringStrategy::SplitKVDecode) {
            auto cache = schedforge::make_kv_cache(config);
            schedforge::append_kv(cache, data.key, data.value, config.sequence_kv);
            measured = schedforge::execute_decode_attention(
                plan, data.query, cache, 1, repetitions);
            std::cout << "[KV Cache]\n" << cache.dump() << "\n\n";
        } else {
            measured = schedforge::execute_attention(plan, data, 1, repetitions);
        }
        if (!experiment_csv.empty())
            schedforge::write_attention_experiment_csv(
                experiment_csv, config, options.schedule.threads, repetitions);
        if (!scaling_csv.empty())
            schedforge::write_attention_scaling_csv(
                scaling_csv, config, options.schedule.threads);
        std::cout << "[Attention]\nstrategy: "
                  << schedforge::attention_strategy_name(options.strategy)
                  << "\nshape: B=" << config.batch << " Hq=" << config.query_heads
                  << " Hkv=" << config.kv_heads << " Sq=" << config.sequence_query
                  << " Sk=" << config.sequence_kv << " D=" << config.head_dim << "\n\n"
                  << "[Schedule]\n" << options.schedule.dump() << '\n'
                  << "[Memory]\n" << plan.memory.dump() << "\n\n"
                  << "[Simulation]\n" << plan.simulation.dump() << "\n\n"
                  << std::fixed << std::setprecision(3)
                  << "[Runtime]\nP50 execution: " << measured.p50_milliseconds << " ms\n"
                  << "P95 execution: " << measured.p95_milliseconds << " ms\n\n"
                  << "[Validation]\nmax error: " << measured.max_error << "\n\n"
                  << "[Hardware]\n" << plan.hardware << '\n';
        return measured.max_error <= 1.0e-4 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
