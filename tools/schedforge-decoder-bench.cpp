#include "schedforge/decoder_compiler.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string suite = "realistic";
    std::string output = "results/decoder_realistic.csv";
    int threads = 8;
    int repetitions = 1;
    std::uint64_t max_real_flops = 1000000000ULL;
    std::size_t max_real_weight_bytes = 256ULL * 1024 * 1024;
    bool optimize = false;
    std::string filter;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--suite=")) options.suite = argument.substr(8);
        else if (argument.starts_with("--output=")) options.output = argument.substr(9);
        else if (argument.starts_with("--threads=")) options.threads = std::stoi(argument.substr(10));
        else if (argument.starts_with("--repetitions=")) options.repetitions = std::stoi(argument.substr(14));
        else if (argument.starts_with("--max-real-gflop="))
            options.max_real_flops = static_cast<std::uint64_t>(
                std::stod(argument.substr(17)) * 1.0e9);
        else if (argument.starts_with("--max-weight-mib="))
            options.max_real_weight_bytes = static_cast<std::size_t>(
                std::stod(argument.substr(17)) * 1024.0 * 1024.0);
        else if (argument.starts_with("--filter=")) options.filter = argument.substr(9);
        else if (argument == "--optimize") options.optimize = true;
        else if (argument == "--help") {
            std::cout << "schedforge-decoder-bench [--suite=realistic|optimizer]\n"
                         "  [--output=results/decoder_realistic.csv] [--threads=8]\n"
                         "  [--repetitions=1 --max-real-gflop=1 --max-weight-mib=256]\n"
                         "  [--filter=tiny-decode --optimize]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.suite == "optimizer") options.optimize = true;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto dense_graph = schedforge::StableHLOImporter{}.importFile(
            "examples/decoder_layer.mlir");
        auto profiles = schedforge::realistic_decoder_profiles();
        if (!options.filter.empty()) {
            std::erase_if(profiles, [&](const auto& profile) {
                return profile.name.find(options.filter) == std::string::npos;
            });
        }
        if (options.suite == "optimizer") {
            std::erase_if(profiles, [](const auto& profile) {
                return profile.config.ffn == schedforge::DecoderFFNKind::MoE ||
                    (profile.name != "tiny-prefill-s128" &&
                     profile.name != "tiny-decode-kv512");
            });
        }
        std::vector<schedforge::DecoderBenchmarkRecord> records;
        records.reserve(profiles.size());
        for (const auto& profile : profiles) {
            std::cout << "[Profile] " << profile.name << " ... " << std::flush;
            auto record = schedforge::benchmark_decoder_profile(
                profile, dense_graph, options.threads, options.max_real_flops,
                options.max_real_weight_bytes, options.optimize, options.repetitions);
            std::cout << schedforge::decoder_evidence_name(record.evidence);
            if (record.evidence == schedforge::DecoderEvidenceKind::Measured)
                std::cout << " " << std::fixed << std::setprecision(3)
                          << record.measured.milliseconds << " ms, "
                          << record.measured.tokens_per_second << " token/s, error "
                          << record.measured.max_error;
            else
                std::cout << " weight=" << record.profile.estimated_weight_bytes
                          << " flops=" << record.profile.estimated_flops;
            if (record.hardware_measurements)
                std::cout << " optimizer=" << record.optimizer_speedup << "x/"
                          << record.hardware_measurements << " measurements";
            std::cout << '\n';
            records.push_back(std::move(record));
        }
        schedforge::write_decoder_benchmark_csv(options.output, records);
        if (options.optimize) {
            std::string candidates = options.output;
            const auto extension = candidates.rfind(".csv");
            if (extension != std::string::npos) candidates.insert(extension, "_candidates");
            else candidates += "_candidates.csv";
            schedforge::write_decoder_plan_candidates_csv(candidates, records);
            std::cout << "[Candidates] output=" << candidates << '\n';
        }
        const auto measured = std::count_if(records.begin(), records.end(), [](const auto& record) {
            return record.evidence == schedforge::DecoderEvidenceKind::Measured;
        });
        std::cout << "[Summary] profiles=" << records.size() << " measured=" << measured
                  << " compile-only=" << records.size() - static_cast<std::size_t>(measured)
                  << " output=" << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
