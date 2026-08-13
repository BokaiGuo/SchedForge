#include "schedforge/attention_compiler.h"
#include "schedforge/compiler.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Options {
    std::filesystem::path output = "results/llvm_codegen_study.csv";
    std::filesystem::path attention_output = "results/fused_attention_llvm.csv";
    int repetitions = 11;
    int threads = 8;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--output=")) options.output = argument.substr(9);
        else if (argument.starts_with("--attention-output="))
            options.attention_output = argument.substr(19);
        else if (argument.starts_with("--repetitions="))
            options.repetitions = std::stoi(argument.substr(14));
        else if (argument.starts_with("--threads="))
            options.threads = std::stoi(argument.substr(10));
        else if (argument == "--help") {
            std::cout << "schedforge-codegen-study [--output=results/llvm_codegen_study.csv] "
                         "[--attention-output=results/fused_attention_llvm.csv] "
                         "[--repetitions=11 --threads=8]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + argument);
    }
    return options;
}
}

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        if (options.output.has_parent_path())
            std::filesystem::create_directories(options.output.parent_path());
        std::ofstream csv(options.output);
        if (!csv) throw std::runtime_error("cannot open output CSV");
        csv << "shape,threads,schedule,native_ms,native_gflops,llvm_ms,llvm_gflops,"
               "native_gflops_over_llvm,compile_ms,instructions,vector_instructions,fma_instructions,"
               "loads,stores,branches,address_instructions,stack_accesses,spill,max_error\n";
        const std::vector<schedforge::Problem> problems = {
            {192, 192, 192, true, true},
            {256, 256, 256, true, true},
            {512, 512, 512, true, true}
        };
        for (const auto& problem : problems) {
            schedforge::Schedule schedule;
            schedule.bm = 32;
            schedule.bn = 64;
            schedule.bk = 32;
            schedule.mr = 6;
            schedule.nr = 16;
            schedule.vector_width = 8;
            schedule.threads = options.threads;
            schedule.fused = true;
            const auto loop = schedforge::apply_schedule(problem, schedule);
            const auto data = schedforge::make_data(problem, 71);
            const auto native = schedforge::benchmark(loop, data, 2, options.repetitions);
            const auto llvm = schedforge::LLVMJITBackend{}.benchmark(
                loop, data, 2, options.repetitions);
            const double ratio = llvm.gflops > 0.0 ? native.gflops / llvm.gflops : 0.0;
            const auto& assembly = llvm.assembly_report;
            csv << problem.m << 'x' << problem.n << 'x' << problem.k << ','
                << llvm.threads << ",\"" << schedforge::ScheduleDSL::print(schedule) << "\","
                << std::fixed << std::setprecision(6)
                << native.milliseconds << ',' << native.gflops << ','
                << llvm.execution_milliseconds << ',' << llvm.gflops << ',' << ratio << ','
                << llvm.compile_milliseconds << ',' << assembly.instructions << ','
                << assembly.vector_instructions << ',' << assembly.fma_instructions << ','
                << assembly.loads << ',' << assembly.stores << ',' << assembly.branches << ','
                << assembly.address_instructions << ',' << assembly.stack_accesses << ','
                << (assembly.has_spill_pattern ? 1 : 0) << ',' << llvm.max_error << '\n';
            std::cout << problem.m << '^' << 3 << " native=" << native.gflops
                      << " LLVM=" << llvm.gflops << " GFLOPS gap=" << ratio
                      << "x error=" << llvm.max_error << '\n';
        }
        std::cout << "output=" << options.output << '\n';
        if (options.attention_output.has_parent_path())
            std::filesystem::create_directories(options.attention_output.parent_path());
        std::ofstream attention_csv(options.attention_output);
        if (!attention_csv) throw std::runtime_error("cannot open Attention output CSV");
        attention_csv << "profile,batch,q_heads,kv_heads,sq,sk,head_dim,threads,native_ms,"
                         "llvm_ms,llvm_speed_fraction,compile_ms,instructions,vector_instructions,"
                         "branches,stack_accesses,spill,max_error\n";
        struct AttentionProfile {
            std::string name;
            schedforge::AttentionConfig config;
            schedforge::AttentionLoweringStrategy native_strategy;
        };
        const std::vector<AttentionProfile> attention_profiles = {
            {"mha-prefill-s128", {1, 8, 8, 128, 128, 64, 64, true},
             schedforge::AttentionLoweringStrategy::IOAware},
            {"gqa-prefill-s128", {1, 8, 2, 128, 128, 64, 64, true},
             schedforge::AttentionLoweringStrategy::IOAware},
            {"gqa-decode-kv1024", {1, 8, 2, 1, 1024, 64, 64, true},
             schedforge::AttentionLoweringStrategy::SplitKVDecode}
        };
        for (const auto& profile : attention_profiles) {
            auto plan_options = schedforge::select_attention_plan(
                profile.config, schedforge::TargetInfo::detect(), options.threads);
            plan_options.strategy = profile.native_strategy;
            plan_options.schedule.threads = options.threads;
            const auto data = schedforge::make_attention_data(profile.config, 73);
            const auto plan = schedforge::AttentionCompiler{}.compile(
                profile.config, plan_options);
            schedforge::AttentionBenchmarkResult native;
            if (profile.native_strategy == schedforge::AttentionLoweringStrategy::SplitKVDecode) {
                auto cache = schedforge::make_kv_cache(profile.config);
                schedforge::append_kv(cache, data.key, data.value, profile.config.sequence_kv);
                native = schedforge::execute_decode_attention(
                    plan, data.query, cache, 2, options.repetitions);
            } else {
                native = schedforge::execute_attention(plan, data, 2, options.repetitions);
            }
            const auto llvm = schedforge::execute_fused_attention_llvm(
                plan, data, 2, options.repetitions);
            const double ratio = llvm.execution_milliseconds > 0.0
                ? native.p50_milliseconds / llvm.execution_milliseconds : 0.0;
            attention_csv << profile.name << ',' << profile.config.batch << ','
                << profile.config.query_heads << ',' << profile.config.kv_heads << ','
                << profile.config.sequence_query << ',' << profile.config.sequence_kv << ','
                << profile.config.head_dim << ',' << llvm.threads << ','
                << std::fixed << std::setprecision(6) << native.p50_milliseconds << ','
                << llvm.execution_milliseconds << ',' << ratio << ','
                << llvm.compile_milliseconds << ',' << llvm.assembly_report.instructions << ','
                << llvm.assembly_report.vector_instructions << ','
                << llvm.assembly_report.branches << ','
                << llvm.assembly_report.stack_accesses << ','
                << (llvm.assembly_report.has_spill_pattern ? 1 : 0) << ','
                << llvm.max_error << '\n';
            std::cout << profile.name << " native=" << native.p50_milliseconds
                      << " ms LLVM=" << llvm.execution_milliseconds << " ms ratio="
                      << ratio << " error=" << llvm.max_error << '\n';
        }
        std::cout << "attention_output=" << options.attention_output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
