#include "schedforge/schedforge.h"
#include "schedforge/compiler.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

struct Options {
    schedforge::Problem problem;
    int threads = 1;
    int warmup = 1;
    int repetitions = 5;
    std::size_t top_k = 8;
    bool autoschedule = false;
    bool dump_ir = false;
    bool experiment = false;
    bool calibrate = false;
    std::string csv = "results/optimization_ladder.csv";
};

int integer_value(const std::string& arg, const std::string& prefix) {
    return std::stoi(arg.substr(prefix.size()));
}

Options parse(int argc, char** argv) {
    Options options;
    options.threads = std::min(8U, std::max(1U, std::thread::hardware_concurrency()));
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg.starts_with("--M=")) options.problem.m = integer_value(arg, "--M=");
        else if (arg.starts_with("--N=")) options.problem.n = integer_value(arg, "--N=");
        else if (arg.starts_with("--K=")) options.problem.k = integer_value(arg, "--K=");
        else if (arg.starts_with("--threads=")) options.threads = integer_value(arg, "--threads=");
        else if (arg.starts_with("--warmup=")) options.warmup = integer_value(arg, "--warmup=");
        else if (arg.starts_with("--repetitions=")) options.repetitions = integer_value(arg, "--repetitions=");
        else if (arg.starts_with("--top-k=")) options.top_k = static_cast<std::size_t>(integer_value(arg, "--top-k="));
        else if (arg.starts_with("--csv=")) options.csv = arg.substr(6);
        else if (arg == "--autoschedule") options.autoschedule = true;
        else if (arg == "--dump-ir") options.dump_ir = true;
        else if (arg == "--experiment") options.experiment = true;
        else if (arg == "--calibrate") options.calibrate = true;
        else if (arg == "--help") {
            std::cout << "schedforge-bench [--M=128 --N=128 --K=128] [--threads=8]\n"
                         "                  [--autoschedule] [--experiment] [--dump-ir] [--calibrate]\n"
                         "                  [--warmup=1 --repetitions=5 --top-k=8 --csv=FILE]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + arg);
    }
    return options;
}

void print_result(const schedforge::SearchResult& result) {
    std::cout << "Search space: " << result.search_space << " schedules\n"
              << "Static pruning: " << result.search_space << " -> " << result.after_static_pruning << "\n"
              << "Cache simulation: " << result.after_static_pruning << " -> " << result.benchmarked << "\n\n"
              << "Best schedule:\n  " << schedforge::schedule_name(result.schedule) << "\n"
              << "  tile = " << result.schedule.bm << 'x' << result.schedule.bn << 'x' << result.schedule.bk << "\n"
              << "  register block = " << result.schedule.mr << 'x' << result.schedule.nr << "\n"
              << "  outer tile = " << result.schedule.mc << 'x' << result.schedule.nc << 'x' << result.schedule.kc << "\n"
              << "  vector width = " << result.schedule.vector_width << "\n"
              << "  packing = " << (result.schedule.pack_a ? "A" : "")
              << (result.schedule.pack_b ? "B" : "none") << "\n"
              << "  prefetch distance = " << result.schedule.prefetch_distance << "\n"
              << "  threads = " << result.schedule.threads << "\n\n"
              << std::fixed << std::setprecision(3)
              << "Performance:\n  execution = " << result.benchmark.milliseconds << " ms\n"
              << "  GFLOPS = " << result.benchmark.gflops << "\n"
              << "  max error = " << result.benchmark.max_error << "\n"
              << "  simulated L1 miss = " << result.simulation.l1_miss_rate() * 100.0 << "%\n"
              << "  simulated LLC miss = " << result.simulation.llc_miss_rate() * 100.0 << "%\n"
              << "  simulated DTLB misses = " << result.simulation.dtlb_misses << "\n"
              << "  register pressure = " << result.simulation.register_pressure << "\n"
              << "  simulator ranking = #" << result.simulator_rank << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        schedforge::GraphIR graph{options.problem};
        const auto data = schedforge::make_data(options.problem);
        if (options.dump_ir) {
            std::cout << "Graph IR:\n" << graph.dump() << '\n';
            schedforge::LowerToLoopsPass lower;
            schedforge::LoopInterchangePass interchange;
            schedforge::LoopTilingPass tiling;
            schedforge::VectorizePass vectorize;
            schedforge::FusionPass fusion;
            auto loop = lower.run(graph);
            interchange.run(loop); tiling.run(loop, 32, 64, 32); vectorize.run(loop, 8); fusion.run(loop);
            loop.schedule.threads = options.threads;
            std::cout << "Optimized Loop IR:\n" << loop.dump() << '\n';
        }
        if (options.experiment) {
            schedforge::write_experiment_csv(options.csv, graph, data, options.threads,
                                              options.warmup, options.repetitions);
            std::cout << "Experiment CSV: " << options.csv << '\n';
        }
        if (options.autoschedule || (!options.dump_ir && !options.experiment)) {
            auto target = schedforge::TargetInfo::detect();
            if (options.calibrate) {
                const schedforge::HardwareCalibrator calibrator;
                target = calibrator.apply(target, calibrator.run());
                std::cout << "Calibrated target: " << target.str() << " bandwidth="
                          << target.memory_bandwidth_gbps << " GB/s\n";
            }
            schedforge::Compiler compiler(target);
            print_result(compiler.compileAndTune(graph, data, options.threads, options.top_k,
                                                  options.warmup, options.repetitions));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
