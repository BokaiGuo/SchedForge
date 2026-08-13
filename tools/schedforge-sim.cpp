#include "schedforge/compiler.h"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    schedforge::Problem problem;
    schedforge::SimulationOptions simulation_options;
    std::string schedule_text = "order=ikj;outer=64,128,32;tile=32,64,32;micro=4,8;vector=8;unroll=4;threads=1;pack=b;prefetch=4;fuse=true";
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg.starts_with("--M=")) problem.m = std::stoi(arg.substr(4));
        else if (arg.starts_with("--N=")) problem.n = std::stoi(arg.substr(4));
        else if (arg.starts_with("--K=")) problem.k = std::stoi(arg.substr(4));
        else if (arg.starts_with("--schedule=")) schedule_text = arg.substr(11);
        else if (arg.starts_with("--sample-max=")) simulation_options.max_extent = std::stoi(arg.substr(13));
    }
    const auto schedule = schedforge::ScheduleDSL::parse(schedule_text);
    const auto target = schedforge::TargetInfo::detect();
    const auto loop = schedforge::apply_schedule(problem, schedule);
    const auto result = schedforge::simulate(loop, target, simulation_options);
    const auto cost = schedforge::CostModel{}.evaluate(loop, result, target);
    std::cout << std::fixed << std::setprecision(4)
              << "target: " << target.str() << '\n'
              << "schedule: " << schedforge::ScheduleDSL::print(schedule) << '\n'
              << "L1 miss rate: " << result.l1_miss_rate() * 100.0 << "%\n"
              << "LLC miss rate: " << result.llc_miss_rate() * 100.0 << "%\n"
              << "DTLB misses: " << result.dtlb_misses << '\n'
              << "prefetch useful/issued: " << result.useful_prefetches << '/' << result.prefetched_lines << '\n'
              << "register pressure: " << result.register_pressure << '\n'
              << "simulated extent: " << result.sampled_m << 'x' << result.sampled_n
              << 'x' << result.sampled_k << '\n'
              << "estimated cycles: " << cost.total_cycles << '\n';
}
