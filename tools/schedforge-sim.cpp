#include "schedforge/compiler.h"

#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    schedforge::Problem problem;
    std::string schedule_text = "order=ikj;outer=64,128,32;tile=32,64,32;micro=4,8;vector=8;unroll=4;threads=1;pack=b;prefetch=4;fuse=true";
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg.starts_with("--M=")) problem.m = std::stoi(arg.substr(4));
        else if (arg.starts_with("--N=")) problem.n = std::stoi(arg.substr(4));
        else if (arg.starts_with("--K=")) problem.k = std::stoi(arg.substr(4));
        else if (arg.starts_with("--schedule=")) schedule_text = arg.substr(11);
    }
    const auto schedule = schedforge::ScheduleDSL::parse(schedule_text);
    const auto target = schedforge::TargetInfo::detect();
    const auto result = schedforge::simulate({problem, schedule}, target);
    const auto cost = schedforge::CostModel{}.evaluate({problem, schedule}, result, target);
    std::cout << std::fixed << std::setprecision(4)
              << "target: " << target.str() << '\n'
              << "schedule: " << schedforge::ScheduleDSL::print(schedule) << '\n'
              << "L1 miss rate: " << result.l1_miss_rate() * 100.0 << "%\n"
              << "LLC miss rate: " << result.llc_miss_rate() * 100.0 << "%\n"
              << "DTLB misses: " << result.dtlb_misses << '\n'
              << "prefetch useful/issued: " << result.useful_prefetches << '/' << result.prefetched_lines << '\n'
              << "register pressure: " << result.register_pressure << '\n'
              << "estimated cycles: " << cost.total_cycles << '\n';
}
