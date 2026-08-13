#include "schedforge/compiler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

struct CandidateSamples {
    schedforge::Schedule schedule;
    schedforge::SimulationResult simulation;
    std::vector<double> milliseconds;
    int wins = 0;
};

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double standard_deviation(const std::vector<double>& values, double average) {
    if (values.size() < 2) return 0.0;
    double squared = 0.0;
    for (double value : values) {
        const double difference = value - average;
        squared += difference * difference;
    }
    return std::sqrt(squared / static_cast<double>(values.size() - 1));
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                  : values[middle];
}

bool valid(const schedforge::Schedule& schedule, const schedforge::TargetInfo& target) {
    if (schedule.bn % schedule.vector_width != 0) return false;
    const bool register_kernel = schedule.order == schedforge::LoopOrder::IKJ &&
        !schedule.pack_a && !schedule.pack_b && schedule.vector_width >= 8 &&
        schedule.mr <= 8;
    if (!register_kernel && schedule.bm % schedule.mr != 0) return false;
    if (schedforge::estimate_register_pressure(schedule, target).spills) return false;
    if (schedule.pack_b && schedule.vector_width == 1) return false;
    if (schedule.pack_a && !schedule.pack_b) return false;
    if (schedule.prefetch_distance > 0 && !schedule.pack_b) return false;
    if (schedule.unroll_k > 1 && schedule.vector_width == 1) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int size = 192;
    int rounds = 40;
    int candidate_count = 12;
    int threads = 8;
    bool pin = false;
    std::uint32_t seed = 20260813;
    std::string output = "results/top_resolution_samples.csv";
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--size=")) size = std::stoi(argument.substr(7));
        else if (argument.starts_with("--rounds=")) rounds = std::stoi(argument.substr(9));
        else if (argument.starts_with("--candidates=")) candidate_count = std::stoi(argument.substr(13));
        else if (argument.starts_with("--threads=")) threads = std::stoi(argument.substr(10));
        else if (argument.starts_with("--output=")) output = argument.substr(9);
        else if (argument.starts_with("--seed=")) seed = static_cast<std::uint32_t>(std::stoul(argument.substr(7)));
        else if (argument == "--pin" || argument == "--pin=true") pin = true;
        else if (argument == "--pin=false") pin = false;
    }

    const schedforge::GraphIR graph{{size, size, size, true, true}};
    const auto data = schedforge::make_data(graph.problem, seed);
    auto target = schedforge::TargetInfo::detect();
    auto schedules = schedforge::generate_schedule_candidates(threads);
    std::vector<CandidateSamples> candidates;
    for (auto schedule : schedules) {
        if (!valid(schedule, target)) continue;
        schedule.pin_threads = pin;
        const schedforge::LoopIR loop{graph.problem, schedule};
        candidates.push_back({schedule, schedforge::simulate(loop, target), {}, 0});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.simulation.estimated_cycles < rhs.simulation.estimated_cycles;
    });
    candidates.resize(std::min<std::size_t>(static_cast<std::size_t>(candidate_count), candidates.size()));

    for (auto& candidate : candidates) {
        (void)schedforge::benchmark({graph.problem, candidate.schedule}, data, 2, 1);
        candidate.milliseconds.reserve(static_cast<std::size_t>(rounds));
    }

    std::mt19937 generator(seed);
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    for (int round = 0; round < rounds; ++round) {
        std::shuffle(order.begin(), order.end(), generator);
        double best_time = std::numeric_limits<double>::infinity();
        std::size_t winner = 0;
        for (const std::size_t index : order) {
            const auto measured = schedforge::benchmark(
                {graph.problem, candidates[index].schedule}, data, 0, 1);
            candidates[index].milliseconds.push_back(measured.milliseconds);
            if (measured.milliseconds < best_time) {
                best_time = measured.milliseconds;
                winner = index;
            }
        }
        ++candidates[winner].wins;
    }

    std::filesystem::create_directories(std::filesystem::path(output).parent_path());
    std::ofstream csv(output);
    csv << "candidate,round,predicted_cycles,time_ms,schedule\n";
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        for (std::size_t round = 0; round < candidates[index].milliseconds.size(); ++round) {
            csv << index << ',' << round << ',' << candidates[index].simulation.estimated_cycles << ','
                << candidates[index].milliseconds[round] << ','
                << '"' << schedforge::ScheduleDSL::print(candidates[index].schedule) << '"' << '\n';
        }
    }

    std::cout << "workload=" << size << '^' << 3 << " rounds=" << rounds
              << " candidates=" << candidates.size() << " threads=" << threads
              << " pin=" << pin << '\n';
    std::cout << "rank,predicted_cycles,median_ms,mean_ms,cv_percent,ci95_half_ms,win_probability,schedule\n";
    double best_median = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) best_median = std::min(best_median, median(candidate.milliseconds));
    std::size_t predicted_ties = 0;
    for (std::size_t index = 1; index < candidates.size(); ++index) {
        if (candidates[index].simulation.estimated_cycles == candidates[index - 1].simulation.estimated_cycles)
            ++predicted_ties;
    }
    std::size_t unresolved_pairs = 0;
    std::size_t total_pairs = 0;
    for (std::size_t left = 0; left < candidates.size(); ++left) {
        const double left_mean = mean(candidates[left].milliseconds);
        const double left_sd = standard_deviation(candidates[left].milliseconds, left_mean);
        const double left_ci = 1.96 * left_sd / std::sqrt(static_cast<double>(rounds));
        for (std::size_t right = left + 1; right < candidates.size(); ++right) {
            const double right_mean = mean(candidates[right].milliseconds);
            const double right_sd = standard_deviation(candidates[right].milliseconds, right_mean);
            const double right_ci = 1.96 * right_sd / std::sqrt(static_cast<double>(rounds));
            ++total_pairs;
            if (std::abs(left_mean - right_mean) <= left_ci + right_ci) ++unresolved_pairs;
        }
    }
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const double average = mean(candidates[index].milliseconds);
        const double sd = standard_deviation(candidates[index].milliseconds, average);
        const double candidate_median = median(candidates[index].milliseconds);
        std::cout << index + 1 << ',' << candidates[index].simulation.estimated_cycles << ','
                  << std::fixed << std::setprecision(6) << candidate_median << ',' << average << ','
                  << (average > 0.0 ? 100.0 * sd / average : 0.0) << ','
                  << 1.96 * sd / std::sqrt(static_cast<double>(rounds)) << ','
                  << static_cast<double>(candidates[index].wins) / rounds << ','
                  << schedforge::ScheduleDSL::print(candidates[index].schedule) << '\n';
    }
    const double predicted_best_regret =
        100.0 * (median(candidates.front().milliseconds) / best_median - 1.0);
    std::cout << "predicted_adjacent_ties=" << predicted_ties << '\n'
              << "unresolved_pair_fraction="
              << (total_pairs ? static_cast<double>(unresolved_pairs) / total_pairs : 0.0) << '\n'
              << "predicted_best_regret_percent=" << predicted_best_regret << '\n'
              << "samples_csv=" << output << '\n';
}
