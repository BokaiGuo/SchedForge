#include "schedforge/schedforge.h"
#include "schedforge/compiler.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <random>
#include <chrono>
#include <vector>

namespace schedforge {
namespace {

struct Candidate { Schedule schedule; SimulationResult simulation; };

std::vector<Schedule> search_space(int max_threads) {
    std::vector<Schedule> schedules;
    const int tiles[] = {16, 32, 64};
    const int k_tiles[] = {16, 32, 64};
    std::vector<int> threads{1};
    for (int value : {2, 4, 8}) if (value <= max_threads) threads.push_back(value);
    for (int bm : tiles) for (int bn : tiles) for (int bk : k_tiles)
        for (int mr : {1, 4, 8, 16}) for (int vector_width : {1, 8})
        for (int thread_count : threads) for (int packing_mode : {0, 1, 2})
        for (int prefetch_distance : {0, 4}) for (int unroll_k : {1, 4}) {
            Schedule schedule;
            schedule.order = LoopOrder::IKJ;
            schedule.bm = bm; schedule.bn = bn; schedule.bk = bk;
            schedule.mc = std::max(64, bm * 2);
            schedule.nc = std::max(64, bn * 2);
            schedule.kc = std::max(32, bk);
            schedule.mr = mr; schedule.nr = vector_width;
            schedule.vector_width = vector_width; schedule.threads = thread_count;
            schedule.unroll_k = unroll_k;
            schedule.pack_a = packing_mode == 2;
            schedule.pack_b = packing_mode >= 1;
            schedule.prefetch_distance = prefetch_distance;
            schedule.tiled = true; schedule.fused = true;
            schedules.push_back(schedule);
        }
    return schedules;
}

bool valid(const Schedule& schedule) {
    if (schedule.bm % schedule.mr != 0 || schedule.bn % schedule.vector_width != 0) return false;
    const auto pressure = estimate_register_pressure(schedule, TargetInfo::detect());
    if (pressure.spills) return false;
    if (schedule.pack_b && schedule.vector_width == 1) return false;
    if (schedule.pack_a && !schedule.pack_b) return false;
    if (schedule.prefetch_distance > 0 && !schedule.pack_b) return false;
    if (schedule.unroll_k > 1 && schedule.vector_width == 1) return false;
    const std::size_t working_set = 4ULL * static_cast<std::size_t>(
        schedule.bm * schedule.bk + schedule.bk * schedule.bn + schedule.bm * schedule.bn);
    return working_set <= 128 * 1024;
}

}  // namespace

std::vector<Schedule> generate_schedule_candidates(int max_threads) {
    return search_space(max_threads);
}

std::string search_strategy_name(SearchStrategy strategy) {
    if (strategy == SearchStrategy::Random) return "random";
    if (strategy == SearchStrategy::Greedy) return "greedy";
    if (strategy == SearchStrategy::Evolutionary) return "evolutionary";
    return "grid";
}

SearchComparison compare_search_strategy(const GraphIR& graph, const TensorData& data,
                                         SearchStrategy strategy, int max_threads,
                                         std::size_t budget, std::uint32_t seed) {
    const auto start = std::chrono::steady_clock::now();
    auto candidates = search_space(max_threads);
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
        [](const Schedule& schedule) { return !valid(schedule); }), candidates.end());
    std::mt19937 generator(seed);
    if (strategy == SearchStrategy::Random) {
        std::shuffle(candidates.begin(), candidates.end(), generator);
    } else {
        std::vector<std::pair<double, Schedule>> scored;
        scored.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            scored.emplace_back(simulate({graph.problem, candidate}).estimated_cycles, candidate);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
        candidates.clear();
        candidates.reserve(scored.size());
        for (auto& [score, candidate] : scored) {
            (void)score;
            candidates.push_back(std::move(candidate));
        }
    }
    if (strategy == SearchStrategy::Greedy && !candidates.empty()) {
        const Schedule seed_schedule = candidates.front();
        std::stable_sort(candidates.begin(), candidates.end(), [&](const Schedule& lhs, const Schedule& rhs) {
            auto distance = [&](const Schedule& value) {
                return std::abs(value.bm - seed_schedule.bm) + std::abs(value.bn - seed_schedule.bn) +
                       std::abs(value.bk - seed_schedule.bk) + std::abs(value.mr - seed_schedule.mr) +
                       (value.pack_b != seed_schedule.pack_b ? 32 : 0);
            };
            return distance(lhs) < distance(rhs);
        });
    }
    if (strategy == SearchStrategy::Evolutionary && candidates.size() > budget) {
        const std::size_t elite = std::min<std::size_t>(std::max<std::size_t>(1, budget / 4), candidates.size());
        std::vector<Schedule> population(candidates.begin(), candidates.begin() + elite);
        std::uniform_int_distribution<std::size_t> choose(0, elite - 1);
        while (population.size() < budget) {
            Schedule child = population[choose(generator)];
            if (generator() % 2 == 0) child.bk = child.bk == 16 ? 32 : 16;
            else child.prefetch_distance = child.prefetch_distance == 0 ? 4 : 0;
            if (valid(child)) population.push_back(child);
        }
        candidates = std::move(population);
    }
    const std::size_t count = std::min(std::max<std::size_t>(1, budget), candidates.size());
    double best_gflops = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto measured = benchmark({graph.problem, candidates[index]}, data, 0, 1);
        if (measured.max_error <= 1.0e-3) best_gflops = std::max(best_gflops, measured.gflops);
    }
    const auto end = std::chrono::steady_clock::now();
    return {strategy, candidates.size(), count, best_gflops,
            std::chrono::duration<double, std::milli>(end - start).count()};
}

SearchResult autoschedule(const GraphIR& graph, const TensorData& data,
                          int max_threads, std::size_t top_k,
                          int warmup, int repetitions) {
    return autoschedule(graph, data, TargetInfo::detect(), max_threads, top_k,
                        warmup, repetitions);
}

SearchResult autoschedule(const GraphIR& graph, const TensorData& data,
                          const TargetInfo& target, int max_threads,
                          std::size_t top_k, int warmup, int repetitions) {
    const auto all = search_space(std::max(1, max_threads));
    std::vector<Candidate> candidates;
    for (const auto& schedule : all) {
        if (!valid(schedule)) continue;
        LoopIR loop{graph.problem, schedule};
        candidates.push_back({schedule, simulate(loop, target)});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.simulation.estimated_cycles != rhs.simulation.estimated_cycles)
            return lhs.simulation.estimated_cycles < rhs.simulation.estimated_cycles;
        return std::tie(lhs.schedule.vector_width, lhs.schedule.threads, lhs.schedule.mr) >
               std::tie(rhs.schedule.vector_width, rhs.schedule.threads, rhs.schedule.mr);
    });
    const std::size_t benchmark_count = std::min(std::max<std::size_t>(1, top_k), candidates.size());
    SearchResult best;
    best.search_space = all.size();
    best.after_static_pruning = candidates.size();
    best.benchmarked = benchmark_count;
    best.benchmark.milliseconds = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < benchmark_count; ++index) {
        LoopIR loop{graph.problem, candidates[index].schedule};
        const auto measured = benchmark(loop, data, warmup, repetitions);
        if (measured.max_error <= 1.0e-3 && measured.milliseconds < best.benchmark.milliseconds) {
            best.schedule = candidates[index].schedule;
            best.simulation = candidates[index].simulation;
            best.benchmark = measured;
            best.simulator_rank = index + 1;
        }
    }
    if (!std::isfinite(best.benchmark.milliseconds)) throw std::runtime_error("no correct schedule found");
    return best;
}

void write_experiment_csv(const std::string& path, const GraphIR& graph,
                          const TensorData& data, int max_threads,
                          int warmup, int repetitions) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) std::filesystem::create_directories(output_path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open experiment output: " + path);
    out << "name,order,mc,nc,kc,bm,bn,bk,mr,nr,vector_width,unroll_k,threads,pack_a,pack_b,prefetch,fused,time_ms,gflops,max_error,l1_miss_rate,llc_miss_rate,dtlb_misses,register_pressure,estimated_cycles\n";
    std::vector<Schedule> ladder;
    Schedule naive; naive.order = LoopOrder::IJK; naive.tiled = false; naive.vector_width = 1; naive.mr = 1; naive.fused = false;
    ladder.push_back(naive);
    Schedule reordered = naive; reordered.order = LoopOrder::IKJ; ladder.push_back(reordered);
    Schedule tiled = reordered; tiled.tiled = true; tiled.bm = 32; tiled.bn = 32; tiled.bk = 32; ladder.push_back(tiled);
    Schedule vectorized = tiled; vectorized.vector_width = 8; vectorized.nr = 8; vectorized.mr = 4; ladder.push_back(vectorized);
    Schedule threaded = vectorized; threaded.threads = std::max(1, max_threads); ladder.push_back(threaded);
    Schedule fused = threaded; fused.fused = true; ladder.push_back(fused);
    Schedule packed = fused; packed.pack_b = true; packed.mc = 64; packed.nc = 128; packed.kc = 32; ladder.push_back(packed);
    Schedule prefetched = packed; prefetched.prefetch_distance = 4; ladder.push_back(prefetched);
    for (const auto& schedule : ladder) {
        LoopIR loop{graph.problem, schedule};
        const auto measured = benchmark(loop, data, warmup, repetitions);
        const auto simulated = simulate(loop);
        out << schedule_name(schedule) << ',' << (schedule.order == LoopOrder::IKJ ? "ikj" : "ijk") << ','
            << schedule.mc << ',' << schedule.nc << ',' << schedule.kc << ','
            << schedule.bm << ',' << schedule.bn << ',' << schedule.bk << ',' << schedule.mr << ',' << schedule.nr << ','
            << schedule.vector_width << ',' << schedule.unroll_k << ',' << schedule.threads << ','
            << schedule.pack_a << ',' << schedule.pack_b << ',' << schedule.prefetch_distance << ',' << schedule.fused << ','
            << measured.milliseconds << ',' << measured.gflops << ',' << measured.max_error << ','
            << simulated.l1_miss_rate() << ',' << simulated.llc_miss_rate() << ','
            << simulated.dtlb_misses << ',' << simulated.register_pressure << ','
            << simulated.estimated_cycles << '\n';
    }
}

}  // namespace schedforge
