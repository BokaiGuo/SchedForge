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
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace schedforge {
namespace {

struct Candidate {
    Schedule schedule;
    double screening_milliseconds = std::numeric_limits<double>::infinity();
};

std::string execution_key(const Schedule& schedule) {
    if (schedule.order == LoopOrder::IKJ && !schedule.pack_a && !schedule.pack_b &&
        schedule.vector_width >= 8) {
        return "register;bm=" + std::to_string(schedule.bm) +
               ";bn=" + std::to_string(schedule.bn) +
               ";mr=" + std::to_string(std::min(schedule.mr, 8)) +
               ";threads=" + std::to_string(schedule.threads) +
               ";fused=" + std::to_string(schedule.fused) +
               ";pin=" + std::to_string(schedule.pin_threads);
    }
    return ScheduleDSL::print(schedule);
}

std::vector<Schedule> search_space(int max_threads) {
    std::vector<Schedule> schedules;
    const int tiles[] = {16, 32, 64};
    const int k_tiles[] = {16, 32, 64};
    std::vector<int> threads{1};
    for (int value : {2, 4, 6, 8, 10, 12, 16}) {
        if (value <= max_threads) threads.push_back(value);
    }
    for (int bm : tiles) for (int bn : tiles) for (int bk : k_tiles)
        for (int mr : {1, 4, 6, 8, 16}) for (int vector_width : {1, 8})
        for (int thread_count : threads) for (int packing_mode : {0, 1, 2})
        for (int prefetch_distance : {0, 4}) for (int unroll_k : {1, 4}) {
            Schedule schedule;
            schedule.order = LoopOrder::IKJ;
            schedule.bm = bm; schedule.bn = bn; schedule.bk = bk;
            schedule.mc = std::max(64, bm * 2);
            schedule.nc = std::max(64, bn * 2);
            schedule.kc = std::max(32, bk);
            schedule.mr = mr;
            schedule.nr = vector_width >= 8 && packing_mode == 0 && mr <= 6 ? 16 : vector_width;
            schedule.vector_width = vector_width; schedule.threads = thread_count;
            schedule.unroll_k = unroll_k;
            schedule.pack_a = packing_mode == 2;
            schedule.pack_b = packing_mode >= 1;
            schedule.prefetch_distance = prefetch_distance;
            schedule.pin_threads = thread_count > 1;
            schedule.tiled = true; schedule.fused = true;
            schedules.push_back(schedule);
        }
    return schedules;
}

bool valid(const Schedule& schedule) {
    if (schedule.bn % schedule.vector_width != 0) return false;
    const bool register_kernel = schedule.order == LoopOrder::IKJ && !schedule.pack_a &&
        !schedule.pack_b && schedule.vector_width >= 8 && schedule.mr <= 8;
    if (!register_kernel && schedule.bm % schedule.mr != 0) return false;
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
    std::unordered_set<std::string> execution_keys;
    for (const auto& schedule : all) {
        if (!valid(schedule)) continue;
        if (!execution_keys.insert(execution_key(schedule)).second) continue;
        candidates.push_back({schedule});
    }
    const auto expected = reference(graph.problem, data);
    std::vector<float> screening_output;
    std::set<int> warmed_thread_counts;
    for (const auto& candidate : candidates) {
        if (warmed_thread_counts.insert(candidate.schedule.threads).second) {
            execute({graph.problem, candidate.schedule}, data, screening_output);
        }
    }

    std::vector<std::size_t> screening_order(candidates.size());
    for (std::size_t index = 0; index < screening_order.size(); ++index) screening_order[index] = index;
    std::mt19937 generator(0x5CEDF04EU);
    std::shuffle(screening_order.begin(), screening_order.end(), generator);
    for (const std::size_t index : screening_order) {
        const auto start = std::chrono::steady_clock::now();
        execute({graph.problem, candidates[index].schedule}, data, screening_output);
        const auto end = std::chrono::steady_clock::now();
        if (max_abs_error(expected, screening_output) <= 1.0e-3) {
            candidates[index].screening_milliseconds =
                std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        return lhs.screening_milliseconds < rhs.screening_milliseconds;
    });
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [](const Candidate& candidate) {
        return !std::isfinite(candidate.screening_milliseconds);
    }), candidates.end());
    if (candidates.empty()) throw std::runtime_error("no correct schedule found during screening");

    const std::size_t finalist_count = std::min(
        std::max<std::size_t>(1, top_k), candidates.size());
    std::vector<std::vector<float>> outputs(finalist_count);
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) {
        for (std::size_t slot = 0; slot < finalist_count; ++slot) {
            execute({graph.problem, candidates[slot].schedule}, data, outputs[slot]);
        }
    }
    std::vector<std::vector<double>> timings(finalist_count);
    std::vector<std::size_t> order(finalist_count);
    for (std::size_t slot = 0; slot < order.size(); ++slot) order[slot] = slot;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        std::shuffle(order.begin(), order.end(), generator);
        for (const std::size_t slot : order) {
            const auto start = std::chrono::steady_clock::now();
            execute({graph.problem, candidates[slot].schedule}, data, outputs[slot]);
            const auto end = std::chrono::steady_clock::now();
            timings[slot].push_back(
                std::chrono::duration<double, std::milli>(end - start).count());
        }
    }

    SearchResult best;
    best.search_space = all.size();
    best.after_static_pruning = execution_keys.size();
    best.benchmarked = candidates.size();
    best.benchmark.milliseconds = std::numeric_limits<double>::infinity();
    const double operations = 2.0 * graph.problem.m * graph.problem.n * graph.problem.k;
    for (std::size_t slot = 0; slot < finalist_count; ++slot) {
        auto& samples = timings[slot];
        std::sort(samples.begin(), samples.end());
        const double milliseconds = samples[samples.size() / 2];
        const BenchmarkResult measured{
            milliseconds,
            operations / (milliseconds * 1.0e6),
            max_abs_error(expected, outputs[slot])};
        if (measured.max_error <= 1.0e-3 && measured.milliseconds < best.benchmark.milliseconds) {
            best.schedule = candidates[slot].schedule;
            best.simulation = simulate({graph.problem, candidates[slot].schedule}, target);
            best.benchmark = measured;
            best.measurement_rank = slot + 1;
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
