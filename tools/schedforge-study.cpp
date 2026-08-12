#include "schedforge/compiler.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

double spearman(std::vector<double> predicted, std::vector<double> measured) {
    const std::size_t count = predicted.size();
    std::vector<std::size_t> predicted_order(count), measured_order(count);
    std::iota(predicted_order.begin(), predicted_order.end(), 0);
    std::iota(measured_order.begin(), measured_order.end(), 0);
    std::sort(predicted_order.begin(), predicted_order.end(), [&](auto lhs, auto rhs) { return predicted[lhs] < predicted[rhs]; });
    std::sort(measured_order.begin(), measured_order.end(), [&](auto lhs, auto rhs) { return measured[lhs] < measured[rhs]; });
    std::vector<double> predicted_rank(count), measured_rank(count);
    for (std::size_t rank = 0; rank < count; ++rank) {
        predicted_rank[predicted_order[rank]] = static_cast<double>(rank);
        measured_rank[measured_order[rank]] = static_cast<double>(rank);
    }
    double squared_difference = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double difference = predicted_rank[index] - measured_rank[index];
        squared_difference += difference * difference;
    }
    const double n = static_cast<double>(count);
    return count < 2 ? 1.0 : 1.0 - 6.0 * squared_difference / (n * (n * n - 1.0));
}

double top_k_recall(const std::vector<double>& predicted, const std::vector<double>& measured,
                    std::size_t top_k) {
    const std::size_t count = predicted.size();
    top_k = std::min(top_k, count);
    std::vector<std::size_t> predicted_order(count), measured_order(count);
    std::iota(predicted_order.begin(), predicted_order.end(), 0);
    std::iota(measured_order.begin(), measured_order.end(), 0);
    std::partial_sort(predicted_order.begin(), predicted_order.begin() + top_k, predicted_order.end(),
        [&](auto lhs, auto rhs) { return predicted[lhs] < predicted[rhs]; });
    std::partial_sort(measured_order.begin(), measured_order.begin() + top_k, measured_order.end(),
        [&](auto lhs, auto rhs) { return measured[lhs] < measured[rhs]; });
    std::size_t overlap = 0;
    for (std::size_t index = 0; index < top_k; ++index) {
        if (std::find(measured_order.begin(), measured_order.begin() + top_k,
                      predicted_order[index]) != measured_order.begin() + top_k) ++overlap;
    }
    return top_k ? static_cast<double>(overlap) / static_cast<double>(top_k) : 1.0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string output = "results/depth_study.csv";
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg.starts_with("--output=")) output = arg.substr(9);
    }
    std::filesystem::create_directories(std::filesystem::path(output).parent_path());
    std::ofstream csv(output);
    csv << "size,packing,prefetch,unroll,time_ms,gflops,predicted_cycles,max_error\n";
    const auto target = schedforge::TargetInfo::detect();
    std::vector<double> predicted;
    std::vector<double> measured;
    for (int size : {32, 64, 128, 256, 512}) {
        const schedforge::GraphIR graph{{size, size, size, true, true}};
        const auto data = schedforge::make_data(graph.problem, static_cast<std::uint32_t>(size));
        for (int packing : {0, 1, 2}) for (int prefetch : {0, 4}) for (int unroll : {1, 4}) {
            if (prefetch > 0 && packing == 0) continue;
            schedforge::Schedule schedule;
            schedule.mc = 64; schedule.nc = 128; schedule.kc = 64;
            schedule.bm = 32; schedule.bn = 64; schedule.bk = 32;
            schedule.mr = 8; schedule.nr = 8; schedule.vector_width = 8;
            schedule.threads = std::min(8, target.logical_cpus);
            schedule.pack_b = packing >= 1; schedule.pack_a = packing == 2;
            schedule.prefetch_distance = prefetch; schedule.unroll_k = unroll;
            const schedforge::LoopIR loop{graph.problem, schedule};
            const auto simulation = schedforge::simulate(loop, target);
            const auto benchmark = schedforge::benchmark(loop, data, 1, size <= 64 ? 5 : 3);
            predicted.push_back(simulation.estimated_cycles);
            measured.push_back(benchmark.milliseconds);
            csv << size << ',' << packing << ',' << prefetch << ',' << unroll << ','
                << benchmark.milliseconds << ',' << benchmark.gflops << ','
                << simulation.estimated_cycles << ',' << benchmark.max_error << '\n';
        }
    }
    std::cout << "study_csv: " << output << '\n'
              << "schedule_spearman: " << spearman(predicted, measured) << '\n'
              << "top5_recall: " << top_k_recall(predicted, measured, 5) << '\n'
              << "top10_recall: " << top_k_recall(predicted, measured, 10) << '\n';
}
