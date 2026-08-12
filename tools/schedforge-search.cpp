#include "schedforge/compiler.h"

#include <iostream>

int main() {
    const schedforge::GraphIR graph{{128, 128, 128, true, true}};
    const auto data = schedforge::make_data(graph.problem, 31);
    for (const auto strategy : {schedforge::SearchStrategy::Grid, schedforge::SearchStrategy::Random,
                                schedforge::SearchStrategy::Greedy, schedforge::SearchStrategy::Evolutionary}) {
        const auto result = schedforge::compare_search_strategy(graph, data, strategy, 4, 12, 31);
        std::cout << schedforge::search_strategy_name(strategy)
                  << ",candidates=" << result.candidates_considered
                  << ",hardware=" << result.hardware_measurements
                  << ",best_gflops=" << result.best_gflops
                  << ",elapsed_ms=" << result.elapsed_milliseconds << '\n';
    }
}
