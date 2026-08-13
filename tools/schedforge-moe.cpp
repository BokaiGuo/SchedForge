#include "schedforge/moe_compiler.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

schedforge::RoutingDistribution parse_distribution(const std::string& value) {
    if (value == "uniform") return schedforge::RoutingDistribution::Uniform;
    if (value == "moderate" || value == "moderate-skew") return schedforge::RoutingDistribution::ModerateSkew;
    if (value == "heavy" || value == "heavy-skew") return schedforge::RoutingDistribution::HeavySkew;
    throw std::invalid_argument("unknown routing distribution: " + value);
}

schedforge::MoeExecutionStrategy parse_strategy(const std::string& value) {
    if (value == "independent") return schedforge::MoeExecutionStrategy::IndependentExperts;
    if (value == "grouped") return schedforge::MoeExecutionStrategy::Grouped;
    if (value == "bucketed") return schedforge::MoeExecutionStrategy::BucketedGrouped;
    throw std::invalid_argument("unknown MoE strategy: " + value);
}

schedforge::MoeTaskScheduling parse_scheduler(const std::string& value) {
    if (value == "fixed") return schedforge::MoeTaskScheduling::FixedExperts;
    if (value == "steal") return schedforge::MoeTaskScheduling::WorkStealing;
    if (value == "split") return schedforge::MoeTaskScheduling::LoadAwareSplit;
    throw std::invalid_argument("unknown MoE scheduler: " + value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        schedforge::MoeConfig config;
        schedforge::MoeExecutionSchedule schedule;
        schedule.threads = 8;
        schedforge::RoutingDistribution distribution = schedforge::RoutingDistribution::Uniform;
        std::string output = "results/moe_mlp.sfe";
        bool router_data = false;
        bool auto_strategy = false;
        int repetitions = 5;
        std::string experiment_csv;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument.starts_with("--tokens=")) config.tokens = std::stoi(argument.substr(9));
            else if (argument.starts_with("--hidden=")) config.hidden = std::stoi(argument.substr(9));
            else if (argument.starts_with("--intermediate=")) config.intermediate = std::stoi(argument.substr(15));
            else if (argument.starts_with("--experts=")) config.experts = std::stoi(argument.substr(10));
            else if (argument.starts_with("--top-k=")) config.top_k = std::stoi(argument.substr(8));
            else if (argument.starts_with("--threads=")) schedule.threads = std::stoi(argument.substr(10));
            else if (argument.starts_with("--split-threshold=")) schedule.split_threshold = std::stoi(argument.substr(18));
            else if (argument.starts_with("--routing=")) distribution = parse_distribution(argument.substr(10));
            else if (argument == "--strategy=auto") auto_strategy = true;
            else if (argument.starts_with("--strategy=")) schedule.strategy = parse_strategy(argument.substr(11));
            else if (argument.starts_with("--scheduler=")) schedule.task_scheduling = parse_scheduler(argument.substr(12));
            else if (argument.starts_with("--repetitions=")) repetitions = std::stoi(argument.substr(14));
            else if (argument.starts_with("--experiment-csv=")) experiment_csv = argument.substr(17);
            else if (argument == "--router-data") router_data = true;
            else if (argument == "-o" && index + 1 < argc) output = argv[++index];
            else if (argument == "--help") {
                std::cout << "schedforge-moe [--tokens=128 --hidden=512 --intermediate=2048]\n"
                             "               [--experts=8 --top-k=2 --threads=8]\n"
                             "               [--routing=uniform|moderate|heavy|--router-data]\n"
                             "               [--strategy=auto|independent|grouped|bucketed]\n"
                             "               [--scheduler=fixed|steal|split]\n"
                             "               [--experiment-csv=results/moe.csv] [-o plan.sfe]\n";
                return 0;
            } else throw std::invalid_argument("unknown option: " + argument);
        }
        const auto data = schedforge::make_moe_data(config, 17);
        const auto routing = router_data ? schedforge::route_topk(config, data)
                                         : schedforge::make_routing_trace(config, distribution, 19);
        if (auto_strategy)
            schedule = schedforge::select_moe_schedule(
                config, routing, schedforge::TargetInfo::detect(), schedule.threads);
        const auto plan = schedforge::MoeCompiler{}.compile(config, schedule);
        plan.save(output);
        const auto simulation = schedforge::simulate_moe(config, routing, plan.schedule);
        const auto measured = router_data
            ? schedforge::execute_moe(plan, data, 1, repetitions)
            : schedforge::execute_moe(plan, data, routing, 1, repetitions);
        if (!experiment_csv.empty())
            schedforge::write_moe_experiment_csv(experiment_csv, config, data,
                                                 schedule.threads, repetitions);
        std::cout << "[MoE Graph]\noperations: " << plan.tensor_graph.operations().size()
                  << "\nprogram operations: " << plan.program.operations.size()
                  << "\nsegmented type: " << plan.segmented_type.str() << "\n\n"
                  << "[Routing]\nmode: " << (router_data ? "router-data" : schedforge::routing_distribution_name(distribution))
                  << "\n" << routing.dump() << "\n\n"
                  << "[Execution Strategy]\n" << plan.schedule.dump() << '\n'
                  << "[Memory]\n" << plan.memory.dump() << "\n\n"
                  << "[Simulation]\n" << simulation.dump() << "\n\n"
                  << std::fixed << std::setprecision(3)
                  << "[Runtime]\nP50 execution: " << measured.p50_milliseconds << " ms\n"
                  << "P95 execution: " << measured.p95_milliseconds << " ms\n"
                  << "dispatch: " << measured.dispatch_milliseconds << " ms\n"
                  << "experts: " << measured.expert_milliseconds << " ms\n"
                  << "combine: " << measured.combine_milliseconds << " ms\n"
                  << "worker imbalance: " << measured.worker_imbalance << "\n\n"
                  << "[Validation]\nmax error: " << measured.max_error << "\n\n"
                  << "[Hardware]\n" << plan.hardware << '\n';
        return measured.max_error <= 1.0e-3 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
