#pragma once

#include "schedforge/graph_compiler.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace schedforge {

enum class MoeOpKind {
    RouterMatMul,
    Softmax,
    TopK,
    Histogram,
    PrefixSum,
    Dispatch,
    GroupedMatMulW1,
    GroupedMatMulW3,
    SwiGLU,
    GroupedMatMulW2,
    WeightedCombine
};

enum class MoeExecutionStrategy { IndependentExperts, Grouped, BucketedGrouped };
enum class MoeTaskScheduling { FixedExperts, WorkStealing, LoadAwareSplit };
enum class RoutingDistribution { Uniform, ModerateSkew, HeavySkew };

struct MoeOperation {
    MoeOpKind kind = MoeOpKind::RouterMatMul;
    std::string result;
    std::vector<std::string> inputs;
    std::string attributes;
    std::string str() const;
};

struct MoeProgram {
    std::vector<MoeOperation> operations;
    std::string dump() const;
};

struct SegmentedTensorType {
    int segments = 0;
    int feature_size = 0;
    DataType dtype = DataType::F32;
    std::string str() const;
};

struct RoutingTrace {
    int tokens = 0;
    int experts = 0;
    int top_k = 0;
    std::vector<int> expert_ids;
    std::vector<float> expert_weights;
    std::vector<int> counts;
    std::string dump() const;
};

struct SegmentedTensor {
    SegmentedTensorType type;
    std::vector<float> values;
    std::vector<int> offsets;
    std::vector<int> token_ids;
    std::vector<int> route_slots;
    std::vector<float> routing_weights;
    std::string dump() const;
};

struct MoeConfig {
    int tokens = 128;
    int hidden = 512;
    int intermediate = 2048;
    int experts = 8;
    int top_k = 2;
};

struct MoeData {
    std::vector<float> input;
    std::vector<float> router_weight;
    std::vector<float> w1;
    std::vector<float> w3;
    std::vector<float> w2;
};

struct MoeTask {
    int expert = 0;
    int begin = 0;
    int end = 0;
    int bucket = 0;
    int preferred_worker = 0;
};

struct MoeKernelVariant {
    int max_tokens = 0;
    LoopIR w1_loop;
    LoopIR w3_loop;
    LoopIR w2_loop;
    std::string llvm_w1;
    std::string llvm_w2;
};

struct MoeMemoryPlan {
    std::size_t naive_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t routing_bytes = 0;
    std::size_t segmented_bytes = 0;
    std::string dump() const;
};

struct MoeExecutionSchedule {
    MoeExecutionStrategy strategy = MoeExecutionStrategy::BucketedGrouped;
    MoeTaskScheduling task_scheduling = MoeTaskScheduling::LoadAwareSplit;
    std::vector<int> token_buckets = {4, 16, 64};
    int split_threshold = 32;
    int threads = 1;
    bool fuse_router_topk = true;
    bool fuse_histogram_prefix = true;
    bool fuse_combine_weight = true;
    std::string dump() const;
};

struct MoeExecutablePlan {
    MoeConfig config;
    TensorGraph tensor_graph;
    MoeProgram program;
    SegmentedTensorType segmented_type;
    MoeExecutionSchedule schedule;
    std::vector<MoeKernelVariant> kernels;
    MoeMemoryPlan memory;
    std::string target;
    std::string hardware;
    std::vector<std::string> guards;
    double llvm_compile_milliseconds = 0.0;
    std::string dump() const;
    void save(const std::filesystem::path& path) const;
};

struct MoeSimulationResult {
    std::vector<int> expert_counts;
    std::size_t tasks = 0;
    double imbalance = 0.0;
    double estimated_makespan = 0.0;
    double dispatch_bytes = 0.0;
    double weight_bytes = 0.0;
    double worker_utilization = 0.0;
    std::string dump() const;
};

struct MoeBenchmarkResult {
    double milliseconds = 0.0;
    double p50_milliseconds = 0.0;
    double p95_milliseconds = 0.0;
    double router_milliseconds = 0.0;
    double dispatch_milliseconds = 0.0;
    double expert_milliseconds = 0.0;
    double combine_milliseconds = 0.0;
    double max_error = 0.0;
    double worker_imbalance = 0.0;
    std::vector<int> expert_counts;
    std::vector<double> worker_milliseconds;
    std::vector<float> output;
};

TensorGraph build_moe_mlp_graph(const MoeConfig& config, bool dynamic_tokens = true);
MoeProgram lower_moe_program(const MoeConfig& config);
MoeData make_moe_data(const MoeConfig& config, std::uint32_t seed = 7);
RoutingTrace route_topk(const MoeConfig& config, const MoeData& data);
RoutingTrace route_topk(const MoeConfig& config, const MoeData& data,
                        const std::vector<float>& input);
RoutingTrace make_routing_trace(const MoeConfig& config, RoutingDistribution distribution,
                                std::uint32_t seed = 7);
SegmentedTensor dispatch_tokens(const MoeConfig& config, const MoeData& data,
                                const RoutingTrace& routing);
SegmentedTensor dispatch_tokens(const MoeConfig& config, const std::vector<float>& input,
                                const RoutingTrace& routing);
std::vector<MoeTask> plan_moe_tasks(const RoutingTrace& routing,
                                    const MoeExecutionSchedule& schedule);
MoeSimulationResult simulate_moe(const MoeConfig& config, const RoutingTrace& routing,
                                 const MoeExecutionSchedule& schedule);
MoeExecutionSchedule select_moe_schedule(const MoeConfig& config,
                                         const RoutingTrace& routing,
                                         const TargetInfo& target,
                                         int max_threads);
std::vector<float> reference_moe(const MoeConfig& config, const MoeData& data,
                                 const RoutingTrace& routing);
std::vector<float> reference_moe(const MoeConfig& config, const MoeData& data,
                                 const std::vector<float>& input,
                                 const RoutingTrace& routing);
MoeBenchmarkResult execute_moe(const MoeExecutablePlan& plan, const MoeData& data,
                               const RoutingTrace& routing, int warmup = 1,
                               int repetitions = 5, bool validate_result = true);
MoeBenchmarkResult execute_moe(const MoeExecutablePlan& plan, const MoeData& data,
                               const std::vector<float>& input,
                               const RoutingTrace& routing, int warmup = 1,
                               int repetitions = 5, bool validate_result = true);
MoeBenchmarkResult execute_moe(const MoeExecutablePlan& plan, const MoeData& data,
                               int warmup = 1, int repetitions = 5,
                               bool validate_result = true);
MoeBenchmarkResult execute_moe(const MoeExecutablePlan& plan, const MoeData& data,
                               const std::vector<float>& input, int warmup = 1,
                               int repetitions = 5, bool validate_result = true);

class MoeCompiler {
public:
    explicit MoeCompiler(TargetInfo target = TargetInfo::detect());
    MoeExecutablePlan compile(const MoeConfig& config,
                              MoeExecutionSchedule schedule = {}) const;
private:
    TargetInfo target_;
};

std::string moe_execution_strategy_name(MoeExecutionStrategy strategy);
std::string moe_task_scheduling_name(MoeTaskScheduling scheduling);
std::string routing_distribution_name(RoutingDistribution distribution);
void write_moe_experiment_csv(const std::filesystem::path& path,
                              const MoeConfig& config, const MoeData& data,
                              int threads, int repetitions);

}  // namespace schedforge
