#pragma once

#include "schedforge/attention_compiler.h"
#include "schedforge/moe_compiler.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace schedforge {

enum class DecoderFFNKind { Dense, MoE };

struct DecoderConfig {
    int batch = 1;
    int sequence = 16;
    int hidden = 64;
    int intermediate = 128;
    int query_heads = 4;
    int kv_heads = 2;
    int head_dim = 16;
    float rms_epsilon = 1.0e-5F;
    float rope_theta = 10000.0F;
    bool causal = true;
    DecoderFFNKind ffn = DecoderFFNKind::Dense;
    int experts = 4;
    int top_k = 2;
    int context_sequence = 0;
};

struct DecoderData {
    std::vector<float> input;
    std::vector<float> rms1_weight;
    std::vector<float> query_weight;
    std::vector<float> key_weight;
    std::vector<float> value_weight;
    std::vector<float> output_weight;
    std::vector<float> rms2_weight;
    std::vector<float> gate_weight;
    std::vector<float> up_weight;
    std::vector<float> down_weight;
    MoeData moe;
    std::vector<float> cached_key;
    std::vector<float> cached_value;
    std::optional<RoutingTrace> routing_trace;
};

enum class DecoderIntermediateLayout { HeadMajor, TokenMajor };
enum class DecoderScheduleFamily { Latency, Throughput };
enum class DecoderCorePlacement { Compact, Spread };

struct DecoderPlanPolicy {
    DecoderIntermediateLayout attention_layout = DecoderIntermediateLayout::HeadMajor;
    DecoderScheduleFamily schedule_family = DecoderScheduleFamily::Latency;
    DecoderCorePlacement core_placement = DecoderCorePlacement::Compact;
    bool materialize_attention_output = true;
    bool reuse_workspace = true;
    int threads = 0;
    AttentionLoweringStrategy attention_strategy = AttentionLoweringStrategy::IOAware;
    std::string dump() const;
    bool operator==(const DecoderPlanPolicy&) const = default;
};

struct DecoderConstant {
    std::string name;
    std::vector<int> logical_shape;
    std::vector<int> packed_shape;
    std::string layout;
    std::size_t bytes = 0;
    std::string dump() const;
};

struct DecoderMemoryPlan {
    std::size_t naive_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::string dump() const;
};

struct DecoderFusionSummary {
    bool fused_qkv = false;
    bool fused_gate_up = false;
    bool fused_rope = false;
    std::string dump() const;
};

class DecoderFusionPass {
public:
    DecoderFusionSummary run(const TensorGraph& graph, DecoderFFNKind ffn) const;
};

struct DecoderExecutablePlan {
    DecoderConfig config;
    TensorGraph imported_graph;
    TensorGraph graph;
    DecoderFusionSummary fusion;
    std::vector<DecoderConstant> constants;
    std::vector<float> packed_qkv_weight;
    std::vector<float> packed_gate_up_weight;
    DecoderMemoryPlan memory;
    LoopIR qkv_loop;
    LoopIR output_loop;
    LoopIR gate_up_loop;
    LoopIR down_loop;
    AttentionExecutablePlan attention;
    std::optional<MoeExecutablePlan> moe;
    std::vector<std::string> llvm_ir;
    DecoderPlanPolicy policy;
    double compile_milliseconds = 0.0;
    double llvm_compile_milliseconds = 0.0;
    double memory_planning_milliseconds = 0.0;
    std::string hardware;
    std::string dump() const;
    void save(const std::filesystem::path& path) const;
};

struct DecoderBenchmarkResult {
    double milliseconds = 0.0;
    double attention_milliseconds = 0.0;
    double projection_milliseconds = 0.0;
    double norm_rope_milliseconds = 0.0;
    double ffn_milliseconds = 0.0;
    double residual_milliseconds = 0.0;
    double dispatch_overhead_milliseconds = 0.0;
    double tokens_per_second = 0.0;
    double max_error = 0.0;
    std::vector<float> output;
};

struct DecoderCompileOptions {
    int max_threads = 8;
    bool tune_attention = false;
    bool emit_llvm = true;
    bool specialize_constants = true;
    std::optional<DecoderPlanPolicy> policy;
};

struct DecoderPlanCost {
    double measured_milliseconds = 0.0;
    double analytical_score = 0.0;
    std::size_t workspace_bytes = 0;
    std::size_t materialization_bytes = 0;
    bool measured = false;
    std::string dump() const;
};

struct DecoderPlanCandidate {
    DecoderPlanPolicy policy;
    DecoderPlanCost cost;
};

struct DecoderOptimizationResult {
    DecoderExecutablePlan plan;
    DecoderPlanCost baseline;
    DecoderPlanCost winner;
    std::vector<DecoderPlanCandidate> candidates;
    std::size_t hardware_measurements = 0;
    double speedup = 1.0;
    std::string dump() const;
};

class ExecutablePlanOptimizer {
public:
    explicit ExecutablePlanOptimizer(TargetInfo target = TargetInfo::detect());
    DecoderPlanCost evaluate(const DecoderExecutablePlan& plan) const;
    DecoderOptimizationResult optimize(const TensorGraph& graph,
                                       const DecoderConfig& config,
                                       const DecoderData& data,
                                       int max_threads,
                                       int measurement_budget = 8,
                                       int repetitions = 2) const;
private:
    TargetInfo target_;
};

enum class DecoderEvidenceKind { Measured, CompileOnly, Skipped };

struct DecoderBenchmarkProfile {
    std::string name;
    DecoderConfig config;
    RoutingDistribution routing = RoutingDistribution::Uniform;
    std::uint64_t estimated_flops = 0;
    std::size_t estimated_weight_bytes = 0;
    std::size_t estimated_activation_bytes = 0;
};

struct DecoderBenchmarkRecord {
    DecoderBenchmarkProfile profile;
    DecoderEvidenceKind evidence = DecoderEvidenceKind::Skipped;
    DecoderBenchmarkResult measured;
    DecoderPlanCost plan_cost;
    double compile_milliseconds = 0.0;
    double llvm_compile_milliseconds = 0.0;
    double memory_planning_milliseconds = 0.0;
    double optimizer_speedup = 1.0;
    double optimizer_baseline_milliseconds = 0.0;
    std::size_t hardware_measurements = 0;
    std::string selected_policy;
    std::vector<DecoderPlanCandidate> optimizer_candidates;
    std::string note;
};

class DecoderCompiler {
public:
    explicit DecoderCompiler(TargetInfo target = TargetInfo::detect());
    DecoderExecutablePlan compile(const TensorGraph& imported_graph,
                                  const DecoderConfig& config,
                                  const DecoderData& data,
                                  DecoderCompileOptions options = {}) const;
private:
    TargetInfo target_;
};

TensorGraph build_decoder_layer_graph(const DecoderConfig& config);
DecoderData make_decoder_data(const DecoderConfig& config, std::uint32_t seed = 7);
std::vector<float> reference_decoder_layer(const DecoderConfig& config,
                                           const DecoderData& data);
DecoderBenchmarkResult execute_decoder_layer(const DecoderExecutablePlan& plan,
                                             const DecoderData& data,
                                             int warmup = 1,
                                             int repetitions = 5);
std::string decoder_ffn_name(DecoderFFNKind kind);
std::string decoder_evidence_name(DecoderEvidenceKind kind);
std::vector<DecoderBenchmarkProfile> realistic_decoder_profiles();
DecoderBenchmarkRecord benchmark_decoder_profile(const DecoderBenchmarkProfile& profile,
                                                  const TensorGraph& graph,
                                                  int max_threads,
                                                  std::uint64_t max_real_flops,
                                                  std::size_t max_real_weight_bytes,
                                                  bool optimize_plan,
                                                  int repetitions = 2);
void write_decoder_benchmark_csv(const std::filesystem::path& path,
                                 const std::vector<DecoderBenchmarkRecord>& records);
void write_decoder_plan_candidates_csv(
    const std::filesystem::path& path,
    const std::vector<DecoderBenchmarkRecord>& records);

}  // namespace schedforge
