#pragma once

#include "schedforge/graph_compiler.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace schedforge {

enum class AttentionLoweringStrategy {
    Materialized,
    TiledMaterialized,
    IOAware,
    AutoScheduledIOAware,
    SplitKVDecode
};

enum class AttentionParallelAxis { BatchHead, QueryBlock, HeadQueryBlock, SplitKV };

struct AttentionConfig {
    int batch = 1;
    int query_heads = 8;
    int kv_heads = 8;
    int sequence_query = 128;
    int sequence_kv = 128;
    int head_dim = 64;
    int head_dim_value = 64;
    bool causal = false;
    float scale = 0.0F;
};

struct AttentionData {
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
};

struct AttentionSchedule {
    int query_tile = 32;
    int kv_tile = 64;
    int qk_micro_rows = 4;
    int qk_micro_columns = 8;
    int pv_micro_rows = 4;
    int pv_micro_columns = 8;
    int softmax_vector_width = 8;
    bool pack_k = true;
    bool pack_v = false;
    int prefetch_k = 2;
    int prefetch_v = 2;
    int threads = 1;
    AttentionParallelAxis parallel_axis = AttentionParallelAxis::HeadQueryBlock;
    std::string dump() const;
};

struct AttentionPlanOptions {
    AttentionLoweringStrategy strategy = AttentionLoweringStrategy::IOAware;
    AttentionSchedule schedule;
    int split_kv = 1;
};

struct TilePipelineOperation {
    std::string name;
    std::string inputs;
    std::string result;
    std::string dump() const;
};

struct TilePipelineIR {
    std::string name = "attention";
    std::vector<TilePipelineOperation> operations;
    std::string dump() const;
};

struct AttentionMemoryPlan {
    std::size_t temporary_bytes = 0;
    std::size_t score_bytes = 0;
    std::size_t probability_bytes = 0;
    std::size_t online_state_bytes = 0;
    std::string dump() const;
};

struct AttentionSimulationResult {
    double flops = 0.0;
    double bytes_read = 0.0;
    double bytes_written = 0.0;
    double arithmetic_intensity = 0.0;
    std::size_t temporary_bytes = 0;
    std::size_t qk_tiles = 0;
    std::size_t skipped_causal_tiles = 0;
    double score_evaluations = 0.0;
    double exp_evaluations = 0.0;
    double reduction_operations = 0.0;
    double estimated_l2_traffic = 0.0;
    double estimated_llc_traffic = 0.0;
    double estimated_l2_hit_rate = 0.0;
    double estimated_llc_hit_rate = 0.0;
    std::string dump() const;
};

struct AttentionExecutablePlan {
    AttentionConfig config;
    AttentionPlanOptions plan;
    TensorGraph tensor_graph;
    TilePipelineIR pipeline;
    AttentionMemoryPlan memory;
    AttentionSimulationResult simulation;
    LoopIR qk_loop;
    LoopIR pv_loop;
    std::string llvm_qk;
    std::string llvm_pv;
    std::string hardware;
    std::vector<std::string> guards;
    std::string dump() const;
    void save(const std::filesystem::path& path) const;
};

struct AttentionBenchmarkResult {
    double milliseconds = 0.0;
    double p50_milliseconds = 0.0;
    double p95_milliseconds = 0.0;
    double max_error = 0.0;
    std::vector<float> output;
};

struct KVCache {
    int batch = 1;
    int kv_heads = 1;
    int head_dim = 1;
    int head_dim_value = 1;
    int capacity = 0;
    int length = 0;
    std::vector<float> keys;
    std::vector<float> values;
    std::string dump() const;
};

class AttentionFusionPass {
public:
    TensorGraph run(const TensorGraph& graph) const;
};

class AttentionCompiler {
public:
    explicit AttentionCompiler(TargetInfo target = TargetInfo::detect());
    AttentionExecutablePlan compile(const AttentionConfig& config,
                                    AttentionPlanOptions options = {}) const;
private:
    TargetInfo target_;
};

TensorGraph build_sdpa_graph(const AttentionConfig& config, bool dynamic_sequence = true);
AttentionData make_attention_data(const AttentionConfig& config, std::uint32_t seed = 7);
AttentionPlanOptions select_attention_plan(const AttentionConfig& config,
                                           const TargetInfo& target,
                                           int max_threads);
AttentionPlanOptions tune_attention_plan(const AttentionConfig& config,
                                         const AttentionData& data,
                                         const TargetInfo& target,
                                         int max_threads,
                                         int warmup = 1,
                                         int repetitions = 2);
AttentionSimulationResult simulate_attention(const AttentionConfig& config,
                                              const AttentionPlanOptions& plan,
                                              const TargetInfo& target);
std::vector<float> reference_attention(const AttentionConfig& config,
                                       const AttentionData& data);
AttentionBenchmarkResult execute_attention(const AttentionExecutablePlan& plan,
                                           const AttentionData& data,
                                           int warmup = 1, int repetitions = 5);
AttentionBenchmarkResult execute_attention(const AttentionExecutablePlan& plan,
                                           const AttentionData& data,
                                           int runtime_sequence_query,
                                           int runtime_sequence_kv,
                                           int warmup, int repetitions);
KVCache make_kv_cache(const AttentionConfig& config, int capacity = 0);
void append_kv(KVCache& cache, const std::vector<float>& keys,
               const std::vector<float>& values, int tokens);
AttentionBenchmarkResult execute_decode_attention(const AttentionExecutablePlan& plan,
                                                  const std::vector<float>& query,
                                                  const KVCache& cache,
                                                  int warmup = 1, int repetitions = 5);
std::string attention_strategy_name(AttentionLoweringStrategy strategy);
void write_attention_experiment_csv(const std::filesystem::path& path,
                                    const AttentionConfig& base,
                                    int threads, int repetitions);
void write_attention_scaling_csv(const std::filesystem::path& path,
                                 const AttentionConfig& base,
                                 int threads);

}  // namespace schedforge
