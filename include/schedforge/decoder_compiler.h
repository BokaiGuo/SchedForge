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
    std::string hardware;
    std::string dump() const;
    void save(const std::filesystem::path& path) const;
};

struct DecoderBenchmarkResult {
    double milliseconds = 0.0;
    double attention_milliseconds = 0.0;
    double ffn_milliseconds = 0.0;
    double max_error = 0.0;
    std::vector<float> output;
};

struct DecoderCompileOptions {
    int max_threads = 8;
    bool tune_attention = false;
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

}  // namespace schedforge
