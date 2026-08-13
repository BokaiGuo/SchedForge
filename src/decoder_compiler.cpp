#include "schedforge/decoder_compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace schedforge {
namespace {

int rows(const DecoderConfig& config) { return config.batch * config.sequence; }
int kv_width(const DecoderConfig& config) { return config.kv_heads * config.head_dim; }
int qkv_width(const DecoderConfig& config) { return config.hidden + 2 * kv_width(config); }

GraphTensorType tensor_type(std::vector<Dimension> shape) {
    return {std::move(shape), DataType::F32, {}, std::nullopt, 0, -1};
}

void validate(const DecoderConfig& config) {
    if (config.batch <= 0 || config.sequence <= 0 || config.hidden <= 0 ||
        config.intermediate <= 0 || config.query_heads <= 0 || config.kv_heads <= 0 ||
        config.head_dim <= 0 || config.hidden != config.query_heads * config.head_dim ||
        config.query_heads % config.kv_heads != 0 || config.rms_epsilon <= 0.0F) {
        throw std::invalid_argument("invalid decoder configuration");
    }
    if (config.ffn == DecoderFFNKind::MoE &&
        (config.experts <= 0 || config.top_k <= 0 || config.top_k > config.experts)) {
        throw std::invalid_argument("invalid decoder MoE configuration");
    }
    if (config.context_sequence < 0)
        throw std::invalid_argument("invalid decoder context sequence");
}

std::vector<float> rms_norm(const std::vector<float>& input,
                            const std::vector<float>& weight,
                            int row_count, int width, float epsilon) {
    if (input.size() != static_cast<std::size_t>(row_count * width) ||
        weight.size() != static_cast<std::size_t>(width))
        throw std::invalid_argument("RMSNorm shape mismatch");
    std::vector<float> output(input.size());
    for (int row = 0; row < row_count; ++row) {
        double square_sum = 0.0;
        for (int column = 0; column < width; ++column) {
            const float value = input[static_cast<std::size_t>(row) * width + column];
            square_sum += static_cast<double>(value) * value;
        }
        const float inverse = 1.0F / std::sqrt(
            static_cast<float>(square_sum / static_cast<double>(width)) + epsilon);
        for (int column = 0; column < width; ++column)
            output[static_cast<std::size_t>(row) * width + column] =
                input[static_cast<std::size_t>(row) * width + column] * inverse *
                weight[static_cast<std::size_t>(column)];
    }
    return output;
}

std::vector<float> scalar_matmul(const std::vector<float>& lhs,
                                 const std::vector<float>& rhs,
                                 int m, int n, int k) {
    std::vector<float> output(static_cast<std::size_t>(m * n), 0.0F);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int reduction = 0; reduction < k; ++reduction)
                output[static_cast<std::size_t>(i) * n + j] +=
                    lhs[static_cast<std::size_t>(i) * k + reduction] *
                    rhs[static_cast<std::size_t>(reduction) * n + j];
    return output;
}

std::vector<float> concatenate_columns(const std::vector<std::vector<float>>& weights,
                                       int input_width,
                                       const std::vector<int>& output_widths) {
    const int total_width = std::accumulate(output_widths.begin(), output_widths.end(), 0);
    std::vector<float> packed(static_cast<std::size_t>(input_width * total_width));
    for (int row = 0; row < input_width; ++row) {
        int destination_column = 0;
        for (std::size_t tensor = 0; tensor < weights.size(); ++tensor) {
            const int width = output_widths[tensor];
            if (weights[tensor].size() != static_cast<std::size_t>(input_width * width))
                throw std::invalid_argument("constant weight shape mismatch");
            std::copy_n(weights[tensor].begin() + static_cast<std::ptrdiff_t>(row * width), width,
                        packed.begin() + static_cast<std::ptrdiff_t>(row * total_width + destination_column));
            destination_column += width;
        }
    }
    return packed;
}

void apply_rope(std::vector<float>& tensor, int batch, int heads, int sequence,
                int head_dim, float theta) {
    for (int batch_index = 0; batch_index < batch; ++batch_index)
        for (int head = 0; head < heads; ++head)
            for (int token = 0; token < sequence; ++token)
                for (int dimension = 0; dimension + 1 < head_dim; dimension += 2) {
                    const float frequency = std::pow(theta,
                        -static_cast<float>(dimension) / static_cast<float>(head_dim));
                    const float angle = static_cast<float>(token) * frequency;
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);
                    const std::size_t base = (((static_cast<std::size_t>(batch_index) * heads + head) *
                                                sequence + token) * head_dim + dimension);
                    const float even = tensor[base];
                    const float odd = tensor[base + 1];
                    tensor[base] = even * cosine - odd * sine;
                    tensor[base + 1] = even * sine + odd * cosine;
                }
}

AttentionData split_qkv(const DecoderConfig& config, const std::vector<float>& fused) {
    AttentionData attention;
    attention.query.resize(static_cast<std::size_t>(config.batch * config.query_heads *
                                                    config.sequence * config.head_dim));
    attention.key.resize(static_cast<std::size_t>(config.batch * config.kv_heads *
                                                  config.sequence * config.head_dim));
    attention.value.resize(attention.key.size());
    const int kv = kv_width(config);
    const int total = qkv_width(config);
    for (int batch = 0; batch < config.batch; ++batch)
        for (int token = 0; token < config.sequence; ++token) {
            const int row = batch * config.sequence + token;
            for (int head = 0; head < config.query_heads; ++head)
                for (int dimension = 0; dimension < config.head_dim; ++dimension)
                    attention.query[(((static_cast<std::size_t>(batch) * config.query_heads + head) *
                                      config.sequence + token) * config.head_dim + dimension)] =
                        fused[static_cast<std::size_t>(row) * total + head * config.head_dim + dimension];
            for (int head = 0; head < config.kv_heads; ++head)
                for (int dimension = 0; dimension < config.head_dim; ++dimension) {
                    const std::size_t destination = (((static_cast<std::size_t>(batch) * config.kv_heads + head) *
                                                       config.sequence + token) * config.head_dim + dimension);
                    attention.key[destination] = fused[static_cast<std::size_t>(row) * total +
                        config.hidden + head * config.head_dim + dimension];
                    attention.value[destination] = fused[static_cast<std::size_t>(row) * total +
                        config.hidden + kv + head * config.head_dim + dimension];
                }
        }
    apply_rope(attention.query, config.batch, config.query_heads, config.sequence,
               config.head_dim, config.rope_theta);
    apply_rope(attention.key, config.batch, config.kv_heads, config.sequence,
               config.head_dim, config.rope_theta);
    return attention;
}

std::vector<float> merge_attention(const DecoderConfig& config,
                                   const std::vector<float>& attention,
                                   DecoderIntermediateLayout layout = DecoderIntermediateLayout::HeadMajor) {
    std::vector<float> merged(static_cast<std::size_t>(rows(config) * config.hidden));
    if (layout == DecoderIntermediateLayout::TokenMajor) {
        for (int batch = 0; batch < config.batch; ++batch)
            for (int token = 0; token < config.sequence; ++token)
                for (int head = 0; head < config.query_heads; ++head)
                    for (int dimension = 0; dimension < config.head_dim; ++dimension)
                        merged[static_cast<std::size_t>(batch * config.sequence + token) * config.hidden +
                               head * config.head_dim + dimension] =
                            attention[(((static_cast<std::size_t>(batch) * config.query_heads + head) *
                                        config.sequence + token) * config.head_dim + dimension)];
    } else {
        for (int batch = 0; batch < config.batch; ++batch)
            for (int head = 0; head < config.query_heads; ++head)
                for (int token = 0; token < config.sequence; ++token)
                    for (int dimension = 0; dimension < config.head_dim; ++dimension)
                        merged[static_cast<std::size_t>(batch * config.sequence + token) * config.hidden +
                               head * config.head_dim + dimension] =
                            attention[(((static_cast<std::size_t>(batch) * config.query_heads + head) *
                                        config.sequence + token) * config.head_dim + dimension)];
    }
    return merged;
}

std::vector<float> add(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("residual shape mismatch");
    std::vector<float> output(lhs.size());
    for (std::size_t index = 0; index < lhs.size(); ++index) output[index] = lhs[index] + rhs[index];
    return output;
}

std::vector<float> swiglu(const std::vector<float>& gate_up, int row_count, int intermediate) {
    std::vector<float> output(static_cast<std::size_t>(row_count * intermediate));
    for (int row = 0; row < row_count; ++row)
        for (int column = 0; column < intermediate; ++column) {
            const float gate = gate_up[static_cast<std::size_t>(row) * 2 * intermediate + column];
            const float up = gate_up[static_cast<std::size_t>(row) * 2 * intermediate +
                                     intermediate + column];
            output[static_cast<std::size_t>(row) * intermediate + column] =
                gate / (1.0F + std::exp(-gate)) * up;
        }
    return output;
}

Schedule decoder_schedule(const TargetInfo& target, const DecoderPlanPolicy& policy) {
    Schedule schedule;
    schedule.threads = std::clamp(policy.threads, 1, target.logical_cpus);
    schedule.pin_threads = policy.core_placement == DecoderCorePlacement::Compact && schedule.threads > 1;
    schedule.bm = policy.schedule_family == DecoderScheduleFamily::Latency ? 8 : 32;
    schedule.bn = policy.schedule_family == DecoderScheduleFamily::Latency ? 32 : 128;
    schedule.bk = policy.schedule_family == DecoderScheduleFamily::Latency ? 32 : 64;
    schedule.mr = 4;
    schedule.nr = std::max(1, target.vector_width);
    schedule.vector_width = std::max(1, target.vector_width);
    schedule.fused = true;
    return schedule;
}

LLVMJITResult compile_llvm(const LoopIR& loop) {
    TensorData data;
    data.a.assign(static_cast<std::size_t>(loop.problem.m * loop.problem.k), 0.0F);
    data.b.assign(static_cast<std::size_t>(loop.problem.k * loop.problem.n), 0.0F);
    if (loop.problem.bias) data.bias.assign(static_cast<std::size_t>(loop.problem.n), 0.0F);
    return LLVMJITBackend{}.benchmark(loop, data, 0, 1);
}

DecoderBenchmarkResult execute_once(const DecoderExecutablePlan& plan,
                                    const DecoderData& data) {
    const auto total_start = std::chrono::steady_clock::now();
    const auto& config = plan.config;
    const int row_count = rows(config);
    const auto norm1_start = std::chrono::steady_clock::now();
    const auto normalized1 = rms_norm(data.input, data.rms1_weight, row_count,
                                      config.hidden, config.rms_epsilon);
    double norm_rope_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - norm1_start).count();
    const auto qkv_start = std::chrono::steady_clock::now();
    std::vector<float> qkv;
    execute(plan.qkv_loop, {normalized1, plan.packed_qkv_weight, {}, {}, {}}, qkv);
    double projection_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - qkv_start).count();
    const auto rope_start = std::chrono::steady_clock::now();
    auto attention_data = split_qkv(config, qkv);
    norm_rope_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - rope_start).count();
    AttentionBenchmarkResult attention;
    if (config.context_sequence > 0) {
        KVCache cache = make_kv_cache(plan.attention.config, config.context_sequence);
        append_kv(cache, data.cached_key, data.cached_value, config.context_sequence);
        attention = execute_decode_attention(plan.attention, attention_data.query, cache, 0, 1, false);
    } else {
        attention = execute_attention(plan.attention, attention_data, 0, 1, false);
    }
    const auto output_projection_start = std::chrono::steady_clock::now();
    auto merged = merge_attention(config, attention.output, plan.policy.attention_layout);
    if (plan.policy.materialize_attention_output) {
        std::vector<float> materialized = merged;
        merged.swap(materialized);
    }
    std::vector<float> projected;
    execute(plan.output_loop, {merged, data.output_weight, {}, {}, {}}, projected);
    projection_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - output_projection_start).count();
    const auto residual1_start = std::chrono::steady_clock::now();
    const auto attention_residual = add(projected, data.input);
    double residual_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - residual1_start).count();
    const auto norm2_start = std::chrono::steady_clock::now();
    const auto normalized2 = rms_norm(attention_residual, data.rms2_weight, row_count,
                                      config.hidden, config.rms_epsilon);
    norm_rope_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - norm2_start).count();

    DecoderBenchmarkResult result;
    result.attention_milliseconds = attention.p50_milliseconds;
    result.projection_milliseconds = projection_ms;
    result.norm_rope_milliseconds = norm_rope_ms;
    if (config.ffn == DecoderFFNKind::Dense) {
        const auto ffn_start = std::chrono::steady_clock::now();
        std::vector<float> gate_up;
        execute(plan.gate_up_loop,
                {normalized2, plan.packed_gate_up_weight, {}, {}, {}}, gate_up);
        const auto activated = swiglu(gate_up, row_count, config.intermediate);
        std::vector<float> down;
        execute(plan.down_loop, {activated, data.down_weight, {}, {}, {}}, down);
        result.output = add(attention_residual, down);
        result.ffn_milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ffn_start).count();
    } else {
        const auto moe_result = data.routing_trace
            ? execute_moe(*plan.moe, data.moe, normalized2, *data.routing_trace,
                          0, 1, false)
            : execute_moe(*plan.moe, data.moe, normalized2, 0, 1, false);
        result.output = add(attention_residual, moe_result.output);
        result.ffn_milliseconds = moe_result.p50_milliseconds;
    }
    const auto residual2_start = std::chrono::steady_clock::now();
    if (!result.output.empty()) {
        volatile float sink = result.output.front();
        (void)sink;
    }
    residual_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - residual2_start).count();
    result.residual_milliseconds = residual_ms;
    result.milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_start).count();
    const double accounted = result.attention_milliseconds + result.projection_milliseconds +
        result.norm_rope_milliseconds + result.ffn_milliseconds + result.residual_milliseconds;
    result.dispatch_overhead_milliseconds = std::max(0.0, result.milliseconds - accounted);
    result.tokens_per_second = result.milliseconds > 0.0
        ? static_cast<double>(rows(config)) * 1000.0 / result.milliseconds : 0.0;
    return result;
}

}  // namespace

std::string decoder_ffn_name(DecoderFFNKind kind) {
    return kind == DecoderFFNKind::MoE ? "moe" : "dense";
}

std::string DecoderPlanPolicy::dump() const {
    const std::string layout = attention_layout == DecoderIntermediateLayout::HeadMajor
        ? "head_major" : "token_major";
    const std::string schedule = schedule_family == DecoderScheduleFamily::Latency
        ? "latency" : "throughput";
    const std::string placement = core_placement == DecoderCorePlacement::Compact
        ? "compact" : "spread";
    return "decoder.policy layout=" + layout + " schedule=" + schedule +
           " placement=" + placement + " materialize=" +
           std::to_string(materialize_attention_output) + " reuse=" +
           std::to_string(reuse_workspace) + " threads=" + std::to_string(threads) +
           " attention=" + attention_strategy_name(attention_strategy);
}

std::string DecoderPlanCost::dump() const {
    std::ostringstream out;
    out << "decoder.plan_cost measured=" << measured
        << " latency_ms=" << measured_milliseconds
        << " analytical=" << analytical_score
        << " workspace=" << workspace_bytes
        << " materialization=" << materialization_bytes;
    return out.str();
}

std::string DecoderOptimizationResult::dump() const {
    std::ostringstream out;
    out << "decoder.optimizer candidates=" << candidates.size()
        << " hardware_measurements=" << hardware_measurements
        << " speedup=" << speedup << '\n'
        << "baseline " << baseline.dump() << '\n'
        << "winner " << winner.dump() << '\n'
        << plan.policy.dump();
    return out.str();
}

std::string decoder_evidence_name(DecoderEvidenceKind kind) {
    if (kind == DecoderEvidenceKind::Measured) return "measured";
    if (kind == DecoderEvidenceKind::CompileOnly) return "compile-only";
    return "skipped";
}

std::string DecoderConstant::dump() const {
    std::ostringstream out;
    out << "constant @" << name << " logical=[";
    for (std::size_t index = 0; index < logical_shape.size(); ++index) {
        if (index) out << ',';
        out << logical_shape[index];
    }
    out << "] packed=[";
    for (std::size_t index = 0; index < packed_shape.size(); ++index) {
        if (index) out << ',';
        out << packed_shape[index];
    }
    out << "] layout=" << layout << " bytes=" << bytes;
    return out.str();
}

std::string DecoderMemoryPlan::dump() const {
    return "decoder.memory naive=" + std::to_string(naive_bytes) +
           " workspace=" + std::to_string(workspace_bytes);
}

std::string DecoderFusionSummary::dump() const {
    return "decoder.fusion qkv=" + std::to_string(fused_qkv) +
           " gate_up=" + std::to_string(fused_gate_up) +
           " rope=" + std::to_string(fused_rope);
}

DecoderFusionSummary DecoderFusionPass::run(const TensorGraph& graph, DecoderFFNKind) const {
    DecoderFusionSummary result;
    std::unordered_map<int, int> matmuls_by_input;
    for (const auto& operation : graph.operations()) {
        if (operation.kind == GraphOpKind::MatMul && !operation.inputs.empty())
            ++matmuls_by_input[operation.inputs.front()];
        if (operation.kind == GraphOpKind::RoPE || operation.name.find("rope") != std::string::npos)
            result.fused_rope = true;
        if (operation.kind == GraphOpKind::SwiGLU && operation.inputs.size() >= 2) {
            const auto& gate = graph.values().at(static_cast<std::size_t>(operation.inputs[0]));
            const auto& up = graph.values().at(static_cast<std::size_t>(operation.inputs[1]));
            if (gate.producer >= 0 && up.producer >= 0) {
                const auto& gate_op = graph.operations().at(static_cast<std::size_t>(gate.producer));
                const auto& up_op = graph.operations().at(static_cast<std::size_t>(up.producer));
                result.fused_gate_up = gate_op.kind == GraphOpKind::MatMul &&
                    up_op.kind == GraphOpKind::MatMul && !gate_op.inputs.empty() &&
                    !up_op.inputs.empty() && gate_op.inputs.front() == up_op.inputs.front();
            }
        }
    }
    result.fused_qkv = std::any_of(matmuls_by_input.begin(), matmuls_by_input.end(),
        [](const auto& entry) { return entry.second >= 3; }) || std::any_of(
        graph.operations().begin(), graph.operations().end(), [](const GraphOperation& operation) {
            return operation.kind == GraphOpKind::FusedQKVProjection;
        });
    result.fused_gate_up = result.fused_gate_up || std::any_of(
        graph.operations().begin(), graph.operations().end(), [](const GraphOperation& operation) {
            return operation.kind == GraphOpKind::FusedGateUpProjection;
        });
    return result;
}

TensorGraph build_decoder_layer_graph(const DecoderConfig& config) {
    validate(config);
    TensorGraph graph;
    const int row_count = rows(config);
    const GraphTensorType hidden_type = tensor_type(
        {Dimension::fixed(row_count), Dimension::fixed(config.hidden)});
    const int input = graph.addInput("x", hidden_type);
    const int rms1_weight = graph.addInput("rms1_weight",
        tensor_type({Dimension::fixed(config.hidden)}), true);
    const int normalized1 = graph.addOperation(GraphOpKind::RmsNorm, "attention_rms_norm",
                                               {input, rms1_weight}, hidden_type);
    const int qkv_weight = graph.addInput("packed_qkv_weight", tensor_type(
        {Dimension::fixed(config.hidden), Dimension::fixed(qkv_width(config))}), true);
    const int qkv = graph.addOperation(GraphOpKind::FusedQKVProjection, "fused_qkv_projection",
        {normalized1, qkv_weight}, tensor_type(
            {Dimension::fixed(row_count), Dimension::fixed(qkv_width(config))}));
    const GraphTensorType q_type = tensor_type({Dimension::fixed(config.batch),
        Dimension::fixed(config.query_heads), Dimension::fixed(config.sequence),
        Dimension::fixed(config.head_dim)});
    const GraphTensorType kv_type = tensor_type({Dimension::fixed(config.batch),
        Dimension::fixed(config.kv_heads), Dimension::fixed(config.sequence),
        Dimension::fixed(config.head_dim)});
    const int query = graph.addOperation(GraphOpKind::Split, "split_query", {qkv}, q_type);
    const int key = graph.addOperation(GraphOpKind::Split, "split_key", {qkv}, kv_type);
    const int value = graph.addOperation(GraphOpKind::Split, "split_value", {qkv}, kv_type);
    const int rope_query = graph.addOperation(GraphOpKind::RoPE, "rope_query", {query}, q_type);
    const int rope_key = graph.addOperation(GraphOpKind::RoPE, "rope_key", {key}, kv_type);
    const int attention = graph.addOperation(GraphOpKind::AttentionSdpa, "flash_attention",
                                              {rope_query, rope_key, value}, q_type);
    const int merged = graph.addOperation(GraphOpKind::Concat, "merge_heads", {attention}, hidden_type);
    const int output_weight = graph.addInput("output_weight", tensor_type(
        {Dimension::fixed(config.hidden), Dimension::fixed(config.hidden)}), true);
    const int projected = graph.addOperation(GraphOpKind::MatMul, "output_projection",
                                              {merged, output_weight}, hidden_type);
    const int attention_residual = graph.addOperation(GraphOpKind::Add, "attention_residual",
                                                       {projected, input}, hidden_type);
    const int rms2_weight = graph.addInput("rms2_weight",
        tensor_type({Dimension::fixed(config.hidden)}), true);
    const int normalized2 = graph.addOperation(GraphOpKind::RmsNorm, "ffn_rms_norm",
                                                {attention_residual, rms2_weight}, hidden_type);
    int ffn_output = -1;
    if (config.ffn == DecoderFFNKind::Dense) {
        const int gate_up_weight = graph.addInput("packed_gate_up_weight", tensor_type(
            {Dimension::fixed(config.hidden), Dimension::fixed(2 * config.intermediate)}), true);
        const int gate_up = graph.addOperation(GraphOpKind::FusedGateUpProjection,
            "fused_gate_up_projection", {normalized2, gate_up_weight},
            tensor_type({Dimension::fixed(row_count), Dimension::fixed(2 * config.intermediate)}));
        const int activated = graph.addOperation(GraphOpKind::SwiGLU, "swiglu", {gate_up},
            tensor_type({Dimension::fixed(row_count), Dimension::fixed(config.intermediate)}));
        const int down_weight = graph.addInput("down_weight", tensor_type(
            {Dimension::fixed(config.intermediate), Dimension::fixed(config.hidden)}), true);
        ffn_output = graph.addOperation(GraphOpKind::MatMul, "down_projection",
                                        {activated, down_weight}, hidden_type);
    } else {
        const int router_weight = graph.addInput("router_weight", tensor_type(
            {Dimension::fixed(config.hidden), Dimension::fixed(config.experts)}), true);
        const int router = graph.addOperation(GraphOpKind::MatMul, "router_projection",
            {normalized2, router_weight}, tensor_type(
                {Dimension::fixed(row_count), Dimension::fixed(config.experts)}));
        const int topk = graph.addOperation(GraphOpKind::TopK, "router_topk", {router});
        const int dispatch = graph.addOperation(GraphOpKind::MoeDispatch, "token_dispatch", {normalized2, topk});
        const int expert = graph.addOperation(GraphOpKind::MoeGroupedMatMul, "expert_grouped_ffn", {dispatch});
        ffn_output = graph.addOperation(GraphOpKind::MoeCombine, "expert_combine", {expert, topk}, hidden_type);
    }
    const int output = graph.addOperation(GraphOpKind::Add, "ffn_residual",
                                           {ffn_output, attention_residual}, hidden_type);
    graph.setReturn(output);
    ShapeInferencePass{}.run(graph);
    return graph;
}

DecoderData make_decoder_data(const DecoderConfig& config, std::uint32_t seed) {
    validate(config);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.15F, 0.15F);
    DecoderData data;
    const auto fill = [&](std::vector<float>& values, std::size_t size) {
        values.resize(size);
        for (float& value : values) value = distribution(generator);
    };
    fill(data.input, static_cast<std::size_t>(rows(config) * config.hidden));
    data.rms1_weight.assign(static_cast<std::size_t>(config.hidden), 1.0F);
    data.rms2_weight.assign(static_cast<std::size_t>(config.hidden), 1.0F);
    fill(data.query_weight, static_cast<std::size_t>(config.hidden * config.hidden));
    fill(data.key_weight, static_cast<std::size_t>(config.hidden * kv_width(config)));
    fill(data.value_weight, static_cast<std::size_t>(config.hidden * kv_width(config)));
    fill(data.output_weight, static_cast<std::size_t>(config.hidden * config.hidden));
    fill(data.gate_weight, static_cast<std::size_t>(config.hidden * config.intermediate));
    fill(data.up_weight, static_cast<std::size_t>(config.hidden * config.intermediate));
    fill(data.down_weight, static_cast<std::size_t>(config.intermediate * config.hidden));
    if (config.ffn == DecoderFFNKind::MoE) {
        data.moe = make_moe_data({rows(config), config.hidden, config.intermediate,
                                  config.experts, config.top_k}, seed + 1);
        data.moe.input = data.input;
    }
    if (config.context_sequence > 0) {
        fill(data.cached_key, static_cast<std::size_t>(config.batch) * config.kv_heads *
             config.context_sequence * config.head_dim);
        fill(data.cached_value, static_cast<std::size_t>(config.batch) * config.kv_heads *
             config.context_sequence * config.head_dim);
    }
    return data;
}

DecoderCompiler::DecoderCompiler(TargetInfo target) : target_(std::move(target)) {}

DecoderExecutablePlan DecoderCompiler::compile(const TensorGraph& imported_graph,
                                                const DecoderConfig& config,
                                                const DecoderData& data,
                                                DecoderCompileOptions options) const {
    const auto compile_start = std::chrono::steady_clock::now();
    validate(config);
    DecoderExecutablePlan plan;
    plan.config = config;
    plan.policy = options.policy.value_or(DecoderPlanPolicy{});
    plan.policy.threads = std::clamp(options.policy && plan.policy.threads > 0
        ? plan.policy.threads : options.max_threads, 1, target_.logical_cpus);
    plan.imported_graph = imported_graph;
    plan.fusion = DecoderFusionPass{}.run(imported_graph, config.ffn);
    if (!plan.fusion.fused_qkv || !plan.fusion.fused_rope ||
        (config.ffn == DecoderFFNKind::Dense && !plan.fusion.fused_gate_up))
        throw std::invalid_argument("decoder graph is missing required fusion patterns");
    plan.graph = build_decoder_layer_graph(config);
    if (options.specialize_constants)
        plan.packed_qkv_weight = concatenate_columns(
            {data.query_weight, data.key_weight, data.value_weight}, config.hidden,
            {config.hidden, kv_width(config), kv_width(config)});
    plan.constants.push_back({"packed_qkv_weight",
        {config.hidden, config.hidden, kv_width(config), kv_width(config)},
        {config.hidden, qkv_width(config)}, "qkv_interleaved_nr" ,
        static_cast<std::size_t>(config.hidden) * qkv_width(config) * sizeof(float)});
    if (config.ffn == DecoderFFNKind::Dense) {
        if (options.specialize_constants)
            plan.packed_gate_up_weight = concatenate_columns(
                {data.gate_weight, data.up_weight}, config.hidden,
                {config.intermediate, config.intermediate});
        plan.constants.push_back({"packed_gate_up_weight",
            {config.hidden, config.intermediate, config.intermediate},
            {config.hidden, 2 * config.intermediate}, "gate_up_interleaved_nr",
            static_cast<std::size_t>(config.hidden) * 2 * config.intermediate * sizeof(float)});
    }
    const auto schedule = decoder_schedule(target_, plan.policy);
    plan.qkv_loop = apply_schedule({rows(config), qkv_width(config), config.hidden, false, false}, schedule);
    plan.output_loop = apply_schedule({rows(config), config.hidden, config.hidden, false, false}, schedule);
    if (config.ffn == DecoderFFNKind::Dense) {
        plan.gate_up_loop = apply_schedule({rows(config), 2 * config.intermediate,
                                            config.hidden, false, false}, schedule);
        plan.down_loop = apply_schedule({rows(config), config.hidden,
                                         config.intermediate, false, false}, schedule);
    }
    const int key_sequence = config.context_sequence > 0
        ? config.context_sequence : config.sequence;
    AttentionConfig attention_config{config.batch, config.query_heads, config.kv_heads,
        config.sequence, key_sequence, config.head_dim, config.head_dim,
        config.causal, 0.0F};
    AttentionPlanOptions attention_options = select_attention_plan(
        attention_config, target_, plan.policy.threads);
    if (options.policy) attention_options.strategy = plan.policy.attention_strategy;
    else plan.policy.attention_strategy = attention_options.strategy;
    if (options.tune_attention)
        attention_options = tune_attention_plan(attention_config,
            make_attention_data(attention_config, 19), target_, plan.policy.threads, 0, 1);
    plan.attention = AttentionCompiler{target_}.compile(attention_config, attention_options);
    plan.llvm_compile_milliseconds += plan.attention.llvm_compile_milliseconds;
    if (config.ffn == DecoderFFNKind::MoE) {
        MoeExecutionSchedule moe_schedule;
        moe_schedule.threads = plan.policy.threads;
        plan.moe = MoeCompiler{target_}.compile(
            {rows(config), config.hidden, config.intermediate, config.experts, config.top_k},
            moe_schedule);
        plan.llvm_compile_milliseconds += plan.moe->llvm_compile_milliseconds;
    }
    if (options.emit_llvm) {
        const auto append_llvm = [&](const LoopIR& loop) {
            const auto compiled = compile_llvm(loop);
            plan.llvm_compile_milliseconds += compiled.compile_milliseconds;
            plan.llvm_ir.push_back(compiled.llvm_ir);
        };
        append_llvm(plan.qkv_loop);
        append_llvm(plan.output_loop);
        if (config.ffn == DecoderFFNKind::Dense) {
            append_llvm(plan.gate_up_loop);
            append_llvm(plan.down_loop);
        }
    }
    const auto memory_planning_start = std::chrono::steady_clock::now();
    const std::size_t hidden_bytes = static_cast<std::size_t>(rows(config) * config.hidden) * sizeof(float);
    const std::size_t qkv_bytes = static_cast<std::size_t>(rows(config) * qkv_width(config)) * sizeof(float);
    const std::size_t gate_up_bytes = config.ffn == DecoderFFNKind::Dense
        ? static_cast<std::size_t>(rows(config) * 2 * config.intermediate) * sizeof(float) : 0;
    const std::size_t activation_bytes = config.ffn == DecoderFFNKind::Dense
        ? static_cast<std::size_t>(rows(config) * config.intermediate) * sizeof(float) : 0;
    plan.memory.naive_bytes = 6 * hidden_bytes + qkv_bytes + gate_up_bytes + activation_bytes +
                              plan.attention.memory.temporary_bytes +
                              (plan.moe ? plan.moe->memory.naive_bytes : 0);
    plan.memory.workspace_bytes = plan.policy.reuse_workspace ? std::max({qkv_bytes, 3 * hidden_bytes,
        gate_up_bytes + activation_bytes,
        plan.attention.memory.temporary_bytes,
        plan.moe ? plan.moe->memory.workspace_bytes : std::size_t{0}})
        : plan.memory.naive_bytes;
    if (plan.policy.materialize_attention_output)
        plan.memory.workspace_bytes += hidden_bytes;
    plan.memory_planning_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - memory_planning_start).count();
    plan.hardware = target_.str();
    plan.compile_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - compile_start).count();
    return plan;
}

std::vector<float> reference_decoder_layer(const DecoderConfig& config,
                                           const DecoderData& data) {
    validate(config);
    const int row_count = rows(config);
    const auto normalized1 = rms_norm(data.input, data.rms1_weight, row_count,
                                      config.hidden, config.rms_epsilon);
    const auto packed_qkv = concatenate_columns(
        {data.query_weight, data.key_weight, data.value_weight}, config.hidden,
        {config.hidden, kv_width(config), kv_width(config)});
    const auto qkv = scalar_matmul(normalized1, packed_qkv, row_count,
                                   qkv_width(config), config.hidden);
    auto attention_data = split_qkv(config, qkv);
    const int key_sequence = config.context_sequence > 0
        ? config.context_sequence : config.sequence;
    AttentionConfig attention_config{config.batch, config.query_heads, config.kv_heads,
        config.sequence, key_sequence, config.head_dim, config.head_dim,
        config.causal, 0.0F};
    std::vector<float> attended;
    if (config.context_sequence > 0) {
        AttentionData decode_data;
        decode_data.query = attention_data.query;
        decode_data.key = data.cached_key;
        decode_data.value = data.cached_value;
        attended = reference_attention(attention_config, decode_data);
    } else {
        attended = reference_attention(attention_config, attention_data);
    }
    const auto merged = merge_attention(config, attended);
    const auto projected = scalar_matmul(merged, data.output_weight, row_count,
                                         config.hidden, config.hidden);
    const auto attention_residual = add(projected, data.input);
    const auto normalized2 = rms_norm(attention_residual, data.rms2_weight, row_count,
                                      config.hidden, config.rms_epsilon);
    std::vector<float> ffn;
    if (config.ffn == DecoderFFNKind::Dense) {
        const auto packed_gate_up = concatenate_columns(
            {data.gate_weight, data.up_weight}, config.hidden,
            {config.intermediate, config.intermediate});
        const auto gate_up = scalar_matmul(normalized2, packed_gate_up, row_count,
                                            2 * config.intermediate, config.hidden);
        ffn = scalar_matmul(swiglu(gate_up, row_count, config.intermediate),
                            data.down_weight, row_count, config.hidden, config.intermediate);
    } else {
        const auto routing = data.routing_trace.value_or(route_topk(
            {row_count, config.hidden, config.intermediate, config.experts, config.top_k},
            data.moe, normalized2));
        ffn = reference_moe(
            {row_count, config.hidden, config.intermediate, config.experts, config.top_k},
            data.moe, normalized2, routing);
    }
    return add(attention_residual, ffn);
}

DecoderBenchmarkResult execute_decoder_layer(const DecoderExecutablePlan& plan,
                                             const DecoderData& data,
                                             int warmup, int repetitions) {
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        (void)execute_once(plan, data);
    std::vector<DecoderBenchmarkResult> measured;
    measured.reserve(static_cast<std::size_t>(std::max(1, repetitions)));
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration)
        measured.push_back(execute_once(plan, data));
    std::sort(measured.begin(), measured.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.milliseconds < rhs.milliseconds;
    });
    auto result = std::move(measured[measured.size() / 2]);
    result.tokens_per_second = result.milliseconds > 0.0
        ? static_cast<double>(rows(plan.config)) * 1000.0 / result.milliseconds : 0.0;
    result.max_error = max_abs_error(reference_decoder_layer(plan.config, data), result.output);
    return result;
}

std::string DecoderExecutablePlan::dump() const {
    std::ostringstream out;
    out << "sfe.decoder_executable version=0.9.0 ffn=" << decoder_ffn_name(config.ffn)
        << " hardware=\"" << hardware << "\" {\n"
        << "imported {\n" << imported_graph.dump() << "}\ncanonical {\n" << graph.dump() << "}\n"
        << fusion.dump() << '\n' << policy.dump()
        << "\ncompile_ms=" << compile_milliseconds
        << " llvm_compile_ms=" << llvm_compile_milliseconds
        << " memory_planning_ms=" << memory_planning_milliseconds
        << "\nconstants {\n";
    for (const auto& constant : constants) out << "  " << constant.dump() << '\n';
    out << "}\n" << memory.dump() << "\nkernels {\n"
        << "  qkv {\n" << qkv_loop.dump() << "  }\n"
        << "  attention {\n" << attention.dump() << "  }\n"
        << "  output_projection {\n" << output_loop.dump() << "  }\n";
    if (config.ffn == DecoderFFNKind::Dense) {
        out << "  gate_up {\n" << gate_up_loop.dump() << "  }\n"
            << "  down {\n" << down_loop.dump() << "  }\n";
    } else {
        out << "  moe {\n" << moe->dump() << "  }\n";
    }
    out << "}\nllvm_kernels=" << llvm_ir.size() << "\n}\n";
    return out.str();
}

void DecoderExecutablePlan::save(const std::filesystem::path& path) const {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write decoder executable plan: " + path.string());
    output << dump();
}

ExecutablePlanOptimizer::ExecutablePlanOptimizer(TargetInfo target) : target_(std::move(target)) {}

DecoderPlanCost ExecutablePlanOptimizer::evaluate(const DecoderExecutablePlan& plan) const {
    DecoderPlanCost cost;
    cost.workspace_bytes = plan.memory.workspace_bytes;
    cost.materialization_bytes = plan.policy.materialize_attention_output
        ? static_cast<std::size_t>(rows(plan.config)) * plan.config.hidden * sizeof(float) : 0;
    const double layout_penalty = plan.policy.attention_layout == DecoderIntermediateLayout::HeadMajor
        ? 0.0 : static_cast<double>(rows(plan.config)) * plan.config.hidden / 4096.0;
    const double materialization_penalty = static_cast<double>(cost.materialization_bytes) / 65536.0;
    const double workspace_penalty = static_cast<double>(cost.workspace_bytes) / (1024.0 * 1024.0);
    const double schedule_penalty = plan.policy.schedule_family == DecoderScheduleFamily::Latency &&
        rows(plan.config) > 32 ? 8.0 : 0.0;
    const double placement_penalty = plan.policy.core_placement == DecoderCorePlacement::Spread &&
        plan.policy.threads <= 2 ? 2.0 : 0.0;
    const double thread_penalty = rows(plan.config) >= 32
        ? 4.0 / static_cast<double>(std::max(1, plan.policy.threads))
        : 0.05 * static_cast<double>(std::max(0, plan.policy.threads - 2));
    cost.analytical_score = layout_penalty + materialization_penalty + workspace_penalty +
        schedule_penalty + placement_penalty + thread_penalty +
        plan.attention.simulation.estimated_l2_traffic / 1.0e7;
    return cost;
}

DecoderOptimizationResult ExecutablePlanOptimizer::optimize(
    const TensorGraph& graph, const DecoderConfig& config, const DecoderData& data,
    int max_threads, int measurement_budget, int repetitions) const {
    std::vector<DecoderPlanPolicy> policies;
    const std::vector<AttentionLoweringStrategy> attention_strategies = config.context_sequence > 0
        ? std::vector<AttentionLoweringStrategy>{AttentionLoweringStrategy::SplitKVDecode}
        : std::vector<AttentionLoweringStrategy>{AttentionLoweringStrategy::IOAware,
                                                 AttentionLoweringStrategy::TiledMaterialized};
    std::vector<int> thread_choices = {1};
    for (const int threads : {2, 4, 8, max_threads})
        if (threads <= max_threads &&
            std::find(thread_choices.begin(), thread_choices.end(), threads) == thread_choices.end())
            thread_choices.push_back(threads);
    for (const auto layout : {DecoderIntermediateLayout::HeadMajor,
                              DecoderIntermediateLayout::TokenMajor})
        for (const auto family : {DecoderScheduleFamily::Latency,
                                  DecoderScheduleFamily::Throughput})
            for (const bool materialize : {false, true})
                for (const bool reuse : {true, false})
                    for (const auto attention : attention_strategies)
                        for (const int threads : thread_choices) {
                        DecoderPlanPolicy policy;
                        policy.attention_layout = layout;
                        policy.schedule_family = family;
                        policy.core_placement = family == DecoderScheduleFamily::Latency
                            ? DecoderCorePlacement::Compact : DecoderCorePlacement::Spread;
                        policy.materialize_attention_output = materialize;
                        policy.reuse_workspace = reuse;
                        policy.threads = std::clamp(threads, 1, target_.logical_cpus);
                        policy.attention_strategy = attention;
                        policies.push_back(policy);
                    }

    DecoderOptimizationResult result;
    DecoderCompileOptions baseline_options;
    baseline_options.max_threads = max_threads;
    baseline_options.emit_llvm = false;
    auto baseline_plan = DecoderCompiler{target_}.compile(
        graph, config, data, baseline_options);
    const auto baseline_policy = baseline_plan.policy;
    const auto baseline_measured = execute_decoder_layer(
        baseline_plan, data, 3, repetitions);
    result.baseline = evaluate(baseline_plan);
    result.baseline.measured = true;
    result.baseline.measured_milliseconds = baseline_measured.milliseconds;
    ++result.hardware_measurements;
    for (const auto& policy : policies) {
        DecoderCompileOptions options;
        options.max_threads = max_threads;
        options.emit_llvm = false;
        options.policy = policy;
        auto plan = DecoderCompiler{target_}.compile(graph, config, data, options);
        result.candidates.push_back({policy, evaluate(plan)});
    }
    std::stable_sort(result.candidates.begin(), result.candidates.end(),
        [](const DecoderPlanCandidate& lhs, const DecoderPlanCandidate& rhs) {
            return lhs.cost.analytical_score < rhs.cost.analytical_score;
        });

    const int budget = std::clamp(measurement_budget, 1,
                                  static_cast<int>(result.candidates.size()));
    double best_latency = std::numeric_limits<double>::infinity();
    const double baseline_latency = baseline_measured.milliseconds;
    for (int index = 0; index < budget; ++index) {
        DecoderCompileOptions options;
        options.max_threads = max_threads;
        options.emit_llvm = false;
        options.policy = result.candidates[static_cast<std::size_t>(index)].policy;
        auto plan = DecoderCompiler{target_}.compile(graph, config, data, options);
        const auto measured = execute_decoder_layer(plan, data, 3, repetitions);
        auto& cost = result.candidates[static_cast<std::size_t>(index)].cost;
        cost.measured = true;
        cost.measured_milliseconds = measured.milliseconds;
        ++result.hardware_measurements;
        if (measured.milliseconds < best_latency) {
            best_latency = measured.milliseconds;
            result.winner = cost;
            result.plan = std::move(plan);
        }
    }
    if (baseline_latency > best_latency) {
        std::vector<double> confirmed_baseline;
        std::vector<double> confirmed_winner;
        confirmed_baseline.reserve(3);
        confirmed_winner.reserve(3);
        for (int confirmation = 0; confirmation < 3; ++confirmation) {
            confirmed_baseline.push_back(execute_decoder_layer(
                baseline_plan, data, confirmation == 0 ? 1 : 0, repetitions).milliseconds);
            confirmed_winner.push_back(execute_decoder_layer(
                result.plan, data, confirmation == 0 ? 1 : 0, repetitions).milliseconds);
            result.hardware_measurements += 2;
        }
        std::sort(confirmed_baseline.begin(), confirmed_baseline.end());
        std::sort(confirmed_winner.begin(), confirmed_winner.end());
        result.baseline.measured_milliseconds = confirmed_baseline[1];
        result.winner.measured_milliseconds = confirmed_winner[1];
        best_latency = result.winner.measured_milliseconds;
        for (auto& candidate : result.candidates)
            if (candidate.policy == result.plan.policy)
                candidate.cost.measured_milliseconds = best_latency;
    }
    const double confirmed_baseline_latency = result.baseline.measured_milliseconds;
    if (confirmed_baseline_latency <= best_latency) {
        result.plan = std::move(baseline_plan);
        result.winner = result.baseline;
        best_latency = confirmed_baseline_latency;
    }
    result.candidates.insert(result.candidates.begin(),
                             {baseline_policy, result.baseline});
    result.speedup = best_latency > 0.0 ? confirmed_baseline_latency / best_latency : 1.0;
    return result;
}

namespace {

std::uint64_t estimate_decoder_flops(const DecoderConfig& config) {
    const std::uint64_t tokens = static_cast<std::uint64_t>(rows(config));
    const std::uint64_t hidden = static_cast<std::uint64_t>(config.hidden);
    const std::uint64_t intermediate = static_cast<std::uint64_t>(config.intermediate);
    const std::uint64_t kv = static_cast<std::uint64_t>(kv_width(config));
    const std::uint64_t keys = static_cast<std::uint64_t>(
        config.context_sequence > 0 ? config.context_sequence : config.sequence);
    const std::uint64_t projections = 2 * tokens * hidden * (hidden + 2 * kv) +
                                      2 * tokens * hidden * hidden;
    const std::uint64_t attention = 4 * static_cast<std::uint64_t>(config.batch) *
        config.query_heads * config.sequence * keys * config.head_dim;
    const std::uint64_t ffn = config.ffn == DecoderFFNKind::Dense
        ? 6 * tokens * hidden * intermediate
        : 6 * tokens * hidden * intermediate * config.top_k;
    return projections + attention + ffn;
}

std::size_t estimate_decoder_weights(const DecoderConfig& config) {
    const std::size_t hidden = static_cast<std::size_t>(config.hidden);
    const std::size_t intermediate = static_cast<std::size_t>(config.intermediate);
    const std::size_t kv = static_cast<std::size_t>(kv_width(config));
    std::size_t values = hidden * (config.hidden + 2 * kv) + hidden * hidden;
    if (config.ffn == DecoderFFNKind::Dense) values += 3 * hidden * intermediate;
    else values += hidden * config.experts + 3 * hidden * intermediate * config.experts;
    return values * sizeof(float);
}

DecoderBenchmarkProfile make_profile(std::string name, DecoderConfig config,
                                     RoutingDistribution routing = RoutingDistribution::Uniform) {
    DecoderBenchmarkProfile profile;
    profile.name = std::move(name);
    profile.config = config;
    profile.routing = routing;
    profile.estimated_flops = estimate_decoder_flops(config);
    profile.estimated_weight_bytes = estimate_decoder_weights(config);
    profile.estimated_activation_bytes = static_cast<std::size_t>(rows(config)) *
        (6 * config.hidden + 3 * config.intermediate) * sizeof(float);
    return profile;
}

}  // namespace

std::vector<DecoderBenchmarkProfile> realistic_decoder_profiles() {
    std::vector<DecoderBenchmarkProfile> profiles;
    const auto append_dense = [&](const std::string& scale, int hidden, int intermediate,
                                  int heads, int kv_heads, int dim) {
        for (const int sequence : {128, 512, 2048}) {
            DecoderConfig config{1, sequence, hidden, intermediate, heads, kv_heads, dim};
            profiles.push_back(make_profile(scale + "-prefill-s" + std::to_string(sequence), config));
        }
        for (const int context : {128, 512, 2048, 4096}) {
            DecoderConfig config{1, 1, hidden, intermediate, heads, kv_heads, dim};
            config.context_sequence = context;
            profiles.push_back(make_profile(scale + "-decode-kv" + std::to_string(context), config));
        }
    };
    append_dense("tiny", 512, 1376, 8, 2, 64);
    append_dense("medium", 1024, 2816, 16, 4, 64);
    append_dense("large", 4096, 11008, 32, 8, 128);
    for (const auto routing : {RoutingDistribution::Uniform,
                               RoutingDistribution::ModerateSkew,
                               RoutingDistribution::HeavySkew}) {
        DecoderConfig config{1, 32, 512, 1376, 8, 2, 64};
        config.ffn = DecoderFFNKind::MoE;
        config.experts = routing == RoutingDistribution::HeavySkew ? 16 : 8;
        config.top_k = 2;
        profiles.push_back(make_profile("tiny-moe-" + routing_distribution_name(routing),
                                        config, routing));
    }
    return profiles;
}

DecoderBenchmarkRecord benchmark_decoder_profile(
    const DecoderBenchmarkProfile& profile, const TensorGraph& graph, int max_threads,
    std::uint64_t max_real_flops, std::size_t max_real_weight_bytes,
    bool optimize_plan, int repetitions) {
    DecoderBenchmarkRecord record;
    record.profile = profile;
    const bool feasible = profile.estimated_flops <= max_real_flops &&
                          profile.estimated_weight_bytes <= max_real_weight_bytes;
    if (!feasible) {
        DecoderCompileOptions options;
        options.max_threads = max_threads;
        options.emit_llvm = false;
        options.specialize_constants = false;
        auto plan = DecoderCompiler{}.compile(graph, profile.config, {}, options);
        record.evidence = DecoderEvidenceKind::CompileOnly;
        record.plan_cost = ExecutablePlanOptimizer{}.evaluate(plan);
        record.compile_milliseconds = plan.compile_milliseconds;
        record.llvm_compile_milliseconds = plan.llvm_compile_milliseconds;
        record.memory_planning_milliseconds = plan.memory_planning_milliseconds;
        record.note = "exceeds real-execution FLOP or weight-memory budget";
        return record;
    }

    auto data = make_decoder_data(profile.config, 101);
    if (profile.config.ffn == DecoderFFNKind::MoE)
        data.routing_trace = make_routing_trace(
            {rows(profile.config), profile.config.hidden, profile.config.intermediate,
             profile.config.experts, profile.config.top_k}, profile.routing, 103);
    TensorGraph selected_graph = graph;
    if (profile.config.ffn == DecoderFFNKind::MoE)
        selected_graph = StableHLOImporter{}.importFile("examples/decoder_layer_moe.mlir");
    DecoderExecutablePlan plan;
    if (optimize_plan) {
        auto optimized = ExecutablePlanOptimizer{}.optimize(
            selected_graph, profile.config, data, max_threads, 6, repetitions);
        plan = std::move(optimized.plan);
        record.optimizer_speedup = optimized.speedup;
        record.optimizer_baseline_milliseconds = optimized.baseline.measured_milliseconds;
        record.hardware_measurements = optimized.hardware_measurements;
        record.selected_policy = optimized.plan.policy.dump();
        record.optimizer_candidates = optimized.candidates;
        record.plan_cost = optimized.winner;
    } else {
        DecoderCompileOptions options;
        options.max_threads = max_threads;
        options.emit_llvm = false;
        plan = DecoderCompiler{}.compile(selected_graph, profile.config, data, options);
        record.plan_cost = ExecutablePlanOptimizer{}.evaluate(plan);
    }
    record.measured = execute_decoder_layer(plan, data, 1, repetitions);
    record.evidence = DecoderEvidenceKind::Measured;
    record.compile_milliseconds = plan.compile_milliseconds;
    record.llvm_compile_milliseconds = plan.llvm_compile_milliseconds;
    record.memory_planning_milliseconds = plan.memory_planning_milliseconds;
    record.plan_cost.workspace_bytes = plan.memory.workspace_bytes;
    return record;
}

void write_decoder_benchmark_csv(const std::filesystem::path& path,
                                 const std::vector<DecoderBenchmarkRecord>& records) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write decoder benchmark: " + path.string());
    out << "profile,evidence,ffn,batch,sequence,context,hidden,intermediate,q_heads,kv_heads,head_dim,experts,top_k,routing,estimated_flops,weight_bytes,activation_bytes,latency_ms,attention_ms,attention_pct,projection_ms,projection_pct,norm_rope_ms,norm_rope_pct,ffn_ms,ffn_pct,residual_ms,residual_pct,dispatch_ms,dispatch_pct,tokens_per_second,workspace_bytes,compile_ms,llvm_compile_ms,memory_planning_ms,memory_planning_pct,optimizer_baseline_ms,optimizer_speedup,hardware_measurements,max_error,selected_policy,note\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& record : records) {
        const auto& config = record.profile.config;
        const auto percent = [&](double value) {
            return record.measured.milliseconds > 0.0
                ? value * 100.0 / record.measured.milliseconds : 0.0;
        };
        const double memory_percent = record.compile_milliseconds > 0.0
            ? record.memory_planning_milliseconds * 100.0 / record.compile_milliseconds : 0.0;
        out << record.profile.name << ',' << decoder_evidence_name(record.evidence) << ','
            << decoder_ffn_name(config.ffn) << ',' << config.batch << ',' << config.sequence << ','
            << config.context_sequence << ',' << config.hidden << ',' << config.intermediate << ','
            << config.query_heads << ',' << config.kv_heads << ',' << config.head_dim << ','
            << config.experts << ',' << config.top_k << ','
            << routing_distribution_name(record.profile.routing) << ','
            << record.profile.estimated_flops << ',' << record.profile.estimated_weight_bytes << ','
            << record.profile.estimated_activation_bytes << ',' << record.measured.milliseconds << ','
            << record.measured.attention_milliseconds << ',' << percent(record.measured.attention_milliseconds) << ','
            << record.measured.projection_milliseconds << ',' << percent(record.measured.projection_milliseconds) << ','
            << record.measured.norm_rope_milliseconds << ',' << percent(record.measured.norm_rope_milliseconds) << ','
            << record.measured.ffn_milliseconds << ',' << percent(record.measured.ffn_milliseconds) << ','
            << record.measured.residual_milliseconds << ',' << percent(record.measured.residual_milliseconds) << ','
            << record.measured.dispatch_overhead_milliseconds << ',' << percent(record.measured.dispatch_overhead_milliseconds) << ','
            << record.measured.tokens_per_second << ',' << record.plan_cost.workspace_bytes << ','
            << record.compile_milliseconds << ',' << record.llvm_compile_milliseconds << ','
            << record.memory_planning_milliseconds << ',' << memory_percent << ','
            << record.optimizer_baseline_milliseconds << ',' << record.optimizer_speedup << ','
            << record.hardware_measurements << ',' << record.measured.max_error << ','
            << record.selected_policy << ',' << record.note << '\n';
    }
}

void write_decoder_plan_candidates_csv(
    const std::filesystem::path& path,
    const std::vector<DecoderBenchmarkRecord>& records) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write decoder candidates: " + path.string());
    out << "profile,candidate,selected,measured,latency_ms,analytical_score,workspace_bytes,materialization_bytes,layout,schedule,placement,materialize,reuse,threads,attention\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& record : records) {
        for (std::size_t index = 0; index < record.optimizer_candidates.size(); ++index) {
            const auto& candidate = record.optimizer_candidates[index];
            const auto& policy = candidate.policy;
            const std::string policy_dump = policy.dump();
            out << record.profile.name << ',' << index << ','
                << (policy_dump == record.selected_policy) << ',' << candidate.cost.measured << ','
                << candidate.cost.measured_milliseconds << ',' << candidate.cost.analytical_score << ','
                << candidate.cost.workspace_bytes << ',' << candidate.cost.materialization_bytes << ','
                << (policy.attention_layout == DecoderIntermediateLayout::HeadMajor
                    ? "head_major" : "token_major") << ','
                << (policy.schedule_family == DecoderScheduleFamily::Latency
                    ? "latency" : "throughput") << ','
                << (policy.core_placement == DecoderCorePlacement::Compact
                    ? "compact" : "spread") << ','
                << policy.materialize_attention_output << ',' << policy.reuse_workspace << ','
                << policy.threads << ',' << attention_strategy_name(policy.attention_strategy) << '\n';
        }
    }
}

}  // namespace schedforge
