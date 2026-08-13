#include "schedforge/decoder_compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
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
                                   const std::vector<float>& attention) {
    std::vector<float> merged(static_cast<std::size_t>(rows(config) * config.hidden));
    for (int batch = 0; batch < config.batch; ++batch)
        for (int token = 0; token < config.sequence; ++token)
            for (int head = 0; head < config.query_heads; ++head)
                for (int dimension = 0; dimension < config.head_dim; ++dimension)
                    merged[static_cast<std::size_t>(batch * config.sequence + token) * config.hidden +
                           head * config.head_dim + dimension] =
                        attention[(((static_cast<std::size_t>(batch) * config.query_heads + head) *
                                    config.sequence + token) * config.head_dim + dimension)];
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

Schedule decoder_schedule(const TargetInfo& target, int threads) {
    Schedule schedule;
    schedule.threads = std::clamp(threads, 1, target.logical_cpus);
    schedule.pin_threads = schedule.threads > 1;
    schedule.bm = 16;
    schedule.bn = 64;
    schedule.bk = 32;
    schedule.mr = 4;
    schedule.nr = std::max(1, target.vector_width);
    schedule.vector_width = std::max(1, target.vector_width);
    schedule.fused = true;
    return schedule;
}

std::string compile_llvm(const LoopIR& loop) {
    TensorData data;
    data.a.assign(static_cast<std::size_t>(loop.problem.m * loop.problem.k), 0.0F);
    data.b.assign(static_cast<std::size_t>(loop.problem.k * loop.problem.n), 0.0F);
    if (loop.problem.bias) data.bias.assign(static_cast<std::size_t>(loop.problem.n), 0.0F);
    return LLVMJITBackend{}.benchmark(loop, data, 0, 1).llvm_ir;
}

DecoderBenchmarkResult execute_once(const DecoderExecutablePlan& plan,
                                    const DecoderData& data) {
    const auto& config = plan.config;
    const int row_count = rows(config);
    const auto normalized1 = rms_norm(data.input, data.rms1_weight, row_count,
                                      config.hidden, config.rms_epsilon);
    std::vector<float> qkv;
    execute(plan.qkv_loop, {normalized1, plan.packed_qkv_weight, {}, {}, {}}, qkv);
    auto attention_data = split_qkv(config, qkv);
    const auto attention = execute_attention(plan.attention, attention_data, 0, 1);
    const auto merged = merge_attention(config, attention.output);
    std::vector<float> projected;
    execute(plan.output_loop, {merged, data.output_weight, {}, {}, {}}, projected);
    const auto attention_residual = add(projected, data.input);
    const auto normalized2 = rms_norm(attention_residual, data.rms2_weight, row_count,
                                      config.hidden, config.rms_epsilon);

    DecoderBenchmarkResult result;
    result.attention_milliseconds = attention.p50_milliseconds;
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
        MoeData moe_data = data.moe;
        moe_data.input = normalized2;
        const auto moe_result = execute_moe(*plan.moe, moe_data, 0, 1);
        result.output = add(attention_residual, moe_result.output);
        result.ffn_milliseconds = moe_result.p50_milliseconds;
    }
    return result;
}

}  // namespace

std::string decoder_ffn_name(DecoderFFNKind kind) {
    return kind == DecoderFFNKind::MoE ? "moe" : "dense";
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
    return data;
}

DecoderCompiler::DecoderCompiler(TargetInfo target) : target_(std::move(target)) {}

DecoderExecutablePlan DecoderCompiler::compile(const TensorGraph& imported_graph,
                                                const DecoderConfig& config,
                                                const DecoderData& data,
                                                DecoderCompileOptions options) const {
    validate(config);
    DecoderExecutablePlan plan;
    plan.config = config;
    plan.imported_graph = imported_graph;
    plan.fusion = DecoderFusionPass{}.run(imported_graph, config.ffn);
    if (!plan.fusion.fused_qkv || !plan.fusion.fused_rope ||
        (config.ffn == DecoderFFNKind::Dense && !plan.fusion.fused_gate_up))
        throw std::invalid_argument("decoder graph is missing required fusion patterns");
    plan.graph = build_decoder_layer_graph(config);
    plan.packed_qkv_weight = concatenate_columns(
        {data.query_weight, data.key_weight, data.value_weight}, config.hidden,
        {config.hidden, kv_width(config), kv_width(config)});
    plan.constants.push_back({"packed_qkv_weight",
        {config.hidden, config.hidden, kv_width(config), kv_width(config)},
        {config.hidden, qkv_width(config)}, "qkv_interleaved_nr" ,
        plan.packed_qkv_weight.size() * sizeof(float)});
    if (config.ffn == DecoderFFNKind::Dense) {
        plan.packed_gate_up_weight = concatenate_columns(
            {data.gate_weight, data.up_weight}, config.hidden,
            {config.intermediate, config.intermediate});
        plan.constants.push_back({"packed_gate_up_weight",
            {config.hidden, config.intermediate, config.intermediate},
            {config.hidden, 2 * config.intermediate}, "gate_up_interleaved_nr",
            plan.packed_gate_up_weight.size() * sizeof(float)});
    }
    const auto schedule = decoder_schedule(target_, options.max_threads);
    plan.qkv_loop = apply_schedule({rows(config), qkv_width(config), config.hidden, false, false}, schedule);
    plan.output_loop = apply_schedule({rows(config), config.hidden, config.hidden, false, false}, schedule);
    if (config.ffn == DecoderFFNKind::Dense) {
        plan.gate_up_loop = apply_schedule({rows(config), 2 * config.intermediate,
                                            config.hidden, false, false}, schedule);
        plan.down_loop = apply_schedule({rows(config), config.hidden,
                                         config.intermediate, false, false}, schedule);
    }
    AttentionConfig attention_config{config.batch, config.query_heads, config.kv_heads,
        config.sequence, config.sequence, config.head_dim, config.head_dim,
        config.causal, 0.0F};
    AttentionPlanOptions attention_options = select_attention_plan(
        attention_config, target_, options.max_threads);
    if (options.tune_attention)
        attention_options = tune_attention_plan(attention_config,
            make_attention_data(attention_config, 19), target_, options.max_threads, 0, 1);
    plan.attention = AttentionCompiler{target_}.compile(attention_config, attention_options);
    if (config.ffn == DecoderFFNKind::MoE) {
        MoeExecutionSchedule moe_schedule;
        moe_schedule.threads = std::clamp(options.max_threads, 1, target_.logical_cpus);
        plan.moe = MoeCompiler{target_}.compile(
            {rows(config), config.hidden, config.intermediate, config.experts, config.top_k},
            moe_schedule);
    }
    plan.llvm_ir.push_back(compile_llvm(plan.qkv_loop));
    plan.llvm_ir.push_back(compile_llvm(plan.output_loop));
    if (config.ffn == DecoderFFNKind::Dense) {
        plan.llvm_ir.push_back(compile_llvm(plan.gate_up_loop));
        plan.llvm_ir.push_back(compile_llvm(plan.down_loop));
    }
    const std::size_t hidden_bytes = static_cast<std::size_t>(rows(config) * config.hidden) * sizeof(float);
    const std::size_t qkv_bytes = static_cast<std::size_t>(rows(config) * qkv_width(config)) * sizeof(float);
    const std::size_t gate_up_bytes = config.ffn == DecoderFFNKind::Dense
        ? static_cast<std::size_t>(rows(config) * 2 * config.intermediate) * sizeof(float) : 0;
    const std::size_t activation_bytes = config.ffn == DecoderFFNKind::Dense
        ? static_cast<std::size_t>(rows(config) * config.intermediate) * sizeof(float) : 0;
    plan.memory.naive_bytes = 6 * hidden_bytes + qkv_bytes + gate_up_bytes + activation_bytes +
                              plan.attention.memory.temporary_bytes +
                              (plan.moe ? plan.moe->memory.naive_bytes : 0);
    plan.memory.workspace_bytes = std::max({qkv_bytes, 3 * hidden_bytes,
        gate_up_bytes + activation_bytes,
        plan.attention.memory.temporary_bytes,
        plan.moe ? plan.moe->memory.workspace_bytes : std::size_t{0}});
    plan.hardware = target_.str();
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
    AttentionConfig attention_config{config.batch, config.query_heads, config.kv_heads,
        config.sequence, config.sequence, config.head_dim, config.head_dim,
        config.causal, 0.0F};
    const auto attended = reference_attention(attention_config, attention_data);
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
        MoeData moe_data = data.moe;
        moe_data.input = normalized2;
        const auto routing = route_topk(
            {row_count, config.hidden, config.intermediate, config.experts, config.top_k},
            moe_data);
        ffn = reference_moe(
            {row_count, config.hidden, config.intermediate, config.experts, config.top_k},
            moe_data, routing);
    }
    return add(attention_residual, ffn);
}

DecoderBenchmarkResult execute_decoder_layer(const DecoderExecutablePlan& plan,
                                             const DecoderData& data,
                                             int warmup, int repetitions) {
    DecoderBenchmarkResult result;
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        (void)execute_once(plan, data);
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        result = execute_once(plan, data);
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    result.milliseconds = timings[timings.size() / 2];
    result.max_error = max_abs_error(reference_decoder_layer(plan.config, data), result.output);
    return result;
}

std::string DecoderExecutablePlan::dump() const {
    std::ostringstream out;
    out << "sfe.decoder_executable version=0.7.0 ffn=" << decoder_ffn_name(config.ffn)
        << " hardware=\"" << hardware << "\" {\n"
        << "imported {\n" << imported_graph.dump() << "}\ncanonical {\n" << graph.dump() << "}\n"
        << fusion.dump() << "\nconstants {\n";
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

}  // namespace schedforge
