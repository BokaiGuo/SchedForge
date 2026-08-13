#include "schedforge/moe_compiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace schedforge {
namespace {

std::string moe_op_name(MoeOpKind kind) {
    switch (kind) {
        case MoeOpKind::RouterMatMul: return "tensor.matmul";
        case MoeOpKind::Softmax: return "tensor.softmax";
        case MoeOpKind::TopK: return "tensor.topk";
        case MoeOpKind::Histogram: return "moe.histogram";
        case MoeOpKind::PrefixSum: return "moe.prefix_sum";
        case MoeOpKind::Dispatch: return "moe.dispatch";
        case MoeOpKind::GroupedMatMulW1: return "moe.grouped_matmul.w1";
        case MoeOpKind::GroupedMatMulW3: return "moe.grouped_matmul.w3";
        case MoeOpKind::SwiGLU: return "tensor.swiglu";
        case MoeOpKind::GroupedMatMulW2: return "moe.grouped_matmul.w2";
        case MoeOpKind::WeightedCombine: return "moe.weighted_combine";
    }
    return "moe.unknown";
}

void validate_config(const MoeConfig& config) {
    if (config.tokens <= 0 || config.hidden <= 0 || config.intermediate <= 0 ||
        config.experts <= 0 || config.top_k <= 0 || config.top_k > config.experts)
        throw std::invalid_argument("invalid MoE configuration");
}

GraphTensorType graph_type(std::vector<Dimension> shape, DataType dtype = DataType::F32) {
    return {std::move(shape), dtype, {}, std::nullopt, 0, -1};
}

float silu(float value) { return value / (1.0F + std::exp(-value)); }

int bucket_for(int count, const std::vector<int>& buckets) {
    for (const int bucket : buckets) if (count <= bucket) return bucket;
    return buckets.empty() ? count : std::max(count, buckets.back());
}

const MoeKernelVariant& kernel_for(const MoeExecutablePlan& plan, int count) {
    const auto iterator = std::find_if(plan.kernels.begin(), plan.kernels.end(),
        [&](const MoeKernelVariant& kernel) { return count <= kernel.max_tokens; });
    return iterator == plan.kernels.end() ? plan.kernels.back() : *iterator;
}

void execute_loop_kernel(const LoopIR& loop, const float* matrix_a, const float* matrix_b,
                         float* output) {
    const auto execution = analyze_loop_ir(loop);
    const int rows = loop.problem.m;
    const int columns = loop.problem.n;
    const int reduction = loop.problem.k;
    const int row_block = std::clamp(execution.mr, 1, 8);
    const int k_unroll = std::max(1, execution.unroll_k);
    for (int row_base = 0; row_base < rows; row_base += row_block) {
        const int active_rows = std::min(row_block, rows - row_base);
        int column = 0;
#if defined(__AVX2__)
        if (execution.vector_width >= 8) {
            for (; column + 8 <= columns; column += 8) {
                __m256 accumulators[8];
                for (int local_row = 0; local_row < active_rows; ++local_row)
                    accumulators[local_row] = _mm256_setzero_ps();
                for (int k_base = 0; k_base < reduction; k_base += k_unroll) {
                  for (int k = k_base; k < std::min(k_base + k_unroll, reduction); ++k) {
                    const __m256 b = _mm256_loadu_ps(
                        matrix_b + static_cast<std::size_t>(k) * columns + column);
                    for (int local_row = 0; local_row < active_rows; ++local_row) {
                        const __m256 a = _mm256_set1_ps(matrix_a[
                            static_cast<std::size_t>(row_base + local_row) * reduction + k]);
#if defined(__FMA__)
                        accumulators[local_row] = _mm256_fmadd_ps(
                            a, b, accumulators[local_row]);
#else
                        accumulators[local_row] = _mm256_add_ps(
                            accumulators[local_row], _mm256_mul_ps(a, b));
#endif
                    }
                  }
                }
                for (int local_row = 0; local_row < active_rows; ++local_row)
                    _mm256_storeu_ps(output +
                        static_cast<std::size_t>(row_base + local_row) * columns + column,
                        accumulators[local_row]);
            }
        }
#endif
        for (; column < columns; ++column) {
            float accumulators[8] = {};
            for (int k_base = 0; k_base < reduction; k_base += k_unroll)
                for (int k = k_base; k < std::min(k_base + k_unroll, reduction); ++k) {
                    const float b = matrix_b[static_cast<std::size_t>(k) * columns + column];
                    for (int local_row = 0; local_row < active_rows; ++local_row)
                        accumulators[local_row] += matrix_a[
                            static_cast<std::size_t>(row_base + local_row) * reduction + k] * b;
                }
            for (int local_row = 0; local_row < active_rows; ++local_row)
                output[static_cast<std::size_t>(row_base + local_row) * columns + column] =
                    accumulators[local_row];
        }
    }
}

void compute_task(const MoeExecutablePlan& plan, const MoeData& data,
                  const SegmentedTensor& dispatched, const MoeTask& task,
                  std::vector<float>& first_workspace,
                  std::vector<float>& gate_workspace,
                  std::vector<float>& expert_output) {
    const int rows = task.end - task.begin;
    if (rows <= 0) return;
    const auto& kernel = kernel_for(plan, task.bucket);
    const float* input = dispatched.values.data() + static_cast<std::size_t>(task.begin) * plan.config.hidden;
    const float* w1 = data.w1.data() + static_cast<std::size_t>(task.expert) *
        plan.config.hidden * plan.config.intermediate;
    const float* w3 = data.w3.data() + static_cast<std::size_t>(task.expert) *
        plan.config.hidden * plan.config.intermediate;
    const float* w2 = data.w2.data() + static_cast<std::size_t>(task.expert) *
        plan.config.intermediate * plan.config.hidden;
    float* first_output = first_workspace.data() +
        static_cast<std::size_t>(task.begin) * plan.config.intermediate;
    float* gate_output = gate_workspace.data() +
        static_cast<std::size_t>(task.begin) * plan.config.intermediate;
    execute_loop_kernel(kernel.w1_loop.specialize(
        {rows, plan.config.intermediate, plan.config.hidden, false, false}), input, w1,
        first_output);
    execute_loop_kernel(kernel.w3_loop.specialize(
        {rows, plan.config.intermediate, plan.config.hidden, false, false}), input, w3,
        gate_output);
    const std::size_t intermediate_values = static_cast<std::size_t>(rows) *
                                            plan.config.intermediate;
    for (std::size_t index = 0; index < intermediate_values; ++index)
        first_output[index] *= silu(gate_output[index]);
    float* output = expert_output.data() +
        static_cast<std::size_t>(task.begin) * plan.config.hidden;
    execute_loop_kernel(kernel.w2_loop.specialize(
        {rows, plan.config.hidden, plan.config.intermediate, false, false}),
        first_output, w2, output);
}

MoeBenchmarkResult execute_once(const MoeExecutablePlan& plan, const MoeData& data,
                                const RoutingTrace& routing) {
    const auto total_begin = std::chrono::steady_clock::now();
    const auto router_begin = total_begin;
    const auto router_end = std::chrono::steady_clock::now();
    const auto dispatch_begin = router_end;
    const auto dispatched = dispatch_tokens(plan.config, data, routing);
    const auto dispatch_end = std::chrono::steady_clock::now();
    const auto tasks = plan_moe_tasks(routing, plan.schedule);
    const std::size_t assignments = static_cast<std::size_t>(routing.tokens) * routing.top_k;
    std::vector<float> first_workspace(assignments * plan.config.intermediate, 0.0F);
    std::vector<float> gate_workspace(assignments * plan.config.intermediate, 0.0F);
    std::vector<float> expert_output(
        assignments * plan.config.hidden, 0.0F);
    const int workers = plan.schedule.strategy == MoeExecutionStrategy::IndependentExperts ? 1 :
        std::max(1, std::min(plan.schedule.threads,
                                             static_cast<int>(std::max<std::size_t>(1, tasks.size()))));
    std::vector<double> worker_milliseconds(static_cast<std::size_t>(workers), 0.0);
    const auto expert_begin = std::chrono::steady_clock::now();
    if (workers == 1) {
        for (const auto& task : tasks)
            compute_task(plan, data, dispatched, task, first_workspace,
                         gate_workspace, expert_output);
    } else if (plan.schedule.task_scheduling == MoeTaskScheduling::FixedExperts) {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(workers));
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                const auto begin = std::chrono::steady_clock::now();
                for (const auto& task : tasks)
                    if (task.expert % workers == worker)
                        compute_task(plan, data, dispatched, task, first_workspace,
                                     gate_workspace, expert_output);
                const auto end = std::chrono::steady_clock::now();
                worker_milliseconds[static_cast<std::size_t>(worker)] =
                    std::chrono::duration<double, std::milli>(end - begin).count();
            });
        }
        for (auto& thread : threads) thread.join();
    } else {
        std::atomic<std::size_t> next_task{0};
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(workers));
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker] {
                const auto begin = std::chrono::steady_clock::now();
                while (true) {
                    const auto index = next_task.fetch_add(1);
                    if (index >= tasks.size()) break;
                    compute_task(plan, data, dispatched, tasks[index], first_workspace,
                                 gate_workspace, expert_output);
                }
                const auto end = std::chrono::steady_clock::now();
                worker_milliseconds[static_cast<std::size_t>(worker)] =
                    std::chrono::duration<double, std::milli>(end - begin).count();
            });
        }
        for (auto& thread : threads) thread.join();
    }
    const auto expert_end = std::chrono::steady_clock::now();
    const auto combine_begin = expert_end;
    std::vector<float> output(static_cast<std::size_t>(routing.tokens) * plan.config.hidden, 0.0F);
    for (std::size_t assignment = 0; assignment < dispatched.token_ids.size(); ++assignment) {
        const int token = dispatched.token_ids[assignment];
        const float weight = dispatched.routing_weights[assignment];
        for (int feature = 0; feature < plan.config.hidden; ++feature)
            output[static_cast<std::size_t>(token) * plan.config.hidden + feature] +=
                weight * expert_output[assignment * plan.config.hidden + feature];
    }
    const auto combine_end = std::chrono::steady_clock::now();
    const auto [minimum, maximum] = std::minmax_element(worker_milliseconds.begin(), worker_milliseconds.end());
    const double average = std::accumulate(worker_milliseconds.begin(), worker_milliseconds.end(), 0.0) /
                           static_cast<double>(worker_milliseconds.size());
    MoeBenchmarkResult result;
    result.milliseconds = std::chrono::duration<double, std::milli>(combine_end - total_begin).count();
    result.router_milliseconds = std::chrono::duration<double, std::milli>(router_end - router_begin).count();
    result.dispatch_milliseconds = std::chrono::duration<double, std::milli>(dispatch_end - dispatch_begin).count();
    result.expert_milliseconds = std::chrono::duration<double, std::milli>(expert_end - expert_begin).count();
    result.combine_milliseconds = std::chrono::duration<double, std::milli>(combine_end - combine_begin).count();
    result.worker_imbalance = average > 0.0 ? (*maximum - *minimum) / average : 0.0;
    result.expert_counts = routing.counts;
    result.worker_milliseconds = std::move(worker_milliseconds);
    result.output = std::move(output);
    return result;
}

}  // namespace

std::string MoeOperation::str() const {
    std::ostringstream out;
    if (!result.empty()) out << result << " = ";
    out << moe_op_name(kind);
    for (const auto& input : inputs) out << ' ' << input;
    if (!attributes.empty()) out << " {" << attributes << '}';
    return out.str();
}

std::string MoeProgram::dump() const {
    std::ostringstream out;
    out << "moe.program {\n";
    for (const auto& operation : operations) out << "  " << operation.str() << '\n';
    out << "}\n";
    return out.str();
}

std::string SegmentedTensorType::str() const {
    return "!sfg.segmented_tensor<values=tensor<?x" + std::to_string(feature_size) + "x" +
           data_type_name(dtype) + ">, segments=" + std::to_string(segments) + ">";
}

std::string RoutingTrace::dump() const {
    std::ostringstream out;
    out << "routing.trace tokens=" << tokens << " experts=" << experts << " top_k=" << top_k << " counts=[";
    for (std::size_t index = 0; index < counts.size(); ++index) {
        if (index) out << ',';
        out << counts[index];
    }
    out << ']';
    return out.str();
}

std::string SegmentedTensor::dump() const {
    std::ostringstream out;
    out << type.str() << " offsets=[";
    for (std::size_t index = 0; index < offsets.size(); ++index) {
        if (index) out << ',';
        out << offsets[index];
    }
    out << "] assignments=" << token_ids.size();
    return out.str();
}

std::string MoeMemoryPlan::dump() const {
    std::ostringstream out;
    out << "moe.memory naive=" << naive_bytes << " workspace=" << workspace_bytes
        << " routing=" << routing_bytes << " segmented=" << segmented_bytes;
    return out.str();
}

std::string moe_execution_strategy_name(MoeExecutionStrategy strategy) {
    if (strategy == MoeExecutionStrategy::IndependentExperts) return "independent";
    if (strategy == MoeExecutionStrategy::Grouped) return "grouped";
    return "bucketed-grouped";
}

std::string moe_task_scheduling_name(MoeTaskScheduling scheduling) {
    if (scheduling == MoeTaskScheduling::FixedExperts) return "fixed-experts";
    if (scheduling == MoeTaskScheduling::WorkStealing) return "work-stealing";
    return "load-aware-split";
}

std::string routing_distribution_name(RoutingDistribution distribution) {
    if (distribution == RoutingDistribution::Uniform) return "uniform";
    if (distribution == RoutingDistribution::ModerateSkew) return "moderate-skew";
    return "heavy-skew";
}

std::string MoeExecutionSchedule::dump() const {
    std::ostringstream out;
    out << "moe.schedule {\n  strategy " << moe_execution_strategy_name(strategy)
        << "\n  task_scheduling " << moe_task_scheduling_name(task_scheduling)
        << "\n  bucket_m [";
    for (std::size_t index = 0; index < token_buckets.size(); ++index) {
        if (index) out << ',';
        out << token_buckets[index];
    }
    out << "]\n  split_threshold " << split_threshold << "\n  threads " << threads
        << "\n  router.fuse_topk " << (fuse_router_topk ? "true" : "false")
        << "\n  dispatch.histogram_prefix " << (fuse_histogram_prefix ? "true" : "false")
        << "\n  combine.fuse_weight " << (fuse_combine_weight ? "true" : "false") << "\n}\n";
    return out.str();
}

TensorGraph build_moe_mlp_graph(const MoeConfig& config, bool dynamic_tokens) {
    validate_config(config);
    TensorGraph graph;
    const Dimension tokens = dynamic_tokens ? Dimension::dynamic("T") : Dimension::fixed(config.tokens);
    const auto x = graph.addInput("x", graph_type({tokens, Dimension::fixed(config.hidden)}));
    const auto router = graph.addInput("router_weight", graph_type({Dimension::fixed(config.hidden), Dimension::fixed(config.experts)}),
                                       true);
    const auto w1 = graph.addInput("expert_w1", graph_type({Dimension::fixed(config.experts), Dimension::fixed(config.hidden),
                                                   Dimension::fixed(config.intermediate)}), true);
    const auto w3 = graph.addInput("expert_w3", graph_type({Dimension::fixed(config.experts), Dimension::fixed(config.hidden),
                                                   Dimension::fixed(config.intermediate)}), true);
    const auto w2 = graph.addInput("expert_w2", graph_type({Dimension::fixed(config.experts), Dimension::fixed(config.intermediate),
                                                   Dimension::fixed(config.hidden)}), true);
    const auto logits = graph.addOperation(GraphOpKind::MatMul, "router", {x, router},
        graph_type({tokens, Dimension::fixed(config.experts)}));
    const auto scores = graph.addOperation(GraphOpKind::Softmax, "router_softmax", {logits},
        graph_type({tokens, Dimension::fixed(config.experts)}));
    const auto ids = graph.addOperation(GraphOpKind::TopK, "topk_ids", {scores},
        graph_type({tokens, Dimension::fixed(config.top_k)}, DataType::I32));
    graph.operations().back().attributes["k"] = std::to_string(config.top_k);
    const auto weights = graph.addOperation(GraphOpKind::TopK, "topk_weights", {scores},
        graph_type({tokens, Dimension::fixed(config.top_k)}));
    graph.operations().back().attributes["k"] = std::to_string(config.top_k);
    const auto counts = graph.addOperation(GraphOpKind::MoeHistogram, "expert_counts", {ids},
        graph_type({Dimension::fixed(config.experts)}, DataType::I32));
    const auto offsets = graph.addOperation(GraphOpKind::MoePrefixSum, "expert_offsets", {counts},
        graph_type({Dimension::fixed(config.experts + 1)}, DataType::I32));
    const auto dispatched = graph.addOperation(GraphOpKind::MoeDispatch, "dispatch_tokens", {x, ids, weights, offsets},
        graph_type({Dimension::dynamic("assignments"), Dimension::fixed(config.hidden)}));
    const auto gate = graph.addOperation(GraphOpKind::MoeGroupedMatMul, "grouped_w1", {dispatched, offsets, w1},
        graph_type({Dimension::dynamic("assignments"), Dimension::fixed(config.intermediate)}));
    const auto up = graph.addOperation(GraphOpKind::MoeGroupedMatMul, "grouped_w3", {dispatched, offsets, w3},
        graph_type({Dimension::dynamic("assignments"), Dimension::fixed(config.intermediate)}));
    const auto activated = graph.addOperation(GraphOpKind::SwiGLU, "swiglu", {gate, up},
        graph_type({Dimension::dynamic("assignments"), Dimension::fixed(config.intermediate)}));
    const auto expert_output = graph.addOperation(GraphOpKind::MoeGroupedMatMul, "grouped_w2", {activated, offsets, w2},
        graph_type({Dimension::dynamic("assignments"), Dimension::fixed(config.hidden)}));
    const auto combined = graph.addOperation(GraphOpKind::MoeCombine, "weighted_combine",
        {expert_output, ids, weights}, graph_type({tokens, Dimension::fixed(config.hidden)}));
    graph.setReturn(combined);
    return graph;
}

MoeProgram lower_moe_program(const MoeConfig& config) {
    validate_config(config);
    MoeProgram program;
    program.operations = {
        {MoeOpKind::RouterMatMul, "%logits", {"%x", "%router_weight"}, "shape=TxE"},
        {MoeOpKind::Softmax, "%scores", {"%logits"}, "axis=experts"},
        {MoeOpKind::TopK, "%expert_ids,%expert_weights", {"%scores"}, "k=" + std::to_string(config.top_k)},
        {MoeOpKind::Histogram, "%counts", {"%expert_ids"}, "experts=" + std::to_string(config.experts)},
        {MoeOpKind::PrefixSum, "%offsets", {"%counts"}, "exclusive=true"},
        {MoeOpKind::Dispatch, "%permuted,%token_ids", {"%x", "%expert_ids", "%offsets"}, "stable=true"},
        {MoeOpKind::GroupedMatMulW1, "%h", {"%permuted", "%offsets", "%W1"}, "variable_m=true"},
        {MoeOpKind::GroupedMatMulW3, "%g", {"%permuted", "%offsets", "%W3"}, "variable_m=true"},
        {MoeOpKind::SwiGLU, "%a", {"%h", "%g"}, "in_place=%h"},
        {MoeOpKind::GroupedMatMulW2, "%expert_result", {"%a", "%offsets", "%W2"}, "variable_m=true"},
        {MoeOpKind::WeightedCombine, "%y", {"%expert_result", "%token_ids", "%expert_weights"}, "topk_reduce=true"}
    };
    return program;
}

MoeData make_moe_data(const MoeConfig& config, std::uint32_t seed) {
    validate_config(config);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.2F, 0.2F);
    MoeData data;
    data.input.resize(static_cast<std::size_t>(config.tokens) * config.hidden);
    data.router_weight.resize(static_cast<std::size_t>(config.hidden) * config.experts);
    data.w1.resize(static_cast<std::size_t>(config.experts) * config.hidden * config.intermediate);
    data.w3.resize(static_cast<std::size_t>(config.experts) * config.hidden * config.intermediate);
    data.w2.resize(static_cast<std::size_t>(config.experts) * config.intermediate * config.hidden);
    for (auto* values : {&data.input, &data.router_weight, &data.w1, &data.w3, &data.w2})
        for (float& value : *values) value = distribution(generator);
    return data;
}

RoutingTrace route_topk(const MoeConfig& config, const MoeData& data) {
    validate_config(config);
    if (data.input.size() % static_cast<std::size_t>(config.hidden) != 0)
        throw std::invalid_argument("MoE input does not match hidden dimension");
    const int tokens = static_cast<int>(data.input.size() / static_cast<std::size_t>(config.hidden));
    RoutingTrace routing;
    routing.tokens = tokens;
    routing.experts = config.experts;
    routing.top_k = config.top_k;
    routing.expert_ids.resize(static_cast<std::size_t>(tokens) * config.top_k);
    routing.expert_weights.resize(static_cast<std::size_t>(tokens) * config.top_k);
    routing.counts.assign(static_cast<std::size_t>(config.experts), 0);
    std::vector<float> logits(static_cast<std::size_t>(config.experts));
    std::vector<int> order(static_cast<std::size_t>(config.experts));
    std::iota(order.begin(), order.end(), 0);
    for (int token = 0; token < tokens; ++token) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (int expert = 0; expert < config.experts; ++expert) {
            float value = 0.0F;
            for (int feature = 0; feature < config.hidden; ++feature)
                value += data.input[static_cast<std::size_t>(token) * config.hidden + feature] *
                         data.router_weight[static_cast<std::size_t>(feature) * config.experts + expert];
            logits[static_cast<std::size_t>(expert)] = value;
            maximum = std::max(maximum, value);
        }
        float denominator = 0.0F;
        for (float& value : logits) { value = std::exp(value - maximum); denominator += value; }
        std::partial_sort(order.begin(), order.begin() + config.top_k, order.end(),
            [&](int lhs, int rhs) { return logits[static_cast<std::size_t>(lhs)] > logits[static_cast<std::size_t>(rhs)]; });
        float selected_sum = 0.0F;
        for (int slot = 0; slot < config.top_k; ++slot)
            selected_sum += logits[static_cast<std::size_t>(order[static_cast<std::size_t>(slot)])] / denominator;
        for (int slot = 0; slot < config.top_k; ++slot) {
            const int expert = order[static_cast<std::size_t>(slot)];
            const std::size_t index = static_cast<std::size_t>(token) * config.top_k + slot;
            routing.expert_ids[index] = expert;
            routing.expert_weights[index] =
                (logits[static_cast<std::size_t>(expert)] / denominator) / selected_sum;
            ++routing.counts[static_cast<std::size_t>(expert)];
        }
    }
    return routing;
}

RoutingTrace make_routing_trace(const MoeConfig& config, RoutingDistribution distribution,
                                std::uint32_t seed) {
    validate_config(config);
    RoutingTrace routing;
    routing.tokens = config.tokens;
    routing.experts = config.experts;
    routing.top_k = config.top_k;
    routing.expert_ids.resize(static_cast<std::size_t>(config.tokens) * config.top_k);
    routing.expert_weights.resize(routing.expert_ids.size());
    routing.counts.assign(static_cast<std::size_t>(config.experts), 0);
    std::mt19937 generator(seed);
    std::vector<double> probabilities(static_cast<std::size_t>(config.experts), 1.0);
    if (distribution != RoutingDistribution::Uniform) {
        const double exponent = distribution == RoutingDistribution::ModerateSkew ? 0.8 : 1.6;
        for (int expert = 0; expert < config.experts; ++expert)
            probabilities[static_cast<std::size_t>(expert)] = 1.0 / std::pow(expert + 1.0, exponent);
    }
    std::discrete_distribution<int> draw(probabilities.begin(), probabilities.end());
    for (int token = 0; token < config.tokens; ++token) {
        std::vector<int> selected;
        while (static_cast<int>(selected.size()) < config.top_k) {
            const int expert = draw(generator);
            if (std::find(selected.begin(), selected.end(), expert) == selected.end()) selected.push_back(expert);
        }
        for (int slot = 0; slot < config.top_k; ++slot) {
            const std::size_t index = static_cast<std::size_t>(token) * config.top_k + slot;
            routing.expert_ids[index] = selected[static_cast<std::size_t>(slot)];
            routing.expert_weights[index] = 1.0F / static_cast<float>(config.top_k);
            ++routing.counts[static_cast<std::size_t>(selected[static_cast<std::size_t>(slot)])];
        }
    }
    return routing;
}

SegmentedTensor dispatch_tokens(const MoeConfig& config, const MoeData& data,
                                const RoutingTrace& routing) {
    if (routing.experts != config.experts || routing.top_k != config.top_k)
        throw std::invalid_argument("routing trace does not match MoE configuration");
    SegmentedTensor result;
    result.type = {config.experts, config.hidden, DataType::F32};
    result.offsets.resize(static_cast<std::size_t>(config.experts + 1), 0);
    for (int expert = 0; expert < config.experts; ++expert)
        result.offsets[static_cast<std::size_t>(expert + 1)] =
            result.offsets[static_cast<std::size_t>(expert)] + routing.counts[static_cast<std::size_t>(expert)];
    const int assignments = routing.tokens * routing.top_k;
    result.values.resize(static_cast<std::size_t>(assignments) * config.hidden);
    result.token_ids.resize(static_cast<std::size_t>(assignments));
    result.route_slots.resize(static_cast<std::size_t>(assignments));
    result.routing_weights.resize(static_cast<std::size_t>(assignments));
    std::vector<int> cursor = result.offsets;
    for (int token = 0; token < routing.tokens; ++token) {
        for (int slot = 0; slot < routing.top_k; ++slot) {
            const std::size_t route_index = static_cast<std::size_t>(token) * routing.top_k + slot;
            const int expert = routing.expert_ids[route_index];
            const int destination = cursor[static_cast<std::size_t>(expert)]++;
            std::copy_n(data.input.begin() + static_cast<std::ptrdiff_t>(token) * config.hidden,
                        config.hidden,
                        result.values.begin() + static_cast<std::ptrdiff_t>(destination) * config.hidden);
            result.token_ids[static_cast<std::size_t>(destination)] = token;
            result.route_slots[static_cast<std::size_t>(destination)] = slot;
            result.routing_weights[static_cast<std::size_t>(destination)] = routing.expert_weights[route_index];
        }
    }
    return result;
}

std::vector<MoeTask> plan_moe_tasks(const RoutingTrace& routing,
                                    const MoeExecutionSchedule& schedule) {
    std::vector<MoeTask> tasks;
    std::vector<int> offsets(static_cast<std::size_t>(routing.experts + 1), 0);
    for (int expert = 0; expert < routing.experts; ++expert)
        offsets[static_cast<std::size_t>(expert + 1)] = offsets[static_cast<std::size_t>(expert)] +
            routing.counts[static_cast<std::size_t>(expert)];
    for (int expert = 0; expert < routing.experts; ++expert) {
        const int count = routing.counts[static_cast<std::size_t>(expert)];
        if (count == 0) continue;
        const int chunk = schedule.task_scheduling == MoeTaskScheduling::LoadAwareSplit &&
                          count > schedule.split_threshold
            ? std::max(1, schedule.split_threshold) : count;
        for (int begin = offsets[static_cast<std::size_t>(expert)];
             begin < offsets[static_cast<std::size_t>(expert + 1)]; begin += chunk) {
            const int end = std::min(begin + chunk, offsets[static_cast<std::size_t>(expert + 1)]);
            const int bucket = schedule.strategy == MoeExecutionStrategy::BucketedGrouped
                ? bucket_for(end - begin, schedule.token_buckets)
                : (schedule.token_buckets.empty() ? end - begin : schedule.token_buckets.back());
            tasks.push_back({expert, begin, end, bucket,
                             expert % std::max(1, schedule.threads)});
        }
    }
    if (schedule.task_scheduling != MoeTaskScheduling::FixedExperts)
        std::stable_sort(tasks.begin(), tasks.end(), [](const MoeTask& lhs, const MoeTask& rhs) {
            return lhs.end - lhs.begin > rhs.end - rhs.begin;
        });
    return tasks;
}

MoeSimulationResult simulate_moe(const MoeConfig& config, const RoutingTrace& routing,
                                 const MoeExecutionSchedule& schedule) {
    const auto tasks = plan_moe_tasks(routing, schedule);
    const int workers = schedule.strategy == MoeExecutionStrategy::IndependentExperts
        ? 1 : std::max(1, schedule.threads);
    std::vector<double> load(static_cast<std::size_t>(workers), 0.0);
    for (const auto& task : tasks) {
        const double compute = 4.0 * static_cast<double>(task.end - task.begin) *
            config.hidden * config.intermediate;
        if (schedule.task_scheduling == MoeTaskScheduling::FixedExperts) {
            load[static_cast<std::size_t>(task.preferred_worker % workers)] += compute;
        } else {
            const auto least = std::min_element(load.begin(), load.end());
            *least += compute;
        }
    }
    const double maximum = *std::max_element(load.begin(), load.end());
    const double average = std::accumulate(load.begin(), load.end(), 0.0) / workers;
    const int assignments = routing.tokens * routing.top_k;
    MoeSimulationResult result;
    result.expert_counts = routing.counts;
    result.tasks = tasks.size();
    result.imbalance = average > 0.0 ? maximum / average - 1.0 : 0.0;
    result.estimated_makespan = maximum;
    const auto bytes_per_assignment = static_cast<double>(config.hidden) * sizeof(float) +
                                      2.0 * sizeof(int) + sizeof(float);
    result.dispatch_bytes = static_cast<double>(assignments) * bytes_per_assignment;
    result.weight_bytes = static_cast<double>(config.experts) *
        (2.0 * config.hidden * config.intermediate + config.intermediate * config.hidden) * sizeof(float);
    result.worker_utilization = maximum > 0.0 ? average / maximum : 1.0;
    return result;
}

MoeExecutionSchedule select_moe_schedule(const MoeConfig& config,
                                         const RoutingTrace& routing,
                                         const TargetInfo& target,
                                         int max_threads) {
    MoeExecutionSchedule best;
    double best_score = std::numeric_limits<double>::infinity();
    for (const auto strategy : {MoeExecutionStrategy::IndependentExperts,
                                MoeExecutionStrategy::Grouped,
                                MoeExecutionStrategy::BucketedGrouped}) {
        for (const auto scheduler : {MoeTaskScheduling::FixedExperts,
                                     MoeTaskScheduling::WorkStealing,
                                     MoeTaskScheduling::LoadAwareSplit}) {
            MoeExecutionSchedule candidate;
            candidate.strategy = strategy;
            candidate.task_scheduling = scheduler;
            candidate.threads = std::clamp(max_threads, 1, target.logical_cpus);
            candidate.split_threshold = std::max(4, config.tokens /
                std::max(1, candidate.threads));
            const auto simulation = simulate_moe(config, routing, candidate);
            const double scheduling_overhead = static_cast<double>(simulation.tasks) *
                (strategy == MoeExecutionStrategy::IndependentExperts ? 4.0 : 1.0);
            const double bucket_penalty = strategy == MoeExecutionStrategy::BucketedGrouped
                ? static_cast<double>(simulation.tasks) * 0.25 : 0.0;
            const double score = simulation.estimated_makespan + scheduling_overhead +
                                 bucket_penalty + simulation.dispatch_bytes / 64.0;
            if (score < best_score) {
                best_score = score;
                best = candidate;
            }
        }
    }
    return best;
}

std::string MoeSimulationResult::dump() const {
    std::ostringstream out;
    out << "moe.simulation tasks=" << tasks << " imbalance=" << imbalance
        << " utilization=" << worker_utilization << " dispatch_bytes=" << dispatch_bytes
        << " weight_bytes=" << weight_bytes << " counts=[";
    for (std::size_t index = 0; index < expert_counts.size(); ++index) {
        if (index) out << ',';
        out << expert_counts[index];
    }
    out << ']';
    return out.str();
}

std::vector<float> reference_moe(const MoeConfig& config, const MoeData& data,
                                 const RoutingTrace& routing) {
    std::vector<float> output(static_cast<std::size_t>(routing.tokens) * config.hidden, 0.0F);
    std::vector<float> gate(static_cast<std::size_t>(config.intermediate));
    std::vector<float> up(static_cast<std::size_t>(config.intermediate));
    for (int token = 0; token < routing.tokens; ++token) {
        for (int slot = 0; slot < routing.top_k; ++slot) {
            const std::size_t route_index = static_cast<std::size_t>(token) * routing.top_k + slot;
            const int expert = routing.expert_ids[route_index];
            const float weight = routing.expert_weights[route_index];
            for (int intermediate = 0; intermediate < config.intermediate; ++intermediate) {
                float first = 0.0F;
                float third = 0.0F;
                for (int feature = 0; feature < config.hidden; ++feature) {
                    const float input = data.input[static_cast<std::size_t>(token) * config.hidden + feature];
                    const std::size_t weight_index =
                        (static_cast<std::size_t>(expert) * config.hidden + feature) * config.intermediate + intermediate;
                    first += input * data.w1[weight_index];
                    third += input * data.w3[weight_index];
                }
                gate[static_cast<std::size_t>(intermediate)] = first;
                up[static_cast<std::size_t>(intermediate)] = silu(third);
            }
            for (int feature = 0; feature < config.hidden; ++feature) {
                float value = 0.0F;
                for (int intermediate = 0; intermediate < config.intermediate; ++intermediate) {
                    const std::size_t weight_index =
                        (static_cast<std::size_t>(expert) * config.intermediate + intermediate) * config.hidden + feature;
                    value += gate[static_cast<std::size_t>(intermediate)] *
                             up[static_cast<std::size_t>(intermediate)] * data.w2[weight_index];
                }
                output[static_cast<std::size_t>(token) * config.hidden + feature] += weight * value;
            }
        }
    }
    return output;
}

MoeBenchmarkResult execute_moe(const MoeExecutablePlan& plan, const MoeData& data,
                               const RoutingTrace& routing, int warmup, int repetitions) {
    if (plan.kernels.empty()) throw std::invalid_argument("MoE plan has no expert kernels");
    if (routing.tokens > plan.config.tokens)
        throw std::invalid_argument("runtime token count exceeds compiled MoE guard");
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        (void)execute_once(plan, data, routing);
    std::vector<MoeBenchmarkResult> measured;
    measured.reserve(static_cast<std::size_t>(std::max(1, repetitions)));
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration)
        measured.push_back(execute_once(plan, data, routing));
    std::sort(measured.begin(), measured.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.milliseconds < rhs.milliseconds;
    });
    const auto p50_index = measured.size() / 2;
    const auto p95_index = std::min(measured.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(measured.size()))) - 1);
    const double p95 = measured[p95_index].milliseconds;
    auto result = std::move(measured[p50_index]);
    result.p50_milliseconds = result.milliseconds;
    result.p95_milliseconds = p95;
    result.max_error = max_abs_error(reference_moe(plan.config, data, routing), result.output);
    return result;
}

MoeBenchmarkResult execute_moe(const MoeExecutablePlan& plan, const MoeData& data,
                               int warmup, int repetitions) {
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) {
        const auto routing = route_topk(plan.config, data);
        (void)execute_once(plan, data, routing);
    }
    std::vector<MoeBenchmarkResult> measured;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto router_begin = std::chrono::steady_clock::now();
        const auto routing = route_topk(plan.config, data);
        const auto router_end = std::chrono::steady_clock::now();
        auto result = execute_once(plan, data, routing);
        result.router_milliseconds =
            std::chrono::duration<double, std::milli>(router_end - router_begin).count();
        result.milliseconds += result.router_milliseconds;
        measured.push_back(std::move(result));
    }
    std::sort(measured.begin(), measured.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.milliseconds < rhs.milliseconds;
    });
    const auto p50_index = measured.size() / 2;
    const auto p95_index = std::min(measured.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(measured.size()))) - 1);
    const double p95 = measured[p95_index].milliseconds;
    auto result = std::move(measured[p50_index]);
    result.p50_milliseconds = result.milliseconds;
    result.p95_milliseconds = p95;
    const auto routing = route_topk(plan.config, data);
    result.max_error = max_abs_error(reference_moe(plan.config, data, routing), result.output);
    return result;
}

MoeCompiler::MoeCompiler(TargetInfo target) : target_(std::move(target)) {}

MoeExecutablePlan MoeCompiler::compile(const MoeConfig& config,
                                       MoeExecutionSchedule schedule) const {
    validate_config(config);
    schedule.threads = std::clamp(schedule.threads, 1, target_.logical_cpus);
    std::sort(schedule.token_buckets.begin(), schedule.token_buckets.end());
    schedule.token_buckets.erase(std::unique(schedule.token_buckets.begin(), schedule.token_buckets.end()),
                                 schedule.token_buckets.end());
    schedule.token_buckets.erase(std::remove_if(schedule.token_buckets.begin(), schedule.token_buckets.end(),
        [](int value) { return value <= 0; }), schedule.token_buckets.end());
    if (schedule.token_buckets.empty() || schedule.token_buckets.back() < config.tokens)
        schedule.token_buckets.push_back(config.tokens);
    MoeExecutablePlan plan;
    plan.config = config;
    plan.tensor_graph = build_moe_mlp_graph(config, true);
    plan.program = lower_moe_program(config);
    plan.segmented_type = {config.experts, config.hidden, DataType::F32};
    plan.schedule = schedule;
    plan.target = "native-cpu";
    plan.hardware = target_.str();
    plan.guards = {"0 < T", "T <= " + std::to_string(config.tokens),
                   "assignments == T * " + std::to_string(config.top_k)};
    Schedule kernel_schedule;
    kernel_schedule.threads = 1;
    kernel_schedule.pin_threads = false;
    kernel_schedule.fused = false;
    kernel_schedule.vector_width = target_.vector_width;
    kernel_schedule.nr = target_.vector_width >= 8 ? 16 : target_.vector_width;
    kernel_schedule.bm = 16;
    kernel_schedule.bn = 64;
    kernel_schedule.bk = 16;
    for (const int bucket : schedule.token_buckets) {
        kernel_schedule.mr = bucket <= 4 ? 1 : bucket <= 16 ? 4 : 6;
        kernel_schedule.unroll_k = bucket <= 4 ? 1 : bucket <= 16 ? 2 : 4;
        const auto w1_loop = apply_schedule({bucket, config.intermediate, config.hidden, false, false}, kernel_schedule);
        const auto w2_loop = apply_schedule({bucket, config.hidden, config.intermediate, false, false}, kernel_schedule);
        const auto w1_compilation = LLVMJITBackend{}.benchmark(
            w1_loop, make_data(w1_loop.problem, static_cast<std::uint32_t>(bucket)), 0, 1);
        const auto w2_compilation = LLVMJITBackend{}.benchmark(
            w2_loop, make_data(w2_loop.problem, static_cast<std::uint32_t>(bucket + 1)), 0, 1);
        plan.kernels.push_back({bucket, w1_loop, w1_loop, w2_loop,
                                w1_compilation.llvm_ir, w2_compilation.llvm_ir});
    }
    const std::size_t assignments = static_cast<std::size_t>(config.tokens) * config.top_k;
    plan.memory.routing_bytes = assignments * (2 * sizeof(int) + sizeof(float)) +
                                static_cast<std::size_t>(config.experts + 1) * sizeof(int);
    plan.memory.segmented_bytes = assignments * config.hidden * sizeof(float);
    const std::size_t intermediates = assignments * config.intermediate * sizeof(float);
    const std::size_t expert_output = assignments * config.hidden * sizeof(float);
    const std::size_t final_output = static_cast<std::size_t>(config.tokens) *
                                     config.hidden * sizeof(float);
    plan.memory.naive_bytes = plan.memory.routing_bytes + plan.memory.segmented_bytes +
                              3 * intermediates + expert_output + final_output;
    plan.memory.workspace_bytes = plan.memory.routing_bytes + plan.memory.segmented_bytes +
                                  2 * intermediates + expert_output + final_output;
    return plan;
}

std::string MoeExecutablePlan::dump() const {
    std::ostringstream out;
    out << "sfe.moe_executable version=0.4.0 target=" << target << '\n'
        << "hardware=" << hardware << '\n'
        << "config tokens<= " << config.tokens << " hidden=" << config.hidden
        << " intermediate=" << config.intermediate << " experts=" << config.experts
        << " top_k=" << config.top_k << '\n'
        << tensor_graph.dump() << program.dump()
        << "segmented_type " << segmented_type.str() << '\n'
        << schedule.dump() << "guards {\n";
    for (const auto& guard : guards) out << "  " << guard << '\n';
    out << "}\n" << memory.dump() << "\nexpert_kernels {\n";
    for (const auto& kernel : kernels) {
        out << "  bucket M<=" << kernel.max_tokens << " {\n"
            << kernel.w1_loop.dump() << kernel.w2_loop.dump()
            << "  llvm.w1 {\n" << kernel.llvm_w1 << "  }\n"
            << "  llvm.w2 {\n" << kernel.llvm_w2 << "  }\n  }\n";
    }
    out << "}\n";
    return out.str();
}

void MoeExecutablePlan::save(const std::filesystem::path& path) const {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write MoE executable plan: " + path.string());
    output << dump();
}

void write_moe_experiment_csv(const std::filesystem::path& path,
                              const MoeConfig& config, const MoeData& data,
                              int threads, int repetitions) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write MoE experiment CSV: " + path.string());
    output << "routing,strategy,scheduler,tokens,experts,top_k,tasks,simulated_imbalance,"
              "simulated_utilization,p50_ms,p95_ms,dispatch_ms,expert_ms,combine_ms,"
              "worker_imbalance,max_error\n";
    for (const auto distribution : {RoutingDistribution::Uniform,
                                    RoutingDistribution::ModerateSkew,
                                    RoutingDistribution::HeavySkew}) {
        const auto routing = make_routing_trace(config, distribution, 29);
        for (const auto strategy : {MoeExecutionStrategy::IndependentExperts,
                                    MoeExecutionStrategy::Grouped,
                                    MoeExecutionStrategy::BucketedGrouped}) {
            for (const auto scheduler : {MoeTaskScheduling::FixedExperts,
                                         MoeTaskScheduling::WorkStealing,
                                         MoeTaskScheduling::LoadAwareSplit}) {
                MoeExecutionSchedule schedule;
                schedule.strategy = strategy;
                schedule.task_scheduling = scheduler;
                schedule.threads = threads;
                schedule.split_threshold = std::max(4, config.tokens / std::max(1, threads));
                const auto plan = MoeCompiler{}.compile(config, schedule);
                const auto simulation = simulate_moe(config, routing, plan.schedule);
                const auto measured = execute_moe(plan, data, routing, 1, repetitions);
                output << routing_distribution_name(distribution) << ','
                       << moe_execution_strategy_name(strategy) << ','
                       << moe_task_scheduling_name(scheduler) << ','
                       << config.tokens << ',' << config.experts << ',' << config.top_k << ','
                       << simulation.tasks << ',' << simulation.imbalance << ','
                       << simulation.worker_utilization << ',' << measured.p50_milliseconds << ','
                       << measured.p95_milliseconds << ',' << measured.dispatch_milliseconds << ','
                       << measured.expert_milliseconds << ',' << measured.combine_milliseconds << ','
                       << measured.worker_imbalance << ',' << measured.max_error << '\n';
            }
        }
    }
}

}  // namespace schedforge
