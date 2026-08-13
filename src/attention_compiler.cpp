#include "schedforge/attention_compiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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

void validate_config(const AttentionConfig& config) {
    if (config.batch <= 0 || config.query_heads <= 0 || config.kv_heads <= 0 ||
        config.sequence_query <= 0 || config.sequence_kv <= 0 || config.head_dim <= 0 ||
        config.head_dim_value <= 0 || config.query_heads % config.kv_heads != 0 ||
        (config.causal && config.sequence_query > config.sequence_kv))
        throw std::invalid_argument("invalid attention configuration");
}

void validate_data(const AttentionConfig& config, const AttentionData& data) {
    const std::size_t query_values = static_cast<std::size_t>(config.batch) *
        config.query_heads * config.sequence_query * config.head_dim;
    const std::size_t key_values = static_cast<std::size_t>(config.batch) *
        config.kv_heads * config.sequence_kv * config.head_dim;
    const std::size_t value_values = static_cast<std::size_t>(config.batch) *
        config.kv_heads * config.sequence_kv * config.head_dim_value;
    if (data.query.size() != query_values || data.key.size() != key_values ||
        data.value.size() != value_values)
        throw std::invalid_argument("attention data shape mismatch");
}

float attention_scale(const AttentionConfig& config) {
    return config.scale > 0.0F ? config.scale :
        1.0F / std::sqrt(static_cast<float>(config.head_dim));
}

std::size_t q_index(const AttentionConfig& config, int batch, int head, int token, int dim) {
    return (((static_cast<std::size_t>(batch) * config.query_heads + head) *
             config.sequence_query + token) * config.head_dim + dim);
}

std::size_t k_index(const AttentionConfig& config, int batch, int head, int token, int dim) {
    return (((static_cast<std::size_t>(batch) * config.kv_heads + head) *
             config.sequence_kv + token) * config.head_dim + dim);
}

std::size_t v_index(const AttentionConfig& config, int batch, int head, int token, int dim) {
    return (((static_cast<std::size_t>(batch) * config.kv_heads + head) *
             config.sequence_kv + token) * config.head_dim_value + dim);
}

std::size_t o_index(const AttentionConfig& config, int batch, int head, int token, int dim) {
    return (((static_cast<std::size_t>(batch) * config.query_heads + head) *
             config.sequence_query + token) * config.head_dim_value + dim);
}

bool allowed(const AttentionConfig& config, int query_token, int key_token) {
    if (!config.causal) return true;
    const int absolute_query = config.sequence_kv - config.sequence_query + query_token;
    return key_token <= absolute_query;
}

float dot_product(const float* lhs, const float* rhs, int size) {
    float result = 0.0F;
    int index = 0;
#if defined(__AVX2__)
    __m256 accumulator = _mm256_setzero_ps();
    for (; index + 8 <= size; index += 8) {
        const __m256 a = _mm256_loadu_ps(lhs + index);
        const __m256 b = _mm256_loadu_ps(rhs + index);
#if defined(__FMA__)
        accumulator = _mm256_fmadd_ps(a, b, accumulator);
#else
        accumulator = _mm256_add_ps(accumulator, _mm256_mul_ps(a, b));
#endif
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, accumulator);
    for (float lane : lanes) result += lane;
#endif
    for (; index < size; ++index) result += lhs[index] * rhs[index];
    return result;
}

struct OnlineState {
    float maximum = -std::numeric_limits<float>::infinity();
    float denominator = 0.0F;
    std::vector<float> numerator;
};

void update_online(OnlineState& state, float score, const float* value, int value_dim) {
    const float new_maximum = std::max(state.maximum, score);
    const float old_scale = std::isfinite(state.maximum)
        ? std::exp(state.maximum - new_maximum) : 0.0F;
    const float probability = std::exp(score - new_maximum);
    state.denominator = state.denominator * old_scale + probability;
    for (int dim = 0; dim < value_dim; ++dim)
        state.numerator[static_cast<std::size_t>(dim)] =
            state.numerator[static_cast<std::size_t>(dim)] * old_scale + probability * value[dim];
    state.maximum = new_maximum;
}

void accumulate_value(float* output, const float* value, float weight, int value_dim) {
    int dim = 0;
#if defined(__AVX2__)
    const __m256 scale = _mm256_set1_ps(weight);
    for (; dim + 8 <= value_dim; dim += 8) {
        const __m256 current = _mm256_loadu_ps(output + dim);
        const __m256 input = _mm256_loadu_ps(value + dim);
#if defined(__FMA__)
        _mm256_storeu_ps(output + dim, _mm256_fmadd_ps(scale, input, current));
#else
        _mm256_storeu_ps(output + dim,
                         _mm256_add_ps(current, _mm256_mul_ps(scale, input)));
#endif
    }
#endif
    for (; dim < value_dim; ++dim) output[dim] += weight * value[dim];
}

OnlineState merge_online(const OnlineState& lhs, const OnlineState& rhs) {
    if (!std::isfinite(lhs.maximum)) return rhs;
    if (!std::isfinite(rhs.maximum)) return lhs;
    OnlineState merged;
    merged.maximum = std::max(lhs.maximum, rhs.maximum);
    const float lhs_scale = std::exp(lhs.maximum - merged.maximum);
    const float rhs_scale = std::exp(rhs.maximum - merged.maximum);
    merged.denominator = lhs.denominator * lhs_scale + rhs.denominator * rhs_scale;
    merged.numerator.resize(lhs.numerator.size());
    for (std::size_t index = 0; index < merged.numerator.size(); ++index)
        merged.numerator[index] = lhs.numerator[index] * lhs_scale +
                                  rhs.numerator[index] * rhs_scale;
    return merged;
}

OnlineState compute_range(const AttentionConfig& config, const AttentionData& data,
                          int batch, int query_head, int query_token,
                          int key_begin, int key_end, int key_tile) {
    const int kv_head = query_head * config.kv_heads / config.query_heads;
    const float* query = data.query.data() + q_index(config, batch, query_head, query_token, 0);
    OnlineState state;
    state.numerator.assign(static_cast<std::size_t>(config.head_dim_value), 0.0F);
    const float scale = attention_scale(config);
    for (int key_base = key_begin; key_base < key_end; key_base += std::max(1, key_tile)) {
        const int tile_end = std::min(key_base + std::max(1, key_tile), key_end);
        if (config.causal && key_base > config.sequence_kv - config.sequence_query + query_token)
            break;
        for (int key_token = key_base; key_token < tile_end; ++key_token) {
            if (!allowed(config, query_token, key_token)) continue;
            const float* key = data.key.data() + k_index(config, batch, kv_head, key_token, 0);
            const float* value = data.value.data() + v_index(config, batch, kv_head, key_token, 0);
            update_online(state, dot_product(query, key, config.head_dim) * scale,
                          value, config.head_dim_value);
        }
    }
    return state;
}

std::vector<float> execute_materialized(const AttentionConfig& config,
                                        const AttentionData& data, bool tiled,
                                        const AttentionSchedule& schedule) {
    const std::size_t rows = static_cast<std::size_t>(config.batch) *
                             config.query_heads * config.sequence_query;
    std::vector<float> scores(rows * config.sequence_kv,
                              -std::numeric_limits<float>::infinity());
    std::vector<float> probabilities(scores.size(), 0.0F);
    std::vector<float> output(rows * config.head_dim_value, 0.0F);
    const float scale = attention_scale(config);
    const int query_tile = tiled ? std::max(1, schedule.query_tile) : config.sequence_query;
    const int key_tile = tiled ? std::max(1, schedule.kv_tile) : config.sequence_kv;
    for (int batch = 0; batch < config.batch; ++batch) {
        for (int head = 0; head < config.query_heads; ++head) {
            const int kv_head = head * config.kv_heads / config.query_heads;
            for (int query_base = 0; query_base < config.sequence_query; query_base += query_tile) {
                for (int key_base = 0; key_base < config.sequence_kv; key_base += key_tile) {
                    for (int query_token = query_base;
                         query_token < std::min(query_base + query_tile, config.sequence_query);
                         ++query_token) {
                        const float* query = data.query.data() +
                            q_index(config, batch, head, query_token, 0);
                        const std::size_t row = (static_cast<std::size_t>(batch) *
                            config.query_heads + head) * config.sequence_query + query_token;
                        for (int key_token = key_base;
                             key_token < std::min(key_base + key_tile, config.sequence_kv);
                             ++key_token) {
                            if (!allowed(config, query_token, key_token)) continue;
                            const float* key = data.key.data() +
                                k_index(config, batch, kv_head, key_token, 0);
                            scores[row * config.sequence_kv + key_token] =
                                dot_product(query, key, config.head_dim) * scale;
                        }
                    }
                }
            }
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        const auto begin = scores.begin() + static_cast<std::ptrdiff_t>(row * config.sequence_kv);
        const float maximum = *std::max_element(begin, begin + config.sequence_kv);
        float denominator = 0.0F;
        for (int key_token = 0; key_token < config.sequence_kv; ++key_token) {
            const float probability = std::isfinite(scores[row * config.sequence_kv + key_token])
                ? std::exp(scores[row * config.sequence_kv + key_token] - maximum) : 0.0F;
            probabilities[row * config.sequence_kv + key_token] = probability;
            denominator += probability;
        }
        for (int key_token = 0; key_token < config.sequence_kv; ++key_token)
            probabilities[row * config.sequence_kv + key_token] /= denominator;
    }
    for (int batch = 0; batch < config.batch; ++batch)
        for (int head = 0; head < config.query_heads; ++head) {
            const int kv_head = head * config.kv_heads / config.query_heads;
            for (int query_token = 0; query_token < config.sequence_query; ++query_token) {
                const std::size_t row = (static_cast<std::size_t>(batch) *
                    config.query_heads + head) * config.sequence_query + query_token;
                for (int key_token = 0; key_token < config.sequence_kv; ++key_token) {
                    const float probability = probabilities[row * config.sequence_kv + key_token];
                    const float* value = data.value.data() +
                        v_index(config, batch, kv_head, key_token, 0);
                    for (int dim = 0; dim < config.head_dim_value; ++dim)
                        output[row * config.head_dim_value + dim] += probability * value[dim];
                }
            }
        }
    return output;
}

std::vector<float> execute_streaming(const AttentionExecutablePlan& plan,
                                     const AttentionData& data) {
    const auto& config = plan.config;
    std::vector<float> output(static_cast<std::size_t>(config.batch) * config.query_heads *
                              config.sequence_query * config.head_dim_value, 0.0F);
    if (plan.plan.strategy == AttentionLoweringStrategy::SplitKVDecode) {
        const int splits = std::max(1, plan.plan.split_kv);
        const int rows = config.batch * config.query_heads * config.sequence_query;
        std::vector<OnlineState> partials(static_cast<std::size_t>(rows) * splits);
        const int tasks = rows * splits;
        const int workers = std::clamp(plan.plan.schedule.threads, 1, std::max(1, tasks));
        std::atomic<int> next_task{0};
        auto worker = [&] {
            while (true) {
                const int task = next_task.fetch_add(1);
                if (task >= tasks) break;
                const int split = task % splits;
                const int row = task / splits;
                const int query_token = row % config.sequence_query;
                const int head_batch = row / config.sequence_query;
                const int head = head_batch % config.query_heads;
                const int batch = head_batch / config.query_heads;
                const int key_begin = config.sequence_kv * split / splits;
                const int key_end = config.sequence_kv * (split + 1) / splits;
                partials[static_cast<std::size_t>(row) * splits + split] = compute_range(
                    config, data, batch, head, query_token, key_begin, key_end,
                    plan.plan.schedule.kv_tile);
            }
        };
        std::vector<std::thread> threads;
        for (int index = 1; index < workers; ++index) threads.emplace_back(worker);
        worker();
        for (auto& thread : threads) thread.join();
        for (int row = 0; row < rows; ++row) {
            OnlineState state = partials[static_cast<std::size_t>(row) * splits];
            for (int split = 1; split < splits; ++split)
                state = merge_online(state, partials[static_cast<std::size_t>(row) * splits + split]);
            const int query_token = row % config.sequence_query;
            const int head_batch = row / config.sequence_query;
            const int head = head_batch % config.query_heads;
            const int batch = head_batch / config.query_heads;
            for (int dim = 0; dim < config.head_dim_value; ++dim)
                output[o_index(config, batch, head, query_token, dim)] =
                    state.numerator[static_cast<std::size_t>(dim)] / state.denominator;
        }
        return output;
    }
    const int query_blocks = (config.sequence_query + plan.plan.schedule.query_tile - 1) /
                             plan.plan.schedule.query_tile;
    const int tasks = config.batch * config.query_heads * query_blocks;
    const int workers = std::clamp(plan.plan.schedule.threads, 1, std::max(1, tasks));
    std::atomic<int> next_task{0};
    auto worker = [&] {
        while (true) {
            const int task = next_task.fetch_add(1);
            if (task >= tasks) break;
            const int query_block = task % query_blocks;
            const int head_batch = task / query_blocks;
            const int head = head_batch % config.query_heads;
            const int batch = head_batch / config.query_heads;
            const int query_begin = query_block * plan.plan.schedule.query_tile;
            const int query_end = std::min(query_begin + plan.plan.schedule.query_tile,
                                           config.sequence_query);
            const int active_queries = query_end - query_begin;
            std::vector<float> maximum(static_cast<std::size_t>(active_queries),
                                       -std::numeric_limits<float>::infinity());
            std::vector<float> denominator(static_cast<std::size_t>(active_queries), 0.0F);
            std::vector<float> numerator(static_cast<std::size_t>(active_queries) *
                                         config.head_dim_value, 0.0F);
            std::vector<float> scores(static_cast<std::size_t>(active_queries) *
                                      plan.plan.schedule.kv_tile);
            const int kv_head = head * config.kv_heads / config.query_heads;
            const float scale = attention_scale(config);
            for (int key_base = 0; key_base < config.sequence_kv;
                 key_base += plan.plan.schedule.kv_tile) {
                if (config.causal && key_base > config.sequence_kv - config.sequence_query +
                                               query_end - 1)
                    break;
                const int key_end = std::min(key_base + plan.plan.schedule.kv_tile,
                                             config.sequence_kv);
                const int active_keys = key_end - key_base;
                for (int local_query = 0; local_query < active_queries; ++local_query) {
                    const int query_token = query_begin + local_query;
                    const float* query = data.query.data() +
                        q_index(config, batch, head, query_token, 0);
                    float block_maximum = -std::numeric_limits<float>::infinity();
                    for (int local_key = 0; local_key < active_keys; ++local_key) {
                        const int key_token = key_base + local_key;
                        float score = -std::numeric_limits<float>::infinity();
                        if (allowed(config, query_token, key_token)) {
                            const float* key = data.key.data() +
                                k_index(config, batch, kv_head, key_token, 0);
                            score = dot_product(query, key, config.head_dim) * scale;
                        }
                        scores[static_cast<std::size_t>(local_query) *
                               plan.plan.schedule.kv_tile + local_key] = score;
                        block_maximum = std::max(block_maximum, score);
                    }
                    if (!std::isfinite(block_maximum)) continue;
                    const float new_maximum = std::max(
                        maximum[static_cast<std::size_t>(local_query)], block_maximum);
                    const float old_scale = std::isfinite(
                        maximum[static_cast<std::size_t>(local_query)])
                        ? std::exp(maximum[static_cast<std::size_t>(local_query)] - new_maximum)
                        : 0.0F;
                    denominator[static_cast<std::size_t>(local_query)] *= old_scale;
                    float* row_output = numerator.data() +
                        static_cast<std::size_t>(local_query) * config.head_dim_value;
                    for (int dim = 0; dim < config.head_dim_value; ++dim)
                        row_output[dim] *= old_scale;
                    for (int local_key = 0; local_key < active_keys; ++local_key) {
                        const float score = scores[static_cast<std::size_t>(local_query) *
                            plan.plan.schedule.kv_tile + local_key];
                        if (!std::isfinite(score)) continue;
                        const float probability = std::exp(score - new_maximum);
                        denominator[static_cast<std::size_t>(local_query)] += probability;
                        const float* value = data.value.data() + v_index(
                            config, batch, kv_head, key_base + local_key, 0);
                        accumulate_value(row_output, value, probability,
                                         config.head_dim_value);
                    }
                    maximum[static_cast<std::size_t>(local_query)] = new_maximum;
                }
            }
            for (int local_query = 0; local_query < active_queries; ++local_query) {
                const int query_token = query_begin + local_query;
                for (int dim = 0; dim < config.head_dim_value; ++dim)
                    output[o_index(config, batch, head, query_token, dim)] = numerator[
                        static_cast<std::size_t>(local_query) * config.head_dim_value + dim] /
                        denominator[static_cast<std::size_t>(local_query)];
            }
        }
    };
    std::vector<std::thread> threads;
    for (int index = 1; index < workers; ++index) threads.emplace_back(worker);
    worker();
    for (auto& thread : threads) thread.join();
    return output;
}

AttentionBenchmarkResult benchmark(const AttentionExecutablePlan& plan,
                                   const AttentionData& data, int warmup, int repetitions) {
    auto run = [&] {
        if (plan.plan.strategy == AttentionLoweringStrategy::Materialized)
            return execute_materialized(plan.config, data, false, plan.plan.schedule);
        if (plan.plan.strategy == AttentionLoweringStrategy::TiledMaterialized)
            return execute_materialized(plan.config, data, true, plan.plan.schedule);
        return execute_streaming(plan, data);
    };
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) (void)run();
    std::vector<std::pair<double, std::vector<float>>> measured;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        auto output = run();
        const auto end = std::chrono::steady_clock::now();
        measured.emplace_back(std::chrono::duration<double, std::milli>(end - begin).count(),
                              std::move(output));
    }
    std::sort(measured.begin(), measured.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    const std::size_t p50_index = measured.size() / 2;
    const std::size_t p95_index = std::min(measured.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(measured.size()))) - 1);
    AttentionBenchmarkResult result;
    result.milliseconds = measured[p50_index].first;
    result.p50_milliseconds = measured[p50_index].first;
    result.p95_milliseconds = measured[p95_index].first;
    result.output = std::move(measured[p50_index].second);
    result.max_error = max_abs_error(reference_attention(plan.config, data), result.output);
    return result;
}

GraphTensorType tensor_type(std::vector<Dimension> shape) {
    return {std::move(shape), DataType::F32, {}, std::nullopt, 0, -1};
}

}  // namespace

std::string attention_strategy_name(AttentionLoweringStrategy strategy) {
    switch (strategy) {
        case AttentionLoweringStrategy::Materialized: return "materialized";
        case AttentionLoweringStrategy::TiledMaterialized: return "tiled-materialized";
        case AttentionLoweringStrategy::IOAware: return "io-aware";
        case AttentionLoweringStrategy::AutoScheduledIOAware: return "auto-io-aware";
        case AttentionLoweringStrategy::SplitKVDecode: return "split-kv-decode";
    }
    return "unknown";
}

std::string AttentionSchedule::dump() const {
    std::ostringstream out;
    out << "attention.schedule {\n  qtile " << query_tile << "\n  kvtile " << kv_tile
        << "\n  qk_micro " << qk_micro_rows << 'x' << qk_micro_columns
        << "\n  pv_micro " << pv_micro_rows << 'x' << pv_micro_columns
        << "\n  softmax_vector " << softmax_vector_width
        << "\n  pack_k " << std::boolalpha << pack_k << "\n  pack_v " << pack_v
        << "\n  prefetch_k " << prefetch_k << "\n  prefetch_v " << prefetch_v
        << "\n  threads " << threads << "\n}\n";
    return out.str();
}

std::string TilePipelineOperation::dump() const {
    return "  " + result + " = " + name + " " + inputs + '\n';
}

std::string TilePipelineIR::dump() const {
    std::ostringstream out;
    out << "tile.pipeline @" << name << " {\n";
    for (const auto& operation : operations) out << operation.dump();
    out << "  tile.yield %output\n}\n";
    return out.str();
}

std::string AttentionMemoryPlan::dump() const {
    std::ostringstream out;
    out << "attention.memory temporary=" << temporary_bytes << " score=" << score_bytes
        << " probability=" << probability_bytes << " online_state=" << online_state_bytes;
    return out.str();
}

std::string AttentionSimulationResult::dump() const {
    std::ostringstream out;
    out << "attention.simulation flops=" << flops << " bytes_read=" << bytes_read
        << " bytes_written=" << bytes_written << " arithmetic_intensity="
        << arithmetic_intensity << " temporary=" << temporary_bytes << " qk_tiles="
        << qk_tiles << " skipped_causal_tiles=" << skipped_causal_tiles
        << " score_evaluations=" << score_evaluations
        << " exp_evaluations=" << exp_evaluations
        << " reduction_operations=" << reduction_operations
        << " l2_traffic=" << estimated_l2_traffic << " llc_traffic=" << estimated_llc_traffic
        << " l2_hit_rate=" << estimated_l2_hit_rate
        << " llc_hit_rate=" << estimated_llc_hit_rate;
    return out.str();
}

TensorGraph build_sdpa_graph(const AttentionConfig& config, bool dynamic_sequence) {
    validate_config(config);
    TensorGraph graph;
    const Dimension sq = dynamic_sequence ? Dimension::dynamic("Sq") :
                                           Dimension::fixed(config.sequence_query);
    const Dimension sk = dynamic_sequence ? Dimension::dynamic("Sk") :
                                           Dimension::fixed(config.sequence_kv);
    const auto query = graph.addInput("query", tensor_type({Dimension::fixed(config.batch),
        Dimension::fixed(config.query_heads), sq, Dimension::fixed(config.head_dim)}));
    const auto key = graph.addInput("key", tensor_type({Dimension::fixed(config.batch),
        Dimension::fixed(config.kv_heads), sk, Dimension::fixed(config.head_dim)}));
    const auto value = graph.addInput("value", tensor_type({Dimension::fixed(config.batch),
        Dimension::fixed(config.kv_heads), sk, Dimension::fixed(config.head_dim_value)}));
    const auto scale = graph.addInput("scale", tensor_type({Dimension::fixed(1)}), true);
    const auto transposed = graph.addOperation(GraphOpKind::Transpose, "transpose_k", {key},
        tensor_type({Dimension::fixed(config.batch), Dimension::fixed(config.kv_heads),
                     Dimension::fixed(config.head_dim), sk}));
    const auto scores = graph.addOperation(GraphOpKind::MatMul, "qk", {query, transposed},
        tensor_type({Dimension::fixed(config.batch), Dimension::fixed(config.query_heads), sq, sk}));
    const auto scaled = graph.addOperation(GraphOpKind::Multiply, "scale", {scores, scale},
        tensor_type({Dimension::fixed(config.batch), Dimension::fixed(config.query_heads), sq, sk}));
    const auto masked = graph.addOperation(GraphOpKind::Mask, "causal_mask", {scaled},
        tensor_type({Dimension::fixed(config.batch), Dimension::fixed(config.query_heads), sq, sk}));
    graph.operations().back().attributes["causal"] = config.causal ? "true" : "false";
    const auto probabilities = graph.addOperation(GraphOpKind::Softmax, "softmax", {masked},
        tensor_type({Dimension::fixed(config.batch), Dimension::fixed(config.query_heads), sq, sk}));
    const auto output = graph.addOperation(GraphOpKind::MatMul, "pv", {probabilities, value},
        tensor_type({Dimension::fixed(config.batch), Dimension::fixed(config.query_heads), sq,
                     Dimension::fixed(config.head_dim_value)}));
    graph.setReturn(output);
    return graph;
}

TensorGraph AttentionFusionPass::run(const TensorGraph& graph) const {
    if (graph.returnValue() < 0) return graph;
    const auto& values = graph.values();
    const auto& operations = graph.operations();
    auto producer = [&](int value) -> const GraphOperation* {
        if (value < 0 || static_cast<std::size_t>(value) >= values.size()) return nullptr;
        const int operation = values[static_cast<std::size_t>(value)].producer;
        if (operation < 0 || static_cast<std::size_t>(operation) >= operations.size()) return nullptr;
        return &operations[static_cast<std::size_t>(operation)];
    };
    const auto* pv = producer(graph.returnValue());
    if (!pv || pv->kind != GraphOpKind::MatMul || pv->inputs.size() != 2) return graph;
    const auto* softmax = producer(pv->inputs[0]);
    if (!softmax || softmax->kind != GraphOpKind::Softmax || softmax->inputs.size() != 1)
        return graph;
    const auto* mask = producer(softmax->inputs[0]);
    if (!mask || mask->kind != GraphOpKind::Mask || mask->inputs.size() != 1) return graph;
    const auto* scale = producer(mask->inputs[0]);
    if (!scale || scale->kind != GraphOpKind::Multiply || scale->inputs.empty()) return graph;
    const auto* qk = producer(scale->inputs[0]);
    if ((!qk || qk->kind != GraphOpKind::MatMul) && scale->inputs.size() > 1)
        qk = producer(scale->inputs[1]);
    if (!qk || qk->kind != GraphOpKind::MatMul || qk->inputs.size() != 2) return graph;
    const auto* transpose = producer(qk->inputs[1]);
    if (!transpose || transpose->kind != GraphOpKind::Transpose || transpose->inputs.size() != 1)
        return graph;
    TensorGraph fused;
    const std::vector<int> original_inputs = {qk->inputs[0], transpose->inputs[0], pv->inputs[1]};
    std::vector<int> inputs;
    for (const int input : original_inputs) {
        const auto& value = values.at(static_cast<std::size_t>(input));
        inputs.push_back(fused.addInput(value.name, value.type, value.constant));
    }
    const auto output_type = graph.values().at(static_cast<std::size_t>(graph.returnValue())).type;
    const auto result = fused.addOperation(GraphOpKind::AttentionSdpa, "sdpa", inputs, output_type);
    fused.operations().back().attributes["semantic"] = "scaled_dot_product_attention";
    const auto causal = mask->attributes.find("causal");
    if (causal != mask->attributes.end()) fused.operations().back().attributes["causal"] = causal->second;
    fused.setReturn(result);
    return fused;
}

AttentionData make_attention_data(const AttentionConfig& config, std::uint32_t seed) {
    validate_config(config);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    AttentionData data;
    data.query.resize(static_cast<std::size_t>(config.batch) * config.query_heads *
                      config.sequence_query * config.head_dim);
    data.key.resize(static_cast<std::size_t>(config.batch) * config.kv_heads *
                    config.sequence_kv * config.head_dim);
    data.value.resize(static_cast<std::size_t>(config.batch) * config.kv_heads *
                      config.sequence_kv * config.head_dim_value);
    for (auto* values : {&data.query, &data.key, &data.value})
        std::generate(values->begin(), values->end(), [&] { return distribution(generator); });
    return data;
}

AttentionPlanOptions select_attention_plan(const AttentionConfig& config,
                                           const TargetInfo& target, int max_threads) {
    validate_config(config);
    AttentionPlanOptions options;
    options.schedule.threads = std::clamp(max_threads, 1, target.logical_cpus);
    if (config.sequence_query <= 2) {
        options.strategy = AttentionLoweringStrategy::SplitKVDecode;
        options.schedule.query_tile = 1;
        options.schedule.kv_tile = config.sequence_kv >= 1024 ? 128 : 64;
        options.split_kv = options.schedule.threads > 1
            ? std::clamp(std::max(2, config.sequence_kv / 128), 2, options.schedule.threads)
            : 1;
        options.schedule.parallel_axis = AttentionParallelAxis::SplitKV;
    } else if (config.sequence_query <= 16 && config.sequence_kv <= 64) {
        options.strategy = AttentionLoweringStrategy::Materialized;
        options.schedule.query_tile = config.sequence_query;
        options.schedule.kv_tile = config.sequence_kv;
    } else {
        options.strategy = AttentionLoweringStrategy::AutoScheduledIOAware;
        const std::size_t bytes_per_pair = static_cast<std::size_t>(config.head_dim * 2 +
            config.head_dim_value + 1) * sizeof(float);
        options.schedule.query_tile = config.head_dim >= 128 ? 16 : 32;
        options.schedule.kv_tile = static_cast<int>(std::clamp<std::size_t>(
            target.l2_bytes / std::max<std::size_t>(1, bytes_per_pair *
                static_cast<std::size_t>(options.schedule.query_tile)), 16, 128));
        options.schedule.kv_tile = std::max(16, options.schedule.kv_tile / 16 * 16);
    }
    return options;
}

AttentionPlanOptions tune_attention_plan(const AttentionConfig& config,
                                         const AttentionData& data,
                                         const TargetInfo& target,
                                         int max_threads,
                                         int warmup,
                                         int repetitions) {
    auto seed = select_attention_plan(config, target, max_threads);
    if (seed.strategy == AttentionLoweringStrategy::Materialized ||
        seed.strategy == AttentionLoweringStrategy::SplitKVDecode)
        return seed;
    struct CandidateMeasurement {
        AttentionPlanOptions options;
        double latency = 0.0;
    };
    std::vector<CandidateMeasurement> candidates;
    const int screening_repetitions = std::max(5, repetitions * 2);
    for (const int query_tile : {16, 32, 64}) {
        if (query_tile > config.sequence_query) continue;
        for (const int kv_tile : {32, 64, 128}) {
            if (kv_tile > config.sequence_kv) continue;
            AttentionPlanOptions candidate = seed;
            candidate.strategy = AttentionLoweringStrategy::AutoScheduledIOAware;
            candidate.schedule.query_tile = query_tile;
            candidate.schedule.kv_tile = kv_tile;
            const auto simulation = simulate_attention(config, candidate, target);
            if (simulation.temporary_bytes > target.l2_bytes) continue;
            const auto executable = AttentionCompiler{target}.compile(config, candidate);
            const auto measured = execute_attention(executable, data, warmup,
                                                    screening_repetitions);
            candidates.push_back({candidate, measured.p50_milliseconds});
        }
    }
    if (candidates.empty()) return seed;
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.latency < rhs.latency;
    });
    candidates.resize(std::min<std::size_t>(3, candidates.size()));
    AttentionPlanOptions best = candidates.front().options;
    double best_latency = std::numeric_limits<double>::infinity();
    const int finalist_repetitions = std::max(7, repetitions * 4);
    for (const auto& candidate : candidates) {
        const auto executable = AttentionCompiler{target}.compile(config, candidate.options);
        const auto measured = execute_attention(executable, data, std::max(1, warmup),
                                                finalist_repetitions);
        if (measured.p50_milliseconds < best_latency) {
            best_latency = measured.p50_milliseconds;
            best = candidate.options;
        }
    }
    return best;
}

AttentionSimulationResult simulate_attention(const AttentionConfig& config,
                                              const AttentionPlanOptions& plan,
                                              const TargetInfo& target) {
    validate_config(config);
    double scores_per_head = static_cast<double>(config.sequence_query) * config.sequence_kv;
    if (config.causal) {
        scores_per_head = 0.0;
        for (int query_token = 0; query_token < config.sequence_query; ++query_token)
            scores_per_head += std::clamp(config.sequence_kv - config.sequence_query +
                                          query_token + 1, 0, config.sequence_kv);
    }
    const double score_count = static_cast<double>(config.batch) * config.query_heads *
                               scores_per_head;
    AttentionSimulationResult result;
    result.flops = 2.0 * score_count * (config.head_dim + config.head_dim_value);
    const std::size_t materialized_score_bytes = static_cast<std::size_t>(config.batch) *
        config.query_heads * config.sequence_query * config.sequence_kv * sizeof(float);
    const std::size_t query_bytes = static_cast<std::size_t>(config.batch) * config.query_heads *
                                    config.sequence_query * config.head_dim * sizeof(float);
    const std::size_t kv_bytes = static_cast<std::size_t>(config.batch) * config.kv_heads *
        config.sequence_kv * (config.head_dim + config.head_dim_value) * sizeof(float);
    const std::size_t input_bytes = query_bytes + kv_bytes;
    const std::size_t output_bytes = static_cast<std::size_t>(config.batch) * config.query_heads *
                                     config.sequence_query * config.head_dim_value * sizeof(float);
    if (plan.strategy == AttentionLoweringStrategy::Materialized ||
        plan.strategy == AttentionLoweringStrategy::TiledMaterialized) {
        result.bytes_read = static_cast<double>(input_bytes + 2 * materialized_score_bytes);
        result.bytes_written = static_cast<double>(output_bytes + 2 * materialized_score_bytes);
        result.temporary_bytes = 2 * materialized_score_bytes;
    } else {
        const int query_blocks = (config.sequence_query + plan.schedule.query_tile - 1) /
                                 plan.schedule.query_tile;
        const int parallel_tasks = config.batch * config.query_heads * query_blocks;
        const int workers = std::clamp(plan.schedule.threads, 1, std::max(1, parallel_tasks));
        const std::size_t tile_state = plan.strategy == AttentionLoweringStrategy::SplitKVDecode
            ? static_cast<std::size_t>(config.batch) * config.query_heads *
                  config.sequence_query * std::max(1, plan.split_kv) *
                  (config.head_dim_value + 2) * sizeof(float)
            : static_cast<std::size_t>(workers) * plan.schedule.query_tile *
                  (plan.schedule.kv_tile + config.head_dim_value + 2) * sizeof(float);
        const int query_tiles = (config.sequence_query + plan.schedule.query_tile - 1) /
                                plan.schedule.query_tile;
        const int llc_kv_passes = kv_bytes <= target.l3_bytes ? 1 : query_tiles;
        result.bytes_read = static_cast<double>(query_bytes) +
                            static_cast<double>(kv_bytes) * llc_kv_passes;
        result.bytes_written = static_cast<double>(output_bytes);
        result.temporary_bytes = tile_state;
    }
    const int query_tiles = (config.sequence_query + plan.schedule.query_tile - 1) /
                            plan.schedule.query_tile;
    const int key_tiles = (config.sequence_kv + plan.schedule.kv_tile - 1) /
                          plan.schedule.kv_tile;
    result.qk_tiles = static_cast<std::size_t>(config.batch) * config.query_heads *
                      query_tiles * key_tiles;
    if (config.causal) {
        for (int query_base = 0; query_base < config.sequence_query;
             query_base += plan.schedule.query_tile) {
            const int absolute_query_end = config.sequence_kv - config.sequence_query +
                std::min(query_base + plan.schedule.query_tile, config.sequence_query) - 1;
            for (int key_base = 0; key_base < config.sequence_kv;
                 key_base += plan.schedule.kv_tile)
                if (key_base > absolute_query_end)
                    result.skipped_causal_tiles += static_cast<std::size_t>(config.batch) *
                                                   config.query_heads;
        }
    }
    result.qk_tiles -= std::min(result.qk_tiles, result.skipped_causal_tiles);
    result.score_evaluations = score_count;
    result.exp_evaluations = score_count + static_cast<double>(config.batch) *
        config.query_heads * config.sequence_query * key_tiles;
    result.reduction_operations = 2.0 * score_count;
    const double total_bytes = result.bytes_read + result.bytes_written;
    result.arithmetic_intensity = total_bytes > 0.0 ? result.flops / total_bytes : 0.0;
    if (plan.strategy == AttentionLoweringStrategy::Materialized ||
        plan.strategy == AttentionLoweringStrategy::TiledMaterialized) {
        result.estimated_l2_traffic = static_cast<double>(input_bytes + output_bytes) +
                                      6.0 * static_cast<double>(materialized_score_bytes);
        result.estimated_llc_traffic = static_cast<double>(input_bytes + output_bytes) +
                                       4.0 * static_cast<double>(materialized_score_bytes);
    } else {
        result.estimated_l2_traffic = static_cast<double>(query_bytes + output_bytes) +
            static_cast<double>(kv_bytes) * query_tiles;
        result.estimated_llc_traffic = total_bytes;
    }
    result.estimated_l2_hit_rate = result.estimated_l2_traffic > 0.0
        ? std::clamp(1.0 - result.estimated_llc_traffic / result.estimated_l2_traffic,
                     0.0, 1.0) : 1.0;
    result.estimated_llc_hit_rate = result.estimated_llc_traffic > 0.0
        ? std::clamp(1.0 - total_bytes / result.estimated_llc_traffic, 0.0, 1.0) : 1.0;
    return result;
}

std::vector<float> reference_attention(const AttentionConfig& config,
                                       const AttentionData& data) {
    validate_config(config);
    std::vector<float> output(static_cast<std::size_t>(config.batch) * config.query_heads *
                              config.sequence_query * config.head_dim_value, 0.0F);
    const float scale = attention_scale(config);
    for (int batch = 0; batch < config.batch; ++batch)
        for (int head = 0; head < config.query_heads; ++head) {
            const int kv_head = head * config.kv_heads / config.query_heads;
            for (int query_token = 0; query_token < config.sequence_query; ++query_token) {
                const float* query = data.query.data() +
                    q_index(config, batch, head, query_token, 0);
                float maximum = -std::numeric_limits<float>::infinity();
                for (int key_token = 0; key_token < config.sequence_kv; ++key_token) {
                    if (!allowed(config, query_token, key_token)) continue;
                    const float* key = data.key.data() +
                        k_index(config, batch, kv_head, key_token, 0);
                    maximum = std::max(maximum,
                        dot_product(query, key, config.head_dim) * scale);
                }
                float denominator = 0.0F;
                for (int key_token = 0; key_token < config.sequence_kv; ++key_token) {
                    if (!allowed(config, query_token, key_token)) continue;
                    const float* key = data.key.data() +
                        k_index(config, batch, kv_head, key_token, 0);
                    const float probability = std::exp(
                        dot_product(query, key, config.head_dim) * scale - maximum);
                    denominator += probability;
                    const float* value = data.value.data() +
                        v_index(config, batch, kv_head, key_token, 0);
                    accumulate_value(output.data() +
                        o_index(config, batch, head, query_token, 0), value, probability,
                        config.head_dim_value);
                }
                for (int dim = 0; dim < config.head_dim_value; ++dim)
                    output[o_index(config, batch, head, query_token, dim)] /= denominator;
            }
        }
    return output;
}

AttentionCompiler::AttentionCompiler(TargetInfo target) : target_(std::move(target)) {}

AttentionExecutablePlan AttentionCompiler::compile(const AttentionConfig& config,
                                                    AttentionPlanOptions options) const {
    validate_config(config);
    options.schedule.threads = std::clamp(options.schedule.threads, 1, target_.logical_cpus);
    options.schedule.query_tile = std::clamp(options.schedule.query_tile, 1, config.sequence_query);
    options.schedule.kv_tile = std::clamp(options.schedule.kv_tile, 1, config.sequence_kv);
    AttentionExecutablePlan executable;
    executable.config = config;
    executable.plan = options;
    executable.tensor_graph = AttentionFusionPass{}.run(build_sdpa_graph(config, true));
    executable.pipeline.operations = {
        {"attention.qk", "%Q_tile, %K_tile", "%scores"},
        {"attention.scale", "%scores", "%scaled"},
        {"attention.causal_mask", "%scaled", "%masked"},
        {"reduce.max", "%masked", "%row_max"},
        {"vector.max", "%running_max, %row_max", "%new_max"},
        {"vector.exp", "%masked - %new_max", "%probability"},
        {"reduce.sum", "%probability", "%block_sum"},
        {"attention.rescale", "%running_state, %new_max", "%rescaled"},
        {"attention.online_softmax", "%probability, %block_sum", "%online"},
        {"attention.pv", "%probability, %V_tile", "%output"},
        {"vector.div", "%output, %denominator", "%normalized"}
    };
    const std::size_t score_bytes = static_cast<std::size_t>(config.batch) * config.query_heads *
        config.sequence_query * config.sequence_kv * sizeof(float);
    executable.memory.score_bytes = (options.strategy == AttentionLoweringStrategy::Materialized ||
        options.strategy == AttentionLoweringStrategy::TiledMaterialized) ? score_bytes : 0;
    executable.memory.probability_bytes = executable.memory.score_bytes;
    executable.simulation = simulate_attention(config, options, target_);
    executable.memory.online_state_bytes =
        (options.strategy == AttentionLoweringStrategy::Materialized ||
         options.strategy == AttentionLoweringStrategy::TiledMaterialized)
        ? 0 : executable.simulation.temporary_bytes;
    executable.memory.temporary_bytes = executable.memory.score_bytes +
        executable.memory.probability_bytes + executable.memory.online_state_bytes;
    Schedule qk_schedule;
    qk_schedule.threads = 1;
    qk_schedule.vector_width = target_.vector_width;
    qk_schedule.mr = options.schedule.qk_micro_rows;
    qk_schedule.nr = options.schedule.qk_micro_columns;
    qk_schedule.bm = options.schedule.query_tile;
    qk_schedule.bn = options.schedule.kv_tile;
    qk_schedule.bk = std::min(64, config.head_dim);
    qk_schedule.pack_b = options.schedule.pack_k;
    qk_schedule.prefetch_distance = options.schedule.prefetch_k;
    Schedule pv_schedule = qk_schedule;
    pv_schedule.mr = options.schedule.pv_micro_rows;
    pv_schedule.nr = options.schedule.pv_micro_columns;
    pv_schedule.bn = config.head_dim_value;
    pv_schedule.bk = options.schedule.kv_tile;
    pv_schedule.pack_b = options.schedule.pack_v;
    pv_schedule.prefetch_distance = options.schedule.prefetch_v;
    executable.qk_loop = apply_schedule({options.schedule.query_tile,
        options.schedule.kv_tile, config.head_dim, false, false}, qk_schedule);
    executable.pv_loop = apply_schedule({options.schedule.query_tile,
        config.head_dim_value, options.schedule.kv_tile, false, false}, pv_schedule);
    const auto qk_compilation = LLVMJITBackend{}.benchmark(executable.qk_loop,
        make_data(executable.qk_loop.problem, 73), 0, 1);
    const auto pv_compilation = LLVMJITBackend{}.benchmark(executable.pv_loop,
        make_data(executable.pv_loop.problem, 79), 0, 1);
    executable.llvm_qk = qk_compilation.llvm_ir;
    executable.llvm_pv = pv_compilation.llvm_ir;
    executable.hardware = target_.str();
    executable.guards = {"0 < Sq", "Sq <= " + std::to_string(config.sequence_query),
        "0 < Sk", "Sk <= " + std::to_string(config.sequence_kv),
        "Hq % Hkv == 0"};
    return executable;
}

std::string AttentionExecutablePlan::dump() const {
    std::ostringstream out;
    out << "sfe.attention_executable version=0.5.0 strategy="
        << attention_strategy_name(plan.strategy) << '\n' << "hardware=" << hardware << '\n'
        << "config B=" << config.batch << " Hq=" << config.query_heads << " Hkv="
        << config.kv_heads << " Sq<=" << config.sequence_query << " Sk<="
        << config.sequence_kv << " D=" << config.head_dim << " Dv="
        << config.head_dim_value << " causal=" << std::boolalpha << config.causal << '\n'
        << tensor_graph.dump() << plan.schedule.dump() << pipeline.dump() << "guards {\n";
    for (const auto& guard : guards) out << "  " << guard << '\n';
    out << "}\n" << memory.dump() << '\n' << simulation.dump() << "\nqk_loop {\n"
        << qk_loop.dump() << "}\npv_loop {\n" << pv_loop.dump() << "}\nllvm.qk {\n"
        << llvm_qk << "}\nllvm.pv {\n" << llvm_pv << "}\n";
    return out.str();
}

void AttentionExecutablePlan::save(const std::filesystem::path& path) const {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write attention executable: " + path.string());
    output << dump();
}

AttentionBenchmarkResult execute_attention(const AttentionExecutablePlan& plan,
                                           const AttentionData& data,
                                           int warmup, int repetitions) {
    validate_data(plan.config, data);
    return benchmark(plan, data, warmup, repetitions);
}

AttentionBenchmarkResult execute_attention(const AttentionExecutablePlan& plan,
                                           const AttentionData& data,
                                           int runtime_sequence_query,
                                           int runtime_sequence_kv,
                                           int warmup, int repetitions) {
    if (runtime_sequence_query <= 0 || runtime_sequence_kv <= 0 ||
        runtime_sequence_query > plan.config.sequence_query ||
        runtime_sequence_kv > plan.config.sequence_kv)
        throw std::invalid_argument("runtime attention shape violates compiled guard");
    AttentionExecutablePlan specialized = plan;
    specialized.config.sequence_query = runtime_sequence_query;
    specialized.config.sequence_kv = runtime_sequence_kv;
    specialized.plan.schedule.query_tile = std::min(
        specialized.plan.schedule.query_tile, runtime_sequence_query);
    specialized.plan.schedule.kv_tile = std::min(
        specialized.plan.schedule.kv_tile, runtime_sequence_kv);
    const std::size_t expected_query = static_cast<std::size_t>(specialized.config.batch) *
        specialized.config.query_heads * runtime_sequence_query * specialized.config.head_dim;
    const std::size_t expected_key = static_cast<std::size_t>(specialized.config.batch) *
        specialized.config.kv_heads * runtime_sequence_kv * specialized.config.head_dim;
    const std::size_t expected_value = static_cast<std::size_t>(specialized.config.batch) *
        specialized.config.kv_heads * runtime_sequence_kv * specialized.config.head_dim_value;
    if (data.query.size() != expected_query || data.key.size() != expected_key ||
        data.value.size() != expected_value)
        throw std::invalid_argument("runtime attention data shape mismatch");
    return benchmark(specialized, data, warmup, repetitions);
}

std::string KVCache::dump() const {
    std::ostringstream out;
    out << "!sfg.kv_cache<batch=" << batch << ", heads=" << kv_heads << ", length="
        << length << "/" << capacity << ", key_dim=" << head_dim << ", value_dim="
        << head_dim_value << ">";
    return out.str();
}

KVCache make_kv_cache(const AttentionConfig& config, int capacity) {
    validate_config(config);
    KVCache cache;
    cache.batch = config.batch;
    cache.kv_heads = config.kv_heads;
    cache.head_dim = config.head_dim;
    cache.head_dim_value = config.head_dim_value;
    cache.capacity = std::max(config.sequence_kv, capacity);
    cache.keys.resize(static_cast<std::size_t>(cache.batch) * cache.kv_heads *
                      cache.capacity * cache.head_dim);
    cache.values.resize(static_cast<std::size_t>(cache.batch) * cache.kv_heads *
                        cache.capacity * cache.head_dim_value);
    return cache;
}

void append_kv(KVCache& cache, const std::vector<float>& keys,
               const std::vector<float>& values, int tokens) {
    if (tokens <= 0 || cache.length + tokens > cache.capacity)
        throw std::invalid_argument("KV cache capacity exceeded");
    const std::size_t key_values = static_cast<std::size_t>(cache.batch) * cache.kv_heads *
                                   tokens * cache.head_dim;
    const std::size_t value_values = static_cast<std::size_t>(cache.batch) * cache.kv_heads *
                                     tokens * cache.head_dim_value;
    if (keys.size() != key_values || values.size() != value_values)
        throw std::invalid_argument("KV append shape mismatch");
    for (int batch = 0; batch < cache.batch; ++batch) {
        for (int head = 0; head < cache.kv_heads; ++head) {
            const std::size_t source_key = (static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                           tokens * cache.head_dim;
            const std::size_t target_key = ((static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                            cache.capacity + cache.length) * cache.head_dim;
            std::copy_n(keys.data() + source_key, static_cast<std::size_t>(tokens) * cache.head_dim,
                        cache.keys.data() + target_key);
            const std::size_t source_value = (static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                             tokens * cache.head_dim_value;
            const std::size_t target_value = ((static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                              cache.capacity + cache.length) * cache.head_dim_value;
            std::copy_n(values.data() + source_value,
                        static_cast<std::size_t>(tokens) * cache.head_dim_value,
                        cache.values.data() + target_value);
        }
    }
    cache.length += tokens;
}

AttentionBenchmarkResult execute_decode_attention(const AttentionExecutablePlan& plan,
                                                  const std::vector<float>& query,
                                                  const KVCache& cache,
                                                  int warmup, int repetitions) {
    if (cache.length <= 0 || cache.length > plan.config.sequence_kv)
        throw std::invalid_argument("KV cache length violates compiled guard");
    AttentionData data;
    data.query = query;
    data.key.resize(static_cast<std::size_t>(cache.batch) * cache.kv_heads *
                    cache.length * cache.head_dim);
    data.value.resize(static_cast<std::size_t>(cache.batch) * cache.kv_heads *
                      cache.length * cache.head_dim_value);
    for (int batch = 0; batch < cache.batch; ++batch)
        for (int head = 0; head < cache.kv_heads; ++head) {
            const std::size_t cache_key = (static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                          cache.capacity * cache.head_dim;
            const std::size_t data_key = (static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                         cache.length * cache.head_dim;
            std::copy_n(cache.keys.data() + cache_key,
                        static_cast<std::size_t>(cache.length) * cache.head_dim,
                        data.key.data() + data_key);
            const std::size_t cache_value = (static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                            cache.capacity * cache.head_dim_value;
            const std::size_t data_value = (static_cast<std::size_t>(batch) * cache.kv_heads + head) *
                                           cache.length * cache.head_dim_value;
            std::copy_n(cache.values.data() + cache_value,
                        static_cast<std::size_t>(cache.length) * cache.head_dim_value,
                        data.value.data() + data_value);
        }
    AttentionExecutablePlan specialized = plan;
    specialized.config.sequence_kv = cache.length;
    specialized.plan.schedule.kv_tile = std::min(
        specialized.plan.schedule.kv_tile, cache.length);
    specialized.plan.split_kv = specialized.plan.schedule.threads > 1
        ? std::clamp(std::max(2, cache.length / 128), 2,
                     specialized.plan.schedule.threads)
        : 1;
    return benchmark(specialized, data, warmup, repetitions);
}

void write_attention_experiment_csv(const std::filesystem::path& path,
                                    const AttentionConfig& base,
                                    int threads, int repetitions) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write attention CSV: " + path.string());
    output << "sequence,heads,head_dim,causal,strategy,qtile,kvtile,p50_ms,p95_ms,"
              "temporary_bytes,bytes_read,bytes_written,arithmetic_intensity,max_error\n";
    for (const int sequence : {128, 256, 512}) {
        AttentionConfig config = base;
        config.sequence_query = sequence;
        config.sequence_kv = sequence;
        const auto data = make_attention_data(config, static_cast<std::uint32_t>(sequence));
        for (const auto strategy : {AttentionLoweringStrategy::Materialized,
                                    AttentionLoweringStrategy::TiledMaterialized,
                                    AttentionLoweringStrategy::IOAware,
                                    AttentionLoweringStrategy::AutoScheduledIOAware}) {
            AttentionPlanOptions options = select_attention_plan(config, TargetInfo::detect(), threads);
            options.strategy = strategy;
            if (strategy == AttentionLoweringStrategy::AutoScheduledIOAware)
                options = tune_attention_plan(config, data, TargetInfo::detect(), threads, 0, 1);
            const auto plan = AttentionCompiler{}.compile(config, options);
            const auto measured = execute_attention(plan, data, 1, repetitions);
            output << sequence << ',' << config.query_heads << ',' << config.head_dim << ','
                   << config.causal << ',' << attention_strategy_name(strategy) << ','
                   << options.schedule.query_tile << ',' << options.schedule.kv_tile << ','
                   << measured.p50_milliseconds << ',' << measured.p95_milliseconds << ','
                   << plan.memory.temporary_bytes << ',' << plan.simulation.bytes_read << ','
                   << plan.simulation.bytes_written << ',' << plan.simulation.arithmetic_intensity
                   << ',' << measured.max_error << '\n';
        }
    }
}

void write_attention_scaling_csv(const std::filesystem::path& path,
                                 const AttentionConfig& base,
                                 int threads) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write attention scaling CSV: " + path.string());
    output << "sequence,heads,kv_heads,head_dim,strategy,qtile,kvtile,temporary_bytes,"
              "bytes_read,bytes_written,arithmetic_intensity,qk_tiles,skipped_causal_tiles,"
              "score_evaluations,exp_evaluations,reduction_operations,l2_traffic,llc_traffic,"
              "l2_hit_rate,llc_hit_rate\n";
    const auto target = TargetInfo::detect();
    for (const int heads : {8, 12}) {
        for (const int head_dim : {64, 128}) {
            for (const int sequence : {128, 256, 512, 1024, 2048, 4096}) {
                AttentionConfig config = base;
                config.query_heads = heads;
                config.kv_heads = heads;
                config.head_dim = head_dim;
                config.head_dim_value = head_dim;
                config.sequence_query = sequence;
                config.sequence_kv = sequence;
                for (const auto strategy : {AttentionLoweringStrategy::Materialized,
                                            AttentionLoweringStrategy::TiledMaterialized,
                                            AttentionLoweringStrategy::IOAware,
                                            AttentionLoweringStrategy::AutoScheduledIOAware}) {
                    auto options = select_attention_plan(config, target, threads);
                    options.strategy = strategy;
                    if (strategy == AttentionLoweringStrategy::TiledMaterialized ||
                        strategy == AttentionLoweringStrategy::IOAware) {
                        options.schedule.query_tile = 32;
                        options.schedule.kv_tile = 64;
                    }
                    const auto simulation = simulate_attention(config, options, target);
                    output << sequence << ',' << heads << ',' << heads << ',' << head_dim << ','
                           << attention_strategy_name(strategy) << ','
                           << options.schedule.query_tile << ',' << options.schedule.kv_tile << ','
                           << simulation.temporary_bytes << ',' << simulation.bytes_read << ','
                           << simulation.bytes_written << ',' << simulation.arithmetic_intensity << ','
                           << simulation.qk_tiles << ',' << simulation.skipped_causal_tiles << ','
                           << simulation.score_evaluations << ',' << simulation.exp_evaluations << ','
                           << simulation.reduction_operations << ',' << simulation.estimated_l2_traffic
                           << ',' << simulation.estimated_llc_traffic << ','
                           << simulation.estimated_l2_hit_rate << ','
                           << simulation.estimated_llc_hit_rate << '\n';
                }
            }
        }
    }
}

}  // namespace schedforge
