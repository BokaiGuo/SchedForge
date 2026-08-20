#include "schedforge/next_milestones.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace schedforge {
namespace {

void validate_paged_config(const PagedKVConfig& config) {
    if (config.batch <= 0 || config.kv_heads <= 0 || config.head_dim <= 0 ||
        config.head_dim_value <= 0 || config.page_tokens <= 0 || config.capacity <= 0)
        throw std::invalid_argument("invalid paged KV configuration");
}

std::size_t page_offset(const PagedKVConfig& config, int page, int batch, int head,
                        int token, int dim, int width) {
    return (((static_cast<std::size_t>(page) * config.batch + batch) * config.kv_heads + head) *
            config.page_tokens + token) * width + dim;
}

void validate_quantized(const QuantizedMatMulConfig& config,
                        const std::vector<float>& input,
                        const QuantizedMatMulWeights& weights,
                        const std::vector<float>& bias) {
    if (config.m <= 0 || config.n <= 0 || config.k <= 0)
        throw std::invalid_argument("invalid quantized matmul shape");
    if (input.size() != static_cast<std::size_t>(config.m) * config.k)
        throw std::invalid_argument("quantized matmul input shape mismatch");
    if (weights.values.size() != static_cast<std::size_t>(config.n) * config.k)
        throw std::invalid_argument("quantized matmul weight shape mismatch");
    const std::size_t scale_count = config.per_channel ? static_cast<std::size_t>(config.n) : 1;
    if (weights.scales.size() != scale_count || (config.bias && bias.size() != static_cast<std::size_t>(config.n)))
        throw std::invalid_argument("quantized matmul metadata shape mismatch");
}

void quantized_invoke(const QuantizedMatMulConfig& config, const std::vector<float>& input,
                      const QuantizedMatMulWeights& weights, const std::vector<float>& bias,
                      std::vector<float>& output) {
    output.assign(static_cast<std::size_t>(config.m) * config.n, 0.0F);
    for (int row = 0; row < config.m; ++row) {
        for (int column = 0; column < config.n; ++column) {
            const float scale = config.per_channel ? weights.scales[static_cast<std::size_t>(column)]
                                                   : weights.scales.front();
            float accumulator = 0.0F;
            for (int depth = 0; depth < config.k; ++depth) {
                accumulator += input[static_cast<std::size_t>(row) * config.k + depth] *
                    static_cast<float>(weights.values[static_cast<std::size_t>(column) * config.k + depth] -
                                       weights.zero_point) * scale;
            }
            if (config.bias) accumulator += bias[static_cast<std::size_t>(column)];
            if (config.relu) accumulator = std::max(0.0F, accumulator);
            output[static_cast<std::size_t>(row) * config.n + column] = accumulator;
        }
    }
}

std::vector<float> quantized_reference(const QuantizedMatMulConfig& config,
                                       const std::vector<float>& input,
                                       const QuantizedMatMulWeights& weights,
                                       const std::vector<float>& bias) {
    std::vector<float> dequantized(weights.values.size());
    for (int column = 0; column < config.n; ++column) {
        const float scale = config.per_channel ? weights.scales[static_cast<std::size_t>(column)]
                                               : weights.scales.front();
        for (int depth = 0; depth < config.k; ++depth)
            dequantized[static_cast<std::size_t>(column) * config.k + depth] =
                static_cast<float>(weights.values[static_cast<std::size_t>(column) * config.k + depth] -
                                   weights.zero_point) * scale;
    }
    std::vector<float> output(static_cast<std::size_t>(config.m) * config.n, 0.0F);
    for (int row = 0; row < config.m; ++row)
        for (int column = 0; column < config.n; ++column) {
            float accumulator = 0.0F;
            for (int depth = 0; depth < config.k; ++depth)
                accumulator += input[static_cast<std::size_t>(row) * config.k + depth] *
                    dequantized[static_cast<std::size_t>(column) * config.k + depth];
            if (config.bias) accumulator += bias[static_cast<std::size_t>(column)];
            if (config.relu) accumulator = std::max(0.0F, accumulator);
            output[static_cast<std::size_t>(row) * config.n + column] = accumulator;
        }
    return output;
}

void validate_paged_attention(const AttentionExecutablePlan& plan,
                              const std::vector<float>& query,
                              const PagedKVCache& cache) {
    const auto& config = plan.config;
    if (cache.length <= 0 || cache.length > config.sequence_kv)
        throw std::invalid_argument("paged KV length violates compiled guard");
    if (config.causal && config.sequence_query > cache.length)
        throw std::invalid_argument("paged causal query exceeds available KV length");
    if (cache.config.batch != config.batch || cache.config.kv_heads != config.kv_heads ||
        cache.config.head_dim != config.head_dim ||
        cache.config.head_dim_value != config.head_dim_value)
        throw std::invalid_argument("paged KV shape violates compiled guard");
    if (config.query_heads % config.kv_heads != 0)
        throw std::invalid_argument("invalid grouped-query attention configuration");
    const std::size_t query_values = static_cast<std::size_t>(config.batch) *
        config.query_heads * config.sequence_query * config.head_dim;
    if (query.size() != query_values)
        throw std::invalid_argument("paged attention query shape mismatch");
    const int required_pages = (cache.length + cache.config.page_tokens - 1) /
        cache.config.page_tokens;
    if (static_cast<int>(cache.page_table.size()) < required_pages)
        throw std::invalid_argument("paged KV page table is incomplete");
}

const float* paged_pointer(const PagedKVCache& cache, const std::vector<float>& storage,
                           int batch, int head, int token, int width) {
    const int physical_page = cache.page_table[static_cast<std::size_t>(
        token / cache.config.page_tokens)];
    const int page_token = token % cache.config.page_tokens;
    return storage.data() + page_offset(cache.config, physical_page, batch, head,
                                        page_token, 0, width);
}

float scalar_dot(const float* lhs, const float* rhs, int size) {
    float result = 0.0F;
    for (int index = 0; index < size; ++index) result += lhs[index] * rhs[index];
    return result;
}

std::vector<float> paged_attention_online(const AttentionExecutablePlan& plan,
                                          const std::vector<float>& query,
                                          const PagedKVCache& cache) {
    const auto& config = plan.config;
    const float scale = config.scale > 0.0F ? config.scale :
        1.0F / std::sqrt(static_cast<float>(config.head_dim));
    std::vector<float> output(static_cast<std::size_t>(config.batch) *
        config.query_heads * config.sequence_query * config.head_dim_value, 0.0F);
    for (int batch = 0; batch < config.batch; ++batch)
        for (int head = 0; head < config.query_heads; ++head) {
            const int kv_head = head * config.kv_heads / config.query_heads;
            for (int query_token = 0; query_token < config.sequence_query; ++query_token) {
                const float* query_row = query.data() +
                    (((static_cast<std::size_t>(batch) * config.query_heads + head) *
                      config.sequence_query + query_token) * config.head_dim);
                float maximum = -std::numeric_limits<float>::infinity();
                float denominator = 0.0F;
                std::vector<float> numerator(static_cast<std::size_t>(config.head_dim_value), 0.0F);
                const int absolute_query = cache.length - config.sequence_query + query_token;
                for (int key_token = 0; key_token < cache.length; ++key_token) {
                    if (config.causal && key_token > absolute_query) continue;
                    const float score = scalar_dot(
                        query_row, paged_pointer(cache, cache.keys, batch, kv_head,
                                                 key_token, config.head_dim),
                        config.head_dim) * scale;
                    const float next_maximum = std::max(maximum, score);
                    const float old_scale = std::isfinite(maximum)
                        ? std::exp(maximum - next_maximum) : 0.0F;
                    const float probability = std::exp(score - next_maximum);
                    denominator = denominator * old_scale + probability;
                    const float* value = paged_pointer(cache, cache.values, batch, kv_head,
                                                       key_token, config.head_dim_value);
                    for (int dim = 0; dim < config.head_dim_value; ++dim)
                        numerator[static_cast<std::size_t>(dim)] =
                            numerator[static_cast<std::size_t>(dim)] * old_scale +
                            probability * value[dim];
                    maximum = next_maximum;
                }
                const std::size_t output_base =
                    (((static_cast<std::size_t>(batch) * config.query_heads + head) *
                      config.sequence_query + query_token) * config.head_dim_value);
                for (int dim = 0; dim < config.head_dim_value; ++dim)
                    output[output_base + dim] = numerator[static_cast<std::size_t>(dim)] /
                        denominator;
            }
        }
    return output;
}

std::vector<float> paged_attention_reference(const AttentionExecutablePlan& plan,
                                             const std::vector<float>& query,
                                             const PagedKVCache& cache) {
    const auto& config = plan.config;
    const float scale = config.scale > 0.0F ? config.scale :
        1.0F / std::sqrt(static_cast<float>(config.head_dim));
    std::vector<float> output(static_cast<std::size_t>(config.batch) *
        config.query_heads * config.sequence_query * config.head_dim_value, 0.0F);
    for (int batch = 0; batch < config.batch; ++batch)
        for (int head = 0; head < config.query_heads; ++head) {
            const int kv_head = head * config.kv_heads / config.query_heads;
            for (int query_token = 0; query_token < config.sequence_query; ++query_token) {
                const float* query_row = query.data() +
                    (((static_cast<std::size_t>(batch) * config.query_heads + head) *
                      config.sequence_query + query_token) * config.head_dim);
                const int absolute_query = cache.length - config.sequence_query + query_token;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int key_token = 0; key_token < cache.length; ++key_token) {
                    if (config.causal && key_token > absolute_query) continue;
                    maximum = std::max(maximum, scalar_dot(
                        query_row, paged_pointer(cache, cache.keys, batch, kv_head,
                                                 key_token, config.head_dim),
                        config.head_dim) * scale);
                }
                float denominator = 0.0F;
                const std::size_t output_base =
                    (((static_cast<std::size_t>(batch) * config.query_heads + head) *
                      config.sequence_query + query_token) * config.head_dim_value);
                for (int key_token = 0; key_token < cache.length; ++key_token) {
                    if (config.causal && key_token > absolute_query) continue;
                    const float probability = std::exp(scalar_dot(
                        query_row, paged_pointer(cache, cache.keys, batch, kv_head,
                                                 key_token, config.head_dim),
                        config.head_dim) * scale - maximum);
                    denominator += probability;
                    const float* value = paged_pointer(cache, cache.values, batch, kv_head,
                                                       key_token, config.head_dim_value);
                    for (int dim = 0; dim < config.head_dim_value; ++dim)
                        output[output_base + dim] += probability * value[dim];
                }
                for (int dim = 0; dim < config.head_dim_value; ++dim)
                    output[output_base + dim] /= denominator;
            }
        }
    return output;
}

std::vector<float> transpose_decoder_weight(const std::vector<float>& weight,
                                            int input_width, int output_width) {
    if (weight.size() != static_cast<std::size_t>(input_width) * output_width)
        throw std::invalid_argument("decoder weight shape mismatch");
    std::vector<float> transposed(weight.size());
    for (int input = 0; input < input_width; ++input)
        for (int output = 0; output < output_width; ++output)
            transposed[static_cast<std::size_t>(output) * input_width + input] =
                weight[static_cast<std::size_t>(input) * output_width + output];
    return transposed;
}

QuantizedMatMulWeights quantize_decoder_matrix(const std::vector<float>& weight,
                                               int input_width, int output_width,
                                               bool per_channel) {
    return quantize_matmul_weights(transpose_decoder_weight(weight, input_width, output_width),
        {1, output_width, input_width, false, false, per_channel});
}

std::vector<float> decoder_rms_norm(const std::vector<float>& input,
                                    const std::vector<float>& weight,
                                    int rows, int width, float epsilon) {
    if (input.size() != static_cast<std::size_t>(rows) * width ||
        weight.size() != static_cast<std::size_t>(width))
        throw std::invalid_argument("quantized decoder RMSNorm shape mismatch");
    std::vector<float> output(input.size());
    for (int row = 0; row < rows; ++row) {
        double square_sum = 0.0;
        for (int column = 0; column < width; ++column) {
            const float value = input[static_cast<std::size_t>(row) * width + column];
            square_sum += static_cast<double>(value) * value;
        }
        const float inverse = 1.0F / std::sqrt(
            static_cast<float>(square_sum / width) + epsilon);
        for (int column = 0; column < width; ++column)
            output[static_cast<std::size_t>(row) * width + column] =
                input[static_cast<std::size_t>(row) * width + column] * inverse * weight[column];
    }
    return output;
}

std::vector<float> decoder_quantized_project(const std::vector<float>& input,
                                             const QuantizedMatMulWeights& weight,
                                             int rows, int input_width,
                                             int output_width) {
    std::vector<float> output;
    quantized_invoke({rows, output_width, input_width, false, false,
                      weight.scales.size() > 1},
                     input, weight, {}, output);
    return output;
}

void decoder_apply_rope(std::vector<float>& tensor, int batch, int heads,
                        int sequence, int head_dim, float theta) {
    for (int batch_index = 0; batch_index < batch; ++batch_index)
        for (int head = 0; head < heads; ++head)
            for (int token = 0; token < sequence; ++token)
                for (int dim = 0; dim + 1 < head_dim; dim += 2) {
                    const float frequency = std::pow(theta,
                        -static_cast<float>(dim) / static_cast<float>(head_dim));
                    const float angle = static_cast<float>(token) * frequency;
                    const std::size_t base = (((static_cast<std::size_t>(batch_index) * heads + head) *
                                                sequence + token) * head_dim + dim);
                    const float even = tensor[base];
                    const float odd = tensor[base + 1];
                    tensor[base] = even * std::cos(angle) - odd * std::sin(angle);
                    tensor[base + 1] = even * std::sin(angle) + odd * std::cos(angle);
                }
}

std::vector<float> row_major_to_heads(const std::vector<float>& input, int batch,
                                      int sequence, int heads, int head_dim) {
    std::vector<float> output(input.size());
    for (int batch_index = 0; batch_index < batch; ++batch_index)
        for (int token = 0; token < sequence; ++token)
            for (int head = 0; head < heads; ++head)
                for (int dim = 0; dim < head_dim; ++dim)
                    output[(((static_cast<std::size_t>(batch_index) * heads + head) *
                              sequence + token) * head_dim + dim)] =
                        input[static_cast<std::size_t>(batch_index * sequence + token) *
                              heads * head_dim + head * head_dim + dim];
    return output;
}

std::vector<float> heads_to_row_major(const std::vector<float>& input, int batch,
                                      int sequence, int heads, int head_dim) {
    std::vector<float> output(input.size());
    for (int batch_index = 0; batch_index < batch; ++batch_index)
        for (int token = 0; token < sequence; ++token)
            for (int head = 0; head < heads; ++head)
                for (int dim = 0; dim < head_dim; ++dim)
                    output[static_cast<std::size_t>(batch_index * sequence + token) *
                           heads * head_dim + head * head_dim + dim] =
                        input[(((static_cast<std::size_t>(batch_index) * heads + head) *
                                sequence + token) * head_dim + dim)];
    return output;
}

std::vector<float> add_vectors(const std::vector<float>& lhs,
                               const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("quantized decoder residual mismatch");
    std::vector<float> output(lhs.size());
    for (std::size_t index = 0; index < lhs.size(); ++index) output[index] = lhs[index] + rhs[index];
    return output;
}

std::vector<float> execute_quantized_decoder_once(
    const DecoderConfig& config, const DecoderData& data,
    const QuantizedDecoderWeights& weights) {
    if (config.ffn != DecoderFFNKind::Dense)
        throw std::invalid_argument("INT8 decoder currently requires dense FFN weights");
    const int rows = config.batch * config.sequence;
    const int kv_width = config.kv_heads * config.head_dim;
    const auto normalized1 = decoder_rms_norm(data.input, data.rms1_weight, rows,
                                               config.hidden, config.rms_epsilon);
    auto query = row_major_to_heads(decoder_quantized_project(
        normalized1, weights.query, rows, config.hidden, config.hidden),
        config.batch, config.sequence, config.query_heads, config.head_dim);
    auto key = row_major_to_heads(decoder_quantized_project(
        normalized1, weights.key, rows, config.hidden, kv_width),
        config.batch, config.sequence, config.kv_heads, config.head_dim);
    auto value = row_major_to_heads(decoder_quantized_project(
        normalized1, weights.value, rows, config.hidden, kv_width),
        config.batch, config.sequence, config.kv_heads, config.head_dim);
    decoder_apply_rope(query, config.batch, config.query_heads, config.sequence,
                       config.head_dim, config.rope_theta);
    decoder_apply_rope(key, config.batch, config.kv_heads, config.sequence,
                       config.head_dim, config.rope_theta);
    AttentionData attention_data{query,
        config.context_sequence > 0 ? data.cached_key : key,
        config.context_sequence > 0 ? data.cached_value : value};
    const AttentionConfig attention_config{config.batch, config.query_heads, config.kv_heads,
        config.sequence, config.context_sequence > 0 ? config.context_sequence : config.sequence,
        config.head_dim, config.head_dim, config.causal, 0.0F};
    const auto attended = reference_attention(attention_config, attention_data);
    const auto merged = heads_to_row_major(attended, config.batch, config.sequence,
                                           config.query_heads, config.head_dim);
    const auto projected = decoder_quantized_project(merged, weights.output, rows,
                                                      config.hidden, config.hidden);
    const auto attention_residual = add_vectors(projected, data.input);
    const auto normalized2 = decoder_rms_norm(attention_residual, data.rms2_weight, rows,
                                               config.hidden, config.rms_epsilon);
    const auto gate = decoder_quantized_project(normalized2, weights.gate, rows,
                                                 config.hidden, config.intermediate);
    const auto up = decoder_quantized_project(normalized2, weights.up, rows,
                                               config.hidden, config.intermediate);
    std::vector<float> activated(gate.size());
    for (std::size_t index = 0; index < gate.size(); ++index)
        activated[index] = (gate[index] / (1.0F + std::exp(-gate[index]))) * up[index];
    const auto down = decoder_quantized_project(activated, weights.down, rows,
                                                 config.intermediate, config.hidden);
    return add_vectors(attention_residual, down);
}

QuantizedMatMulWeights quantize_moe_matrix(const std::vector<float>& weights,
                                           int expert, int experts, int input_width,
                                           int output_width, bool per_channel) {
    const std::size_t matrix_size = static_cast<std::size_t>(input_width) * output_width;
    if (weights.size() != static_cast<std::size_t>(experts) * matrix_size)
        throw std::invalid_argument("MoE expert weight shape mismatch");
    std::vector<float> matrix(matrix_size);
    std::copy_n(weights.begin() + static_cast<std::ptrdiff_t>(expert * matrix_size),
                matrix_size, matrix.begin());
    return quantize_decoder_matrix(matrix, input_width, output_width, per_channel);
}

void validate_quantized_moe(const MoeConfig& config, const MoeData& data,
                            const RoutingTrace& routing,
                            const QuantizedMoeWeights& weights) {
    if (config.tokens <= 0 || config.hidden <= 0 || config.intermediate <= 0 ||
        config.experts <= 0 || config.top_k <= 0 || config.top_k > config.experts)
        throw std::invalid_argument("invalid quantized MoE configuration");
    if (routing.tokens <= 0 || routing.tokens > config.tokens ||
        routing.experts != config.experts || routing.top_k != config.top_k ||
        routing.expert_ids.size() != static_cast<std::size_t>(routing.tokens * routing.top_k) ||
        routing.expert_weights.size() != routing.expert_ids.size())
        throw std::invalid_argument("invalid quantized MoE routing trace");
    if (data.input.size() != static_cast<std::size_t>(routing.tokens * config.hidden))
        throw std::invalid_argument("quantized MoE input shape mismatch");
    if (weights.w1.size() != static_cast<std::size_t>(config.experts) ||
        weights.w3.size() != static_cast<std::size_t>(config.experts) ||
        weights.w2.size() != static_cast<std::size_t>(config.experts))
        throw std::invalid_argument("quantized MoE expert count mismatch");
}

std::vector<float> execute_quantized_moe_once(const MoeConfig& config,
                                              const MoeData& data,
                                              const RoutingTrace& routing,
                                              const QuantizedMoeWeights& weights) {
    std::vector<float> output(static_cast<std::size_t>(routing.tokens * config.hidden), 0.0F);
    for (int token = 0; token < routing.tokens; ++token) {
        const std::vector<float> input(data.input.begin() +
            static_cast<std::ptrdiff_t>(token * config.hidden), data.input.begin() +
            static_cast<std::ptrdiff_t>((token + 1) * config.hidden));
        for (int slot = 0; slot < routing.top_k; ++slot) {
            const std::size_t route_index = static_cast<std::size_t>(token * routing.top_k + slot);
            const int expert = routing.expert_ids[route_index];
            if (expert < 0 || expert >= config.experts)
                throw std::invalid_argument("quantized MoE routing expert out of range");
            std::vector<float> gate;
            std::vector<float> up;
            quantized_invoke({1, config.intermediate, config.hidden, false, false,
                              weights.w1[static_cast<std::size_t>(expert)].scales.size() > 1},
                             input, weights.w1[static_cast<std::size_t>(expert)], {}, gate);
            quantized_invoke({1, config.intermediate, config.hidden, false, false,
                              weights.w3[static_cast<std::size_t>(expert)].scales.size() > 1},
                             input, weights.w3[static_cast<std::size_t>(expert)], {}, up);
            for (int intermediate = 0; intermediate < config.intermediate; ++intermediate)
                up[static_cast<std::size_t>(intermediate)] =
                    up[static_cast<std::size_t>(intermediate)] /
                    (1.0F + std::exp(-up[static_cast<std::size_t>(intermediate)]));
            std::vector<float> activated(static_cast<std::size_t>(config.intermediate));
            for (int intermediate = 0; intermediate < config.intermediate; ++intermediate)
                activated[static_cast<std::size_t>(intermediate)] =
                    gate[static_cast<std::size_t>(intermediate)] *
                    up[static_cast<std::size_t>(intermediate)];
            std::vector<float> expert_output;
            quantized_invoke({1, config.hidden, config.intermediate, false, false,
                              weights.w2[static_cast<std::size_t>(expert)].scales.size() > 1},
                             activated, weights.w2[static_cast<std::size_t>(expert)], {},
                             expert_output);
            for (int feature = 0; feature < config.hidden; ++feature)
                output[static_cast<std::size_t>(token) * config.hidden + feature] +=
                    routing.expert_weights[route_index] * expert_output[static_cast<std::size_t>(feature)];
        }
    }
    return output;
}

}  // namespace

std::string PagedKVCache::dump() const {
    std::ostringstream out;
    out << "!sfg.paged_kv<pages=" << page_table.size() << ", page_tokens="
        << config.page_tokens << ", length=" << length << ", capacity=" << config.capacity
        << ", free=" << free_pages.size() << ">";
    return out.str();
}

PagedKVCache make_paged_kv_cache(const PagedKVConfig& config) {
    validate_paged_config(config);
    PagedKVCache cache;
    cache.config = config;
    const int page_count = (config.capacity + config.page_tokens - 1) / config.page_tokens;
    cache.free_pages.resize(static_cast<std::size_t>(page_count));
    std::iota(cache.free_pages.begin(), cache.free_pages.end(), 0);
    const std::size_t page_size = static_cast<std::size_t>(config.batch) * config.kv_heads *
                                  config.page_tokens;
    cache.keys.resize(static_cast<std::size_t>(page_count) * page_size * config.head_dim);
    cache.values.resize(static_cast<std::size_t>(page_count) * page_size * config.head_dim_value);
    return cache;
}

void append_paged_kv(PagedKVCache& cache, const std::vector<float>& keys,
                     const std::vector<float>& values, int tokens) {
    if (tokens <= 0 || cache.length + tokens > cache.config.capacity)
        throw std::invalid_argument("paged KV capacity exceeded");
    const std::size_t key_count = static_cast<std::size_t>(cache.config.batch) * cache.config.kv_heads *
                                   tokens * cache.config.head_dim;
    const std::size_t value_count = static_cast<std::size_t>(cache.config.batch) * cache.config.kv_heads *
                                     tokens * cache.config.head_dim_value;
    if (keys.size() != key_count || values.size() != value_count)
        throw std::invalid_argument("paged KV append shape mismatch");
    const int last_page = (cache.length + tokens - 1) / cache.config.page_tokens;
    while (static_cast<int>(cache.page_table.size()) <= last_page) {
        if (cache.free_pages.empty()) throw std::runtime_error("paged KV page allocator exhausted");
        cache.page_table.push_back(cache.free_pages.back());
        cache.free_pages.pop_back();
    }
    for (int token = 0; token < tokens; ++token) {
        const int logical_token = cache.length + token;
        const int logical_page = logical_token / cache.config.page_tokens;
        const int page_token = logical_token % cache.config.page_tokens;
        const int physical_page = cache.page_table[static_cast<std::size_t>(logical_page)];
        for (int batch = 0; batch < cache.config.batch; ++batch)
            for (int head = 0; head < cache.config.kv_heads; ++head) {
                const std::size_t source_key =
                    (((static_cast<std::size_t>(batch) * cache.config.kv_heads + head) * tokens + token) *
                     cache.config.head_dim);
                const std::size_t source_value =
                    (((static_cast<std::size_t>(batch) * cache.config.kv_heads + head) * tokens + token) *
                     cache.config.head_dim_value);
                for (int dim = 0; dim < cache.config.head_dim; ++dim)
                    cache.keys[page_offset(cache.config, physical_page, batch, head, page_token, dim,
                                           cache.config.head_dim)] = keys[source_key + dim];
                for (int dim = 0; dim < cache.config.head_dim_value; ++dim)
                    cache.values[page_offset(cache.config, physical_page, batch, head, page_token, dim,
                                             cache.config.head_dim_value)] = values[source_value + dim];
            }
    }
    cache.length += tokens;
}

void release_paged_kv_pages(PagedKVCache& cache, const std::vector<int>& pages) {
    for (int page : pages) {
        const int page_count = static_cast<int>(cache.keys.size() /
            (static_cast<std::size_t>(cache.config.batch) * cache.config.kv_heads *
             cache.config.page_tokens * cache.config.head_dim));
        if (page < 0 || page >= page_count)
            throw std::invalid_argument("invalid physical KV page");
        if (std::find(cache.page_table.begin(), cache.page_table.end(), page) != cache.page_table.end())
            throw std::invalid_argument("cannot release an active paged KV page");
        if (std::find(cache.free_pages.begin(), cache.free_pages.end(), page) == cache.free_pages.end())
            cache.free_pages.push_back(page);
    }
}

void truncate_paged_kv(PagedKVCache& cache, int new_length) {
    if (new_length < 0 || new_length > cache.length)
        throw std::invalid_argument("invalid paged KV truncation length");
    const int retained_pages = new_length == 0 ? 0 :
        (new_length + cache.config.page_tokens - 1) / cache.config.page_tokens;
    while (static_cast<int>(cache.page_table.size()) > retained_pages) {
        cache.free_pages.push_back(cache.page_table.back());
        cache.page_table.pop_back();
    }
    cache.length = new_length;
}

KVCache gather_paged_kv(const PagedKVCache& cache) {
    if (cache.length <= 0) throw std::invalid_argument("cannot gather empty paged KV cache");
    AttentionConfig config;
    config.batch = cache.config.batch;
    config.kv_heads = cache.config.kv_heads;
    config.head_dim = cache.config.head_dim;
    config.head_dim_value = cache.config.head_dim_value;
    config.sequence_kv = cache.length;
    KVCache flat = make_kv_cache(config, cache.length);
    flat.length = cache.length;
    for (int token = 0; token < cache.length; ++token) {
        const int physical_page = cache.page_table[static_cast<std::size_t>(token / cache.config.page_tokens)];
        const int page_token = token % cache.config.page_tokens;
        for (int batch = 0; batch < cache.config.batch; ++batch)
            for (int head = 0; head < cache.config.kv_heads; ++head) {
                const std::size_t key_base = ((static_cast<std::size_t>(batch) * cache.config.kv_heads + head) *
                                              cache.length + token) * cache.config.head_dim;
                const std::size_t value_base = ((static_cast<std::size_t>(batch) * cache.config.kv_heads + head) *
                                                cache.length + token) * cache.config.head_dim_value;
                for (int dim = 0; dim < cache.config.head_dim; ++dim)
                    flat.keys[key_base + dim] = cache.keys[page_offset(cache.config, physical_page, batch, head,
                                                                       page_token, dim, cache.config.head_dim)];
                for (int dim = 0; dim < cache.config.head_dim_value; ++dim)
                    flat.values[value_base + dim] = cache.values[page_offset(cache.config, physical_page, batch,
                                                                               head, page_token, dim,
                                                                               cache.config.head_dim_value)];
            }
    }
    return flat;
}

AttentionBenchmarkResult execute_paged_decode_attention(
    const AttentionExecutablePlan& plan, const std::vector<float>& query,
    const PagedKVCache& cache, int warmup, int repetitions, bool validate_result) {
    validate_paged_attention(plan, query, cache);
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        (void)paged_attention_online(plan, query, cache);
    std::vector<double> timings;
    std::vector<float> output;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        output = paged_attention_online(plan, query, cache);
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    AttentionBenchmarkResult result;
    result.milliseconds = std::accumulate(timings.begin(), timings.end(), 0.0) /
        static_cast<double>(timings.size());
    result.p50_milliseconds = timings[timings.size() / 2];
    result.p95_milliseconds = timings[std::min(timings.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(timings.size())) - 1))];
    result.output = std::move(output);
    if (validate_result)
        result.max_error = max_abs_error(result.output,
            paged_attention_reference(plan, query, cache));
    return result;
}

QuantizedMatMulWeights quantize_matmul_weights(const std::vector<float>& weights,
                                                const QuantizedMatMulConfig& config) {
    if (weights.size() != static_cast<std::size_t>(config.n) * config.k)
        throw std::invalid_argument("quantized weight shape mismatch");
    QuantizedMatMulWeights result;
    result.scales.resize(config.per_channel ? static_cast<std::size_t>(config.n) : 1, 0.0F);
    result.values.resize(weights.size());
    for (int column = 0; column < config.n; ++column) {
        float maximum = 0.0F;
        for (int depth = 0; depth < config.k; ++depth)
            maximum = std::max(maximum, std::abs(weights[static_cast<std::size_t>(column) * config.k + depth]));
        const float scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
        if (config.per_channel) result.scales[static_cast<std::size_t>(column)] = scale;
        else result.scales.front() = std::max(result.scales.front(), scale);
    }
    for (float& scale : result.scales) if (scale == 0.0F) scale = 1.0F;
    for (int column = 0; column < config.n; ++column) {
        const float scale = config.per_channel ? result.scales[static_cast<std::size_t>(column)]
                                               : result.scales.front();
        for (int depth = 0; depth < config.k; ++depth) {
            const float value = weights[static_cast<std::size_t>(column) * config.k + depth] / scale;
            result.values[static_cast<std::size_t>(column) * config.k + depth] =
                static_cast<std::int8_t>(std::clamp(std::round(value), -127.0F, 127.0F));
        }
    }
    return result;
}

QuantizedMatMulResult execute_quantized_matmul(
    const QuantizedMatMulConfig& config, const std::vector<float>& input,
    const QuantizedMatMulWeights& weights, const std::vector<float>& bias,
    int warmup, int repetitions) {
    validate_quantized(config, input, weights, bias);
    std::vector<float> output;
    const auto invoke = [&] { quantized_invoke(config, input, weights, bias, output); };
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) invoke();
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        invoke();
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double milliseconds = timings[timings.size() / 2];
    return {milliseconds, 2.0 * config.m * config.n * config.k / (milliseconds * 1.0e6),
            max_abs_error(output, quantized_reference(config, input, weights, bias)), output};
}

QuantizedDecoderWeights quantize_decoder_weights(const DecoderConfig& config,
                                                  const DecoderData& data,
                                                  bool per_channel) {
    if (config.ffn != DecoderFFNKind::Dense)
        throw std::invalid_argument("INT8 decoder currently requires dense FFN weights");
    const int kv_width = config.kv_heads * config.head_dim;
    return {
        quantize_decoder_matrix(data.query_weight, config.hidden, config.hidden, per_channel),
        quantize_decoder_matrix(data.key_weight, config.hidden, kv_width, per_channel),
        quantize_decoder_matrix(data.value_weight, config.hidden, kv_width, per_channel),
        quantize_decoder_matrix(data.output_weight, config.hidden, config.hidden, per_channel),
        quantize_decoder_matrix(data.gate_weight, config.hidden, config.intermediate, per_channel),
        quantize_decoder_matrix(data.up_weight, config.hidden, config.intermediate, per_channel),
        quantize_decoder_matrix(data.down_weight, config.intermediate, config.hidden, per_channel)};
}

QuantizedDecoderResult execute_quantized_decoder_layer(
    const DecoderConfig& config, const DecoderData& data,
    const QuantizedDecoderWeights& weights, int warmup, int repetitions) {
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        (void)execute_quantized_decoder_once(config, data, weights);
    std::vector<double> timings;
    std::vector<float> output;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        output = execute_quantized_decoder_once(config, data, weights);
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    QuantizedDecoderResult result;
    result.milliseconds = timings[timings.size() / 2];
    result.tokens_per_second = result.milliseconds > 0.0
        ? static_cast<double>(config.batch * config.sequence) * 1000.0 / result.milliseconds : 0.0;
    result.output = std::move(output);
    result.max_error = max_abs_error(result.output, reference_decoder_layer(config, data));
    return result;
}

QuantizedMoeWeights quantize_moe_weights(const MoeConfig& config,
                                         const MoeData& data, bool per_channel) {
    QuantizedMoeWeights result;
    result.w1.reserve(static_cast<std::size_t>(config.experts));
    result.w3.reserve(static_cast<std::size_t>(config.experts));
    result.w2.reserve(static_cast<std::size_t>(config.experts));
    for (int expert = 0; expert < config.experts; ++expert) {
        result.w1.push_back(quantize_moe_matrix(data.w1, expert, config.experts,
                                                config.hidden, config.intermediate, per_channel));
        result.w3.push_back(quantize_moe_matrix(data.w3, expert, config.experts,
                                                config.hidden, config.intermediate, per_channel));
        result.w2.push_back(quantize_moe_matrix(data.w2, expert, config.experts,
                                                config.intermediate, config.hidden, per_channel));
    }
    return result;
}

QuantizedMoeResult execute_quantized_moe(
    const MoeConfig& config, const MoeData& data, const RoutingTrace& routing,
    const QuantizedMoeWeights& weights, int warmup, int repetitions) {
    validate_quantized_moe(config, data, routing, weights);
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        (void)execute_quantized_moe_once(config, data, routing, weights);
    std::vector<double> timings;
    std::vector<float> output;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        output = execute_quantized_moe_once(config, data, routing, weights);
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    QuantizedMoeResult result;
    result.milliseconds = timings[timings.size() / 2];
    result.tokens_per_second = result.milliseconds > 0.0
        ? static_cast<double>(routing.tokens) * 1000.0 / result.milliseconds : 0.0;
    result.output = std::move(output);
    result.max_error = max_abs_error(result.output, reference_moe(config, data, routing));
    return result;
}

std::string TransferSchedule::dump() const {
    std::ostringstream out;
    out << "chunk=" << chunk_bytes << ",workers=" << workers << ",non_temporal=" << non_temporal;
    return out.str();
}

TransferBenchmarkResult benchmark_transfer(const std::vector<std::uint8_t>& source,
                                            const TransferSchedule& schedule,
                                            int repetitions) {
    if (source.empty() || schedule.chunk_bytes == 0 || schedule.workers <= 0)
        throw std::invalid_argument("invalid transfer benchmark configuration");
    std::vector<std::uint8_t> destination(source.size());
    const auto invoke = [&] {
        if (schedule.workers == 1) {
            for (std::size_t offset = 0; offset < source.size(); offset += schedule.chunk_bytes)
                std::memcpy(destination.data() + offset, source.data() + offset,
                            std::min(schedule.chunk_bytes, source.size() - offset));
        } else {
            std::vector<std::thread> workers;
            const std::size_t chunk = (source.size() + static_cast<std::size_t>(schedule.workers) - 1) /
                                      static_cast<std::size_t>(schedule.workers);
            for (int worker = 0; worker < schedule.workers; ++worker) {
                const std::size_t begin = static_cast<std::size_t>(worker) * chunk;
                const std::size_t end = std::min(source.size(), begin + chunk);
                if (begin >= end) continue;
                workers.emplace_back([&, begin, end] {
                    std::memcpy(destination.data() + begin, source.data() + begin, end - begin);
                });
            }
            for (auto& worker : workers) worker.join();
        }
    };
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        invoke();
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double milliseconds = timings[timings.size() / 2];
    if (destination.front() != source.front() || destination.back() != source.back())
        throw std::runtime_error("transfer validation failed");
    return {milliseconds, static_cast<double>(source.size()) / (milliseconds * 1.0e6),
            source.size(), schedule};
}

TransferSchedule tune_transfer_schedule(const std::vector<std::uint8_t>& source,
                                         int max_workers, int repetitions) {
    if (max_workers <= 0) throw std::invalid_argument("max_workers must be positive");
    TransferBenchmarkResult best;
    bool initialized = false;
    for (const std::size_t chunk : {4096U, 16384U, 65536U, 262144U})
        for (int workers = 1; workers <= max_workers; ++workers) {
            const auto candidate = benchmark_transfer(source, {chunk, workers, false}, repetitions);
            if (!initialized || candidate.milliseconds < best.milliseconds) {
                best = candidate;
                initialized = true;
            }
        }
    return best.schedule;
}

std::string generate_neon_matmul_source(int m, int n, int k) {
    if (m <= 0 || n < 4 || k <= 0) throw std::invalid_argument("invalid NEON source shape");
    std::ostringstream out;
    out << "#include <arm_neon.h>\n"
        << "void schedforge_neon_matmul(const float* a, const float* b, float* c) {\n"
        << "  for (int row = 0; row < " << m << "; ++row) {\n"
        << "    for (int column = 0; column + 4 <= " << n << "; column += 4) {\n"
        << "      float32x4_t acc = vdupq_n_f32(0.0f);\n"
        << "      for (int kk = 0; kk < " << k << "; ++kk) {\n"
        << "        acc = vfmaq_n_f32(acc, vld1q_f32(b + kk * " << n
        << " + column), a[row * " << k << " + kk]);\n"
        << "      }\n      vst1q_f32(c + row * " << n << " + column, acc);\n"
        << "    }\n  }\n}\n";
    return out.str();
}

void execute_neon_matmul(const float* input, const float* weights, float* output,
                         int m, int n, int k) {
    if (!input || !weights || !output || m <= 0 || n <= 0 || k <= 0)
        throw std::invalid_argument("invalid NEON matmul arguments");
    for (int row = 0; row < m; ++row) {
        int column = 0;
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (; column + 4 <= n; column += 4) {
            float32x4_t accumulator = vdupq_n_f32(0.0F);
            for (int depth = 0; depth < k; ++depth)
                accumulator = vfmaq_n_f32(accumulator,
                    vld1q_f32(weights + depth * n + column), input[row * k + depth]);
            vst1q_f32(output + row * n + column, accumulator);
        }
#endif
        for (; column < n; ++column) {
            float accumulator = 0.0F;
            for (int depth = 0; depth < k; ++depth)
                accumulator += input[row * k + depth] * weights[depth * n + column];
            output[row * n + column] = accumulator;
        }
    }
}

NeonCodegenReport inspect_neon_codegen() {
    NeonCodegenReport report;
#if defined(__aarch64__)
    report.architecture = "aarch64";
#elif defined(__arm__)
    report.architecture = "arm";
#else
    report.architecture = "x86_64";
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    report.compiled_for_neon = true;
    report.host_supports_neon = true;
    report.diagnostic = "compiled with ARM NEON intrinsics";
#else
    report.compiled_for_neon = false;
    report.host_supports_neon = false;
    report.diagnostic = "host build is not ARM NEON; use an AArch64 cross compiler for codegen validation";
#endif
    report.source = generate_neon_matmul_source();
    report.runtime_attempted = true;
    constexpr int runtime_m = 8;
    constexpr int runtime_n = 8;
    constexpr int runtime_k = 8;
    std::vector<float> runtime_input(runtime_m * runtime_k, 0.125F);
    std::vector<float> runtime_weights(runtime_k * runtime_n, 0.25F);
    std::vector<float> runtime_output(runtime_m * runtime_n, 0.0F);
    std::vector<float> runtime_reference(runtime_output.size(), 0.0F);
    for (int row = 0; row < runtime_m; ++row)
        for (int column = 0; column < runtime_n; ++column)
            for (int depth = 0; depth < runtime_k; ++depth)
                runtime_reference[row * runtime_n + column] +=
                    runtime_input[row * runtime_k + depth] *
                    runtime_weights[depth * runtime_n + column];
    const auto runtime_start = std::chrono::steady_clock::now();
    execute_neon_matmul(runtime_input.data(), runtime_weights.data(), runtime_output.data(),
                        runtime_m, runtime_n, runtime_k);
    report.runtime_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - runtime_start).count();
    report.runtime_max_error = max_abs_error(runtime_output, runtime_reference);
#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    report.runtime_succeeded = report.runtime_max_error < 1.0e-6;
#else
    report.runtime_succeeded = false;
    report.runtime_attempted = false;
#endif
    for (const std::string candidate : {"/usr/bin/clang++-18", "/usr/bin/clang++",
                                        "/usr/local/swift/usr/bin/clang++"}) {
        if (!std::filesystem::exists(candidate)) continue;
        report.cross_compile_attempted = true;
        report.cross_compiler = candidate;
        const auto unique = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        const auto source_path = std::filesystem::temp_directory_path() /
            ("schedforge-neon-" + unique + ".cpp");
        const auto diagnostic_path = source_path.string() + ".log";
        {
            std::ofstream source_file(source_path);
            source_file << report.source;
        }
        const std::string command = candidate +
            " -target aarch64-linux-gnu -march=armv8-a+simd -ffreestanding -fsyntax-only \"" +
            source_path.string() + "\" >\"" + diagnostic_path + "\" 2>&1";
        report.cross_compile_succeeded = std::system(command.c_str()) == 0;
        std::ifstream diagnostic_file(diagnostic_path);
        const std::string cross_diagnostic((std::istreambuf_iterator<char>(diagnostic_file)),
                                           std::istreambuf_iterator<char>());
        if (!cross_diagnostic.empty()) report.diagnostic += "; " + cross_diagnostic;
        std::filesystem::remove(source_path);
        std::filesystem::remove(diagnostic_path);
        break;
    }
    if (!report.cross_compile_attempted)
        report.diagnostic += "; no Clang cross compiler found";
    return report;
}

std::string NeonCodegenReport::dump() const {
    std::ostringstream out;
    out << "architecture=" << architecture << ",compiled_for_neon=" << compiled_for_neon
        << ",host_supports_neon=" << host_supports_neon
        << ",cross_compile_attempted=" << cross_compile_attempted
        << ",cross_compile_succeeded=" << cross_compile_succeeded
        << ",runtime_attempted=" << runtime_attempted
        << ",runtime_succeeded=" << runtime_succeeded
        << ",runtime_ms=" << runtime_milliseconds
        << ",runtime_error=" << runtime_max_error
        << ",cross_compiler=" << cross_compiler << ",diagnostic=" << diagnostic;
    return out.str();
}

std::string FuzzSummary::dump() const {
    std::ostringstream out;
    out << "seed=" << seed << ",iterations=" << iterations << ",passed=" << passed
        << ",rejected=" << rejected << ",failures=" << failures;
    if (!first_failure.empty()) out << ",first_failure=" << first_failure;
    return out.str();
}

FuzzSummary run_schedforge_fuzz(std::uint64_t seed, int iterations) {
    if (iterations <= 0) throw std::invalid_argument("fuzz iterations must be positive");
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    FuzzSummary summary{seed, iterations, 0, 0, 0, {}};
    for (int iteration = 0; iteration < iterations; ++iteration) {
        try {
            Problem problem{1 + static_cast<int>(generator() % 17),
                            1 + static_cast<int>(generator() % 17),
                            1 + static_cast<int>(generator() % 17),
                            generator() % 2 != 0, generator() % 2 != 0};
            Schedule schedule;
            schedule.mr = 1 + static_cast<int>(generator() % 8);
            schedule.nr = 1 + static_cast<int>(generator() % 16);
            schedule.vector_width = 1 + static_cast<int>(generator() % 8);
            schedule.threads = 1 + static_cast<int>(generator() % 4);
            const auto loop = apply_schedule(problem, schedule);
            verify_loop_ir(loop);
            const auto data = make_data(problem, static_cast<std::uint32_t>(generator()));
            std::vector<float> output;
            execute(loop, data, output);
            const auto expected = reference(problem, data);
            if (max_abs_error(output, expected) > 1.0e-3)
                throw std::runtime_error("fuzz numerical invariant failed");
            ++summary.passed;
        } catch (const std::invalid_argument&) {
            ++summary.rejected;
        } catch (const std::exception& error) {
            ++summary.failures;
            if (summary.first_failure.empty()) summary.first_failure = error.what();
        }
    }
    return summary;
}

}  // namespace schedforge
