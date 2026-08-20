#include "schedforge/next_milestones.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
    const auto flat = gather_paged_kv(cache);
    return execute_decode_attention(plan, query, flat, warmup, repetitions, validate_result);
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
    if (m <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("invalid NEON source shape");
    std::ostringstream out;
    out << "#include <arm_neon.h>\n"
        << "void schedforge_neon_matmul(const float* a, const float* b, float* c) {\n"
        << "  float32x4_t acc = vdupq_n_f32(0.0f);\n"
        << "  for (int kk = 0; kk < " << k << "; ++kk) {\n"
        << "    acc = vfmaq_n_f32(acc, vld1q_f32(a + kk * " << n << "), b[kk]);\n"
        << "  }\n  vst1q_f32(c, acc);\n}\n";
    return out.str();
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
    return report;
}

std::string NeonCodegenReport::dump() const {
    std::ostringstream out;
    out << "architecture=" << architecture << ",compiled_for_neon=" << compiled_for_neon
        << ",host_supports_neon=" << host_supports_neon << ",diagnostic=" << diagnostic;
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
