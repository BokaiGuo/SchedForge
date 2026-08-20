#pragma once

#include "schedforge/attention_compiler.h"
#include "schedforge/compiler.h"
#include "schedforge/decoder_compiler.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace schedforge {

struct PagedKVConfig {
    int batch = 1;
    int kv_heads = 1;
    int head_dim = 1;
    int head_dim_value = 1;
    int page_tokens = 16;
    int capacity = 0;
};

struct PagedKVCache {
    PagedKVConfig config;
    int length = 0;
    std::vector<int> page_table;
    std::vector<float> keys;
    std::vector<float> values;
    std::vector<int> free_pages;
    std::string dump() const;
};

PagedKVCache make_paged_kv_cache(const PagedKVConfig& config);
void append_paged_kv(PagedKVCache& cache, const std::vector<float>& keys,
                     const std::vector<float>& values, int tokens);
void truncate_paged_kv(PagedKVCache& cache, int new_length);
void release_paged_kv_pages(PagedKVCache& cache, const std::vector<int>& pages);
KVCache gather_paged_kv(const PagedKVCache& cache);
AttentionBenchmarkResult execute_paged_decode_attention(
    const AttentionExecutablePlan& plan, const std::vector<float>& query,
    const PagedKVCache& cache, int warmup = 1, int repetitions = 5,
    bool validate_result = true);

struct QuantizedMatMulConfig {
    int m = 128;
    int n = 128;
    int k = 128;
    bool bias = true;
    bool relu = true;
    bool per_channel = true;
};

struct QuantizedMatMulWeights {
    std::vector<std::int8_t> values;
    std::vector<float> scales;
    int zero_point = 0;
};

struct QuantizedMatMulResult {
    double milliseconds = 0.0;
    double gflops = 0.0;
    double max_error = 0.0;
    std::vector<float> output;
};

QuantizedMatMulWeights quantize_matmul_weights(const std::vector<float>& weights,
                                                const QuantizedMatMulConfig& config);
QuantizedMatMulResult execute_quantized_matmul(
    const QuantizedMatMulConfig& config, const std::vector<float>& input,
    const QuantizedMatMulWeights& weights, const std::vector<float>& bias,
    int warmup = 1, int repetitions = 5);

struct QuantizedDecoderWeights {
    QuantizedMatMulWeights query;
    QuantizedMatMulWeights key;
    QuantizedMatMulWeights value;
    QuantizedMatMulWeights output;
    QuantizedMatMulWeights gate;
    QuantizedMatMulWeights up;
    QuantizedMatMulWeights down;
};

struct QuantizedDecoderResult {
    double milliseconds = 0.0;
    double tokens_per_second = 0.0;
    double max_error = 0.0;
    std::vector<float> output;
};

QuantizedDecoderWeights quantize_decoder_weights(const DecoderConfig& config,
                                                  const DecoderData& data,
                                                  bool per_channel = true);
QuantizedDecoderResult execute_quantized_decoder_layer(
    const DecoderConfig& config, const DecoderData& data,
    const QuantizedDecoderWeights& weights, int warmup = 1,
    int repetitions = 5);

struct TransferSchedule {
    std::size_t chunk_bytes = 64 * 1024;
    int workers = 1;
    bool non_temporal = false;
    std::string dump() const;
};

struct TransferBenchmarkResult {
    double milliseconds = 0.0;
    double gigabytes_per_second = 0.0;
    std::size_t bytes = 0;
    TransferSchedule schedule;
};

TransferBenchmarkResult benchmark_transfer(const std::vector<std::uint8_t>& source,
                                            const TransferSchedule& schedule,
                                            int repetitions = 5);
TransferSchedule tune_transfer_schedule(const std::vector<std::uint8_t>& source,
                                         int max_workers = 4,
                                         int repetitions = 3);

struct NeonCodegenReport {
    bool compiled_for_neon = false;
    bool host_supports_neon = false;
    bool cross_compile_attempted = false;
    bool cross_compile_succeeded = false;
    std::string architecture;
    std::string cross_compiler;
    std::string source;
    std::string diagnostic;
    std::string dump() const;
};

NeonCodegenReport inspect_neon_codegen();
std::string generate_neon_matmul_source(int m = 4, int n = 4, int k = 4);

struct FuzzSummary {
    std::uint64_t seed = 1;
    int iterations = 0;
    int passed = 0;
    int rejected = 0;
    int failures = 0;
    std::string first_failure;
    std::string dump() const;
};

FuzzSummary run_schedforge_fuzz(std::uint64_t seed, int iterations);

}  // namespace schedforge
