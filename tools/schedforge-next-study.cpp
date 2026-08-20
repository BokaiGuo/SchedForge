#include "schedforge/next_milestones.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char** argv) {
    try {
        bool quick = false;
        for (int index = 1; index < argc; ++index) quick = quick || std::string(argv[index]) == "--quick";
        const int extent = quick ? 32 : 96;
        const schedforge::QuantizedMatMulConfig quant_config{extent, extent, extent, true, true, true};
        std::mt19937 generator(31);
        std::vector<float> input(static_cast<std::size_t>(extent) * extent);
        std::vector<float> weights(input.size());
        std::vector<float> bias(extent, 0.1F);
        for (float& value : input) value = static_cast<float>(generator() % 100) / 100.0F - 0.5F;
        for (float& value : weights) value = static_cast<float>(generator() % 100) / 100.0F - 0.5F;
        const auto quantized = schedforge::quantize_matmul_weights(weights, quant_config);
        const auto quantized_result = schedforge::execute_quantized_matmul(
            quant_config, input, quantized, bias, 1, quick ? 2 : 7);
        const schedforge::DecoderConfig decoder_config{
            1, quick ? 2 : 4, 16, 32, 4, 2, 4, 1.0e-5F, 10000.0F, true,
            schedforge::DecoderFFNKind::Dense, 4, 2};
        const auto decoder_data = schedforge::make_decoder_data(decoder_config, 37);
        const auto decoder_weights = schedforge::quantize_decoder_weights(
            decoder_config, decoder_data);
        const auto decoder_result = schedforge::execute_quantized_decoder_layer(
            decoder_config, decoder_data, decoder_weights, 1, quick ? 2 : 5);
        const schedforge::MoeConfig moe_config{quick ? 8 : 32, 16, 32, 4, 2};
        const auto moe_data = schedforge::make_moe_data(moe_config, 41);
        const auto moe_routing = schedforge::route_topk(moe_config, moe_data);
        const auto moe_weights = schedforge::quantize_moe_weights(moe_config, moe_data);
        const auto moe_result = schedforge::execute_quantized_moe(
            moe_config, moe_data, moe_routing, moe_weights, 1, quick ? 2 : 5);

        schedforge::PagedKVConfig paged_config{1, 2, 8, 8, 8, 64};
        auto paged = schedforge::make_paged_kv_cache(paged_config);
        std::vector<float> keys(2 * 16 * 8, 0.1F), values(2 * 16 * 8, 0.2F);
        schedforge::append_paged_kv(paged, keys, values, 16);
        const schedforge::AttentionConfig paged_attention_config{1, 2, 2, 1, 16, 8, 8, true, 0.0F};
        const auto paged_plan = schedforge::AttentionCompiler{}.compile(
            paged_attention_config,
            {schedforge::AttentionLoweringStrategy::SplitKVDecode, {}, 1});
        const auto paged_attention = schedforge::execute_paged_decode_attention(
            paged_plan, std::vector<float>(16, 0.125F), paged, 1, quick ? 2 : 5, true);

        std::vector<std::uint8_t> transfer(quick ? 1U << 20U : 8U << 20U, 7);
        const auto transfer_schedule = schedforge::tune_transfer_schedule(transfer, 4, quick ? 1 : 3);
        const auto transfer_result = schedforge::benchmark_transfer(transfer, transfer_schedule, quick ? 2 : 7);
        const auto neon = schedforge::inspect_neon_codegen();
        const auto neon_source_path = std::filesystem::path("results/generated_neon_matmul.cpp");
        std::ofstream(neon_source_path) << neon.source;
        const auto output_path = std::filesystem::path("results/next_milestones.csv");
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream csv(output_path);
        csv << "version,paged_tokens,paged_pages,paged_attention_error,paged_attention_ms,int8_ms,int8_error,decoder_int8_ms,decoder_int8_error,moe_int8_ms,moe_int8_error,moe_int8_tokens_per_second,transfer_chunk,transfer_workers,transfer_gbps,neon_compiled,neon_host,neon_cross_compile,neon_runtime\n"
            << "0.18.0," << paged.length << ',' << paged.page_table.size() << ','
            << paged_attention.max_error << ',' << paged_attention.p50_milliseconds << ','
            << quantized_result.milliseconds << ',' << quantized_result.max_error << ','
            << decoder_result.milliseconds << ',' << decoder_result.max_error << ','
            << moe_result.milliseconds << ',' << moe_result.max_error << ','
            << moe_result.tokens_per_second << ','
            << transfer_schedule.chunk_bytes << ',' << transfer_schedule.workers << ','
            << transfer_result.gigabytes_per_second << ',' << neon.compiled_for_neon << ','
            << neon.host_supports_neon << ',' << neon.cross_compile_succeeded << ','
            << neon.runtime_succeeded << '\n';
        std::cout << "version=0.18.0\n"
                  << "paged_kv=" << paged.dump() << " direct_attention_error="
                  << paged_attention.max_error << '\n'
                  << "int8_ms=" << quantized_result.milliseconds << " int8_error="
                  << quantized_result.max_error << '\n'
                  << "decoder_int8_ms=" << decoder_result.milliseconds
                  << " decoder_int8_error=" << decoder_result.max_error << '\n'
                  << "moe_int8_ms=" << moe_result.milliseconds
                  << " moe_int8_error=" << moe_result.max_error
                  << " moe_int8_tokens_per_second=" << moe_result.tokens_per_second << '\n'
                  << "transfer=" << transfer_schedule.dump() << " gbps="
                  << transfer_result.gigabytes_per_second << '\n'
                  << "neon=" << neon.dump() << '\n';
        return quantized_result.max_error < 0.05 && decoder_result.max_error < 0.15 &&
            moe_result.max_error < 0.2 &&
            paged_attention.max_error < 1.0e-5 &&
            neon.cross_compile_succeeded &&
            (!neon.host_supports_neon || neon.runtime_succeeded) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
