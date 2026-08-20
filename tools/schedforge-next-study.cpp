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

        schedforge::PagedKVConfig paged_config{1, 2, 8, 8, 8, 64};
        auto paged = schedforge::make_paged_kv_cache(paged_config);
        std::vector<float> keys(2 * 16 * 8, 0.1F), values(2 * 16 * 8, 0.2F);
        schedforge::append_paged_kv(paged, keys, values, 16);
        const auto flat = schedforge::gather_paged_kv(paged);

        std::vector<std::uint8_t> transfer(quick ? 1U << 20U : 8U << 20U, 7);
        const auto transfer_schedule = schedforge::tune_transfer_schedule(transfer, 4, quick ? 1 : 3);
        const auto transfer_result = schedforge::benchmark_transfer(transfer, transfer_schedule, quick ? 2 : 7);
        const auto neon = schedforge::inspect_neon_codegen();
        const auto output_path = std::filesystem::path("results/next_milestones.csv");
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream csv(output_path);
        csv << "version,paged_tokens,paged_pages,int8_ms,int8_error,transfer_chunk,transfer_workers,transfer_gbps,neon_compiled,neon_host\n"
            << "0.16.0," << paged.length << ',' << paged.page_table.size() << ','
            << quantized_result.milliseconds << ',' << quantized_result.max_error << ','
            << transfer_schedule.chunk_bytes << ',' << transfer_schedule.workers << ','
            << transfer_result.gigabytes_per_second << ',' << neon.compiled_for_neon << ','
            << neon.host_supports_neon << '\n';
        std::cout << "version=0.16.0\n"
                  << "paged_kv=" << paged.dump() << " flat_length=" << flat.length << '\n'
                  << "int8_ms=" << quantized_result.milliseconds << " int8_error="
                  << quantized_result.max_error << '\n'
                  << "transfer=" << transfer_schedule.dump() << " gbps="
                  << transfer_result.gigabytes_per_second << '\n'
                  << "neon=" << neon.dump() << '\n';
        return quantized_result.max_error < 0.05 && flat.length == 16 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
