#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace schedforge {

enum class LoopOrder { IJK, IKJ };
enum class ISA { Scalar, AVX2, AVX512, NEON };

struct TargetInfo {
    std::string architecture = "x86_64";
    ISA isa = ISA::AVX2;
    int vector_width = 8;
    int vector_registers = 16;
    int logical_cpus = 1;
    std::size_t l1_bytes = 32 * 1024;
    std::size_t l2_bytes = 1024 * 1024;
    std::size_t l3_bytes = 24 * 1024 * 1024;
    std::size_t cache_line_bytes = 64;
    std::size_t page_bytes = 4096;
    double l1_latency = 4.0;
    double l2_latency = 12.0;
    double l3_latency = 40.0;
    double dram_latency = 200.0;
    double memory_bandwidth_gbps = 0.0;
    double fma_gflops = 0.0;
    static TargetInfo detect();
    std::string str() const;
};

struct Problem {
    int m = 128;
    int n = 128;
    int k = 128;
    bool bias = true;
    bool relu = true;
};

struct Schedule {
    LoopOrder order = LoopOrder::IKJ;
    int bm = 32;
    int bn = 64;
    int bk = 32;
    int mr = 4;
    int nr = 8;
    int vector_width = 8;
    int threads = 1;
    int mc = 64;
    int nc = 128;
    int kc = 64;
    int unroll_k = 1;
    int prefetch_distance = 0;
    bool tiled = true;
    bool fused = true;
    bool pack_a = false;
    bool pack_b = false;
    bool pin_threads = false;
};

struct GraphIR {
    Problem problem;
    std::string dump() const;
};

struct LoopIR {
    Problem problem;
    Schedule schedule;
    std::string dump() const;
};

class LowerToLoopsPass {
public:
    LoopIR run(const GraphIR& graph) const;
};

class LoopInterchangePass {
public:
    void run(LoopIR& loop) const;
};

class LoopTilingPass {
public:
    void run(LoopIR& loop, int bm, int bn, int bk) const;
};

class VectorizePass {
public:
    void run(LoopIR& loop, int width) const;
};

class FusionPass {
public:
    void run(LoopIR& loop) const;
};

struct TensorData {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> bias;
    std::vector<float> output;
};

TensorData make_data(const Problem& problem, std::uint32_t seed = 7);
std::vector<float> reference(const Problem& problem, const TensorData& data);
void execute(const LoopIR& loop, const TensorData& data, std::vector<float>& output);
double max_abs_error(const std::vector<float>& lhs, const std::vector<float>& rhs);

struct BenchmarkResult {
    double milliseconds = 0.0;
    double gflops = 0.0;
    double max_error = 0.0;
};

BenchmarkResult benchmark(const LoopIR& loop, const TensorData& data,
                          int warmup, int repetitions);
BenchmarkResult benchmark_bf16(const Problem& problem, const TensorData& data,
                               int repetitions);
BenchmarkResult benchmark_int8(const Problem& problem, const TensorData& data,
                               int repetitions);

struct CacheLevelStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
};

struct SimulationResult {
    CacheLevelStats l1;
    CacheLevelStats l2;
    CacheLevelStats l3;
    std::uint64_t dram_accesses = 0;
    std::uint64_t memory_accesses = 0;
    std::uint64_t dtlb_hits = 0;
    std::uint64_t dtlb_misses = 0;
    std::uint64_t prefetched_lines = 0;
    std::uint64_t useful_prefetches = 0;
    double estimated_bandwidth_bytes = 0.0;
    double register_pressure = 0.0;
    double estimated_cycles = 0.0;
    double l1_miss_rate() const;
    double llc_miss_rate() const;
};

SimulationResult simulate(const LoopIR& loop);
SimulationResult simulate(const LoopIR& loop, const TargetInfo& target);

struct RegisterPressure {
    int accumulators = 0;
    int broadcasts = 0;
    int temporaries = 0;
    int total = 0;
    bool spills = false;
};

RegisterPressure estimate_register_pressure(const Schedule& schedule,
                                             const TargetInfo& target);

struct SearchResult {
    Schedule schedule;
    SimulationResult simulation;
    BenchmarkResult benchmark;
    std::size_t search_space = 0;
    std::size_t after_static_pruning = 0;
    std::size_t benchmarked = 0;
    std::size_t simulator_rank = 0;
};

SearchResult autoschedule(const GraphIR& graph, const TensorData& data,
                          int max_threads, std::size_t top_k,
                          int warmup, int repetitions);
SearchResult autoschedule(const GraphIR& graph, const TensorData& data,
                          const TargetInfo& target, int max_threads,
                          std::size_t top_k, int warmup, int repetitions);

std::string schedule_name(const Schedule& schedule);
void write_experiment_csv(const std::string& path, const GraphIR& graph,
                          const TensorData& data, int max_threads,
                          int warmup, int repetitions);

}  // namespace schedforge
