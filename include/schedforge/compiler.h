#pragma once

#include "schedforge/ir.h"
#include "schedforge/schedforge.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace schedforge {

class Compiler;

enum class DataType { F32, BF16, I8, I32 };
enum class Layout { RowMajor, ColumnMajor, PackedA, PackedB };

struct TensorSpec {
    std::string name;
    std::vector<std::int64_t> shape;
    DataType dtype = DataType::F32;
    Layout layout = Layout::RowMajor;
};

struct QuantizedTensor {
    std::vector<std::int8_t> values;
    float scale = 1.0F;
    int zero_point = 0;
};

std::vector<std::uint16_t> convert_to_bf16(const std::vector<float>& values);
std::vector<float> convert_from_bf16(const std::vector<std::uint16_t>& values);
QuantizedTensor quantize_int8(const std::vector<float>& values);
std::vector<float> dequantize_int8(const QuantizedTensor& tensor);

struct BufferPlan {
    std::string name;
    std::size_t bytes = 0;
    std::size_t alignment = 64;
    int first_use = 0;
    int last_use = 0;
    bool reusable = false;
    std::size_t offset = 0;
};

class LayoutPropagation {
public:
    std::vector<TensorSpec> run(const GraphIR& graph, const Schedule& schedule) const;
};

class MemoryPlanner {
public:
    std::vector<BufferPlan> plan(const GraphIR& graph, const Schedule& schedule) const;
    std::size_t peakBytes(const std::vector<BufferPlan>& buffers) const;
};

struct DynamicProblem {
    std::int64_t m = -1;
    std::int64_t n = -1;
    std::int64_t k = -1;
    Problem specialize(int runtime_m, int runtime_n, int runtime_k) const;
    std::string signature() const;
};

class ScheduleDSL {
public:
    static Schedule parse(const std::string& text);
    static std::string print(const Schedule& schedule);
};

class TensorToLoopLowering {
public:
    LoopIR lower(const GraphIR& graph) const;
};

class LoopPass {
public:
    virtual ~LoopPass() = default;
    virtual void run(LoopIR& loop) = 0;
};

class LoopPassManager {
public:
    template <typename Pass, typename... Args>
    Pass& addPass(Args&&... args) {
        auto pass = std::make_unique<Pass>(std::forward<Args>(args)...);
        Pass& reference = *pass;
        passes_.push_back(std::move(pass));
        return reference;
    }
    void run(LoopIR& loop);
private:
    std::vector<std::unique_ptr<LoopPass>> passes_;
};

class ApplySchedulePass final : public LoopPass {
public:
    explicit ApplySchedulePass(Schedule schedule);
    void run(LoopIR& loop) override;
private:
    Schedule schedule_;
};

class LLVMTextCodeGen {
public:
    std::string lower(const LoopIR& loop, const TargetInfo& target) const;
};

struct AssemblyReport {
    std::size_t instructions = 0;
    std::size_t vector_instructions = 0;
    std::size_t fma_instructions = 0;
    std::size_t loads = 0;
    std::size_t stores = 0;
    std::size_t branches = 0;
    std::size_t address_instructions = 0;
    std::size_t stack_accesses = 0;
    bool has_spill_pattern = false;
};

struct LLVMJITResult {
    double compile_milliseconds = 0.0;
    double execution_milliseconds = 0.0;
    double gflops = 0.0;
    double max_error = 0.0;
    std::string llvm_ir;
    std::string assembly;
    AssemblyReport assembly_report;
    int threads = 1;
};

class LLVMJITBackend {
public:
    bool available() const;
    LLVMJITResult benchmark(const LoopIR& loop, const TensorData& data,
                            int warmup, int repetitions) const;
};

struct AOTObject {
    std::string symbol = "schedforge_matmul_v1";
    std::string target_triple;
    std::string target_cpu;
    std::string llvm_ir;
    std::string assembly;
    std::vector<std::uint8_t> object_code;
    AssemblyReport assembly_report;
    double compile_milliseconds = 0.0;
};

class LLVMAOTBackend {
public:
    bool available() const;
    AOTObject compile(const LoopIR& loop) const;
};

struct AOTManifest {
    int format_version = 1;
    std::string schedforge_version = "0.11.0";
    std::string abi = "schedforge_matmul_v1";
    std::string symbol = "schedforge_matmul_v1";
    std::string target_triple;
    std::string target_cpu;
    Problem problem;
    int threads = 1;
    std::string loop_checksum;
    std::string object_checksum;
    std::string shared_object_checksum;
    std::string dump() const;
    static AOTManifest parse(const std::string& text);
};

struct AOTPackageResult {
    AOTManifest manifest;
    double compile_milliseconds = 0.0;
    double link_milliseconds = 0.0;
    std::filesystem::path path;
};

struct AOTBenchmarkResult {
    double load_milliseconds = 0.0;
    double execution_milliseconds = 0.0;
    double gflops = 0.0;
    double max_error = 0.0;
};

AOTPackageResult create_aot_package(const LoopIR& loop,
                                    const std::filesystem::path& path);
AOTManifest inspect_aot_package(const std::filesystem::path& path);
AOTBenchmarkResult benchmark_aot_package(const std::filesystem::path& path,
                                         const TensorData& data,
                                         int warmup, int repetitions);

struct CostBreakdown {
    double compute_cycles = 0.0;
    double memory_cycles = 0.0;
    double tlb_cycles = 0.0;
    double packing_cycles = 0.0;
    double spill_cycles = 0.0;
    double total_cycles = 0.0;
};

class CostModel {
public:
    CostBreakdown evaluate(const LoopIR& loop, const SimulationResult& simulation,
                           const TargetInfo& target) const;
    void calibrate(double predicted_cycles, double measured_nanoseconds,
                   double nominal_ghz = 3.5);
    double calibrationFactor() const;
private:
    double calibration_factor_ = 1.0;
};

class KernelCache {
public:
    explicit KernelCache(std::filesystem::path root);
    std::string key(const Problem& problem, const Schedule& schedule,
                    const TargetInfo& target) const;
    std::optional<Schedule> lookup(const std::string& key) const;
    void store(const std::string& key, const Schedule& schedule) const;
    std::optional<std::string> lookupArtifact(const std::string& key,
                                              const std::string& extension) const;
    void storeArtifact(const std::string& key, const std::string& extension,
                       const std::string& content) const;
private:
    std::filesystem::path root_;
};

struct RuntimeInfo {
    int logical_cpus = 1;
    int numa_nodes = 1;
    std::vector<int> allowed_cpus;
    static RuntimeInfo detect();
    std::string str() const;
};

class DynamicDispatcher {
public:
    DynamicDispatcher(Compiler& compiler, std::vector<int> buckets = {64, 128, 256, 512, 1024});
    SearchResult dispatch(const DynamicProblem& problem, int runtime_m, int runtime_n, int runtime_k,
                          int max_threads, std::size_t top_k, int warmup, int repetitions);
private:
    int bucket(int extent) const;
    Compiler& compiler_;
    std::vector<int> buckets_;
};

class AssemblyAnalyzer {
public:
    AssemblyReport analyze(const std::string& assembly) const;
};

struct HardwareCalibration {
    double memory_bandwidth_gbps = 0.0;
    double fma_gflops = 0.0;
    double stream_nanoseconds = 0.0;
};

class HardwareCalibrator {
public:
    HardwareCalibration run(std::size_t elements = 4 * 1024 * 1024,
                            int repetitions = 5) const;
    TargetInfo apply(TargetInfo target, const HardwareCalibration& calibration) const;
};

enum class SearchStrategy { Grid, Random, Greedy, Evolutionary };

struct SearchComparison {
    SearchStrategy strategy = SearchStrategy::Grid;
    std::size_t candidates_considered = 0;
    std::size_t hardware_measurements = 0;
    double best_gflops = 0.0;
    double elapsed_milliseconds = 0.0;
};

std::vector<Schedule> generate_schedule_candidates(int max_threads);
SearchComparison compare_search_strategy(const GraphIR& graph, const TensorData& data,
                                         SearchStrategy strategy, int max_threads,
                                         std::size_t budget, std::uint32_t seed = 7);
std::string search_strategy_name(SearchStrategy strategy);

struct CompilationResult {
    Module tensor_module{"tensor_program"};
    LoopIR loop;
    std::string llvm_ir;
    Schedule schedule;
    std::vector<TensorSpec> layouts;
    std::vector<BufferPlan> buffers;
};

class Compiler {
public:
    explicit Compiler(TargetInfo target = TargetInfo::detect(),
                      std::filesystem::path cache_root = ".schedforge-cache");
    CompilationResult compile(const GraphIR& graph, const Schedule& schedule) const;
    SearchResult compileAndTune(const GraphIR& graph, const TensorData& data,
                                int max_threads, std::size_t top_k,
                                int warmup, int repetitions);
    const TargetInfo& target() const;
private:
    Module buildTensorModule(const GraphIR& graph) const;
    TargetInfo target_;
    KernelCache cache_;
};

std::string data_type_name(DataType dtype);
std::string layout_name(Layout layout);

}  // namespace schedforge
