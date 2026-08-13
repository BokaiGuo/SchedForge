#pragma once

#include "schedforge/compiler.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace schedforge {

enum class GraphOpKind {
    Input,
    Constant,
    MatMul,
    Add,
    Multiply,
    Maximum,
    Gelu,
    Broadcast,
    Reshape,
    Transpose,
    Reduce,
    Exp,
    Rsqrt,
    Convert,
    Softmax,
    Mask,
    AttentionSdpa,
    TopK,
    MoeHistogram,
    MoePrefixSum,
    MoeDispatch,
    MoeGroupedMatMul,
    SwiGLU,
    MoeCombine,
    Return
};

enum class DimensionKind { Static, Dynamic, Symbolic };

struct Dimension {
    DimensionKind kind = DimensionKind::Static;
    std::int64_t value = 1;
    std::string symbol;

    static Dimension fixed(std::int64_t value);
    static Dimension dynamic(std::string symbol = "?");
    static Dimension symbolic(std::string symbol);
    std::string str() const;
};

enum class TensorLayoutKind { RowMajor, ColumnMajor, Blocked, PackedA, PackedB };

struct TensorLayout {
    TensorLayoutKind kind = TensorLayoutKind::RowMajor;
    int block_m = 1;
    int block_n = 1;
    std::string str() const;
};

struct GraphTensorType {
    std::vector<Dimension> shape;
    DataType dtype = DataType::F32;
    TensorLayout layout;
    std::optional<float> quant_scale;
    int quant_zero_point = 0;
    int quant_axis = -1;
    std::string str() const;
    std::optional<std::size_t> staticBytes() const;
};

struct GraphValue {
    std::string name;
    GraphTensorType type;
    int producer = -1;
    std::vector<int> users;
    bool constant = false;
};

struct GraphOperation {
    GraphOpKind kind = GraphOpKind::Input;
    std::string name;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::unordered_map<std::string, std::string> attributes;
    std::string str(const std::vector<GraphValue>& values) const;
};

class TensorGraph {
public:
    int addInput(std::string name, GraphTensorType type, bool constant = false);
    int addConstant(std::string name, GraphTensorType type, std::string literal);
    int addOperation(GraphOpKind kind, std::string name, std::vector<int> inputs,
                     GraphTensorType output_type = {});
    void setReturn(int value);
    int returnValue() const;
    std::vector<GraphValue>& values();
    const std::vector<GraphValue>& values() const;
    std::vector<GraphOperation>& operations();
    const std::vector<GraphOperation>& operations() const;
    std::string dump() const;
private:
    std::vector<GraphValue> values_;
    std::vector<GraphOperation> operations_;
    int return_value_ = -1;
};

struct ShapeConstraint {
    std::string expression;
    bool runtime_guard = false;
};

class ShapeInferencePass {
public:
    std::vector<ShapeConstraint> run(TensorGraph& graph) const;
};

class GraphCanonicalizationPass {
public:
    TensorGraph run(const TensorGraph& graph) const;
};

enum class IteratorKind { Parallel, Reduction };

struct StructuredCompute {
    std::string name;
    std::vector<std::string> iterators;
    std::vector<IteratorKind> iterator_kinds;
    std::vector<std::string> indexing_maps;
    std::string body;
    std::string dump() const;
};

class StructuredComputeLowering {
public:
    std::vector<StructuredCompute> run(const TensorGraph& graph) const;
};

enum class TransformKind {
    Match,
    Tile,
    Interchange,
    RegisterTile,
    Vectorize,
    Unroll,
    Pack,
    Prefetch,
    Parallelize,
    Tensorize
};

struct TransformOperation {
    TransformKind kind = TransformKind::Match;
    std::vector<int> integers;
    std::string text;
    std::string str() const;
};

class TransformProgram {
public:
    void add(TransformOperation operation);
    const std::vector<TransformOperation>& operations() const;
    Schedule replay(Schedule seed = {}) const;
    LoopIR apply(const Problem& problem, Schedule seed = {}) const;
    std::string dump() const;
    static TransformProgram fromSchedule(const Schedule& schedule);
    static TransformProgram parse(const std::string& text);
private:
    std::vector<TransformOperation> operations_;
};

struct TensorIntrinsic {
    std::string name;
    DataType input_type = DataType::F32;
    DataType accumulator_type = DataType::F32;
    int mr = 1;
    int nr = 1;
    int vector_width = 1;
    ISA isa = ISA::Scalar;
};

class TensorizationPass {
public:
    std::optional<TensorIntrinsic> match(const StructuredCompute& compute,
                                         const Schedule& schedule,
                                         const TargetInfo& target) const;
};

class QuantizationPropagationPass {
public:
    std::size_t run(TensorGraph& graph) const;
};

struct FusionCost {
    double saved_memory_bytes = 0.0;
    double extra_compute = 0.0;
    double register_pressure = 0.0;
    double estimated_speedup = 1.0;
};

struct Dispatch {
    std::string name;
    std::vector<int> operations;
    std::vector<int> inputs;
    std::vector<int> outputs;
    FusionCost fusion_cost;
    Problem kernel_problem;
    Schedule schedule;
    TransformProgram transforms;
    std::optional<TensorIntrinsic> intrinsic;
    std::size_t tuning_search_space = 0;
    std::size_t hardware_measurements = 0;
    bool tuning_cache_hit = false;
    std::string tuning_source = "default";
    std::string epilogue;
    std::string dump(const TensorGraph& graph) const;
};

class FusionPlanner {
public:
    bool legal(const TensorGraph& graph, int producer, int consumer) const;
    FusionCost profitability(const TensorGraph& graph,
                             const std::vector<int>& operations) const;
    std::vector<Dispatch> run(const TensorGraph& graph) const;
};

struct GraphBufferPlan {
    std::string name;
    std::size_t bytes = 0;
    std::size_t offset = 0;
    std::size_t alignment = 64;
    int first_dispatch = 0;
    int last_dispatch = 0;
    int value = -1;
    bool external = false;
    TensorLayout layout;
};

struct BufferizationResult {
    std::vector<GraphBufferPlan> buffers;
    std::size_t naive_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::string dump() const;
};

class GraphLayoutPlanner {
public:
    std::size_t run(TensorGraph& graph, std::vector<Dispatch>& dispatches) const;
};

class GraphBufferizer {
public:
    BufferizationResult run(const TensorGraph& graph,
                            const std::vector<Dispatch>& dispatches) const;
};

struct ShapeGuard {
    std::string expression;
    std::string target;
};

struct MLPConfig;
struct MLPData;

struct ExecutablePlan {
    TensorGraph graph;
    std::vector<StructuredCompute> structured_computes;
    std::vector<Dispatch> dispatches;
    std::vector<LoopIR> scheduled_loops;
    BufferizationResult memory;
    std::vector<ShapeGuard> guards;
    std::vector<std::string> llvm_ir;
    std::size_t layout_conversions_removed = 0;
    double llvm_compile_milliseconds = 0.0;
    std::string target = "native-cpu";
    std::string hardware;
    ExecutablePlan specializeMLP(const MLPConfig& config,
                                 const MLPData* data = nullptr) const;
    std::string dump() const;
    void save(const std::filesystem::path& path) const;
};

struct MLPConfig {
    int batch = 1;
    int sequence = 16;
    int hidden = 64;
    int intermediate = 128;
};

struct MLPData {
    std::vector<float> input;
    std::vector<float> weight1;
    std::vector<float> bias1;
    std::vector<float> weight2;
    std::vector<float> bias2;
};

struct GraphBenchmarkResult {
    double milliseconds = 0.0;
    double max_error = 0.0;
    std::vector<float> output;
};

struct ScheduleMeasurement {
    Problem problem;
    Schedule schedule;
    SimulationResult simulation;
    double milliseconds = 0.0;
};

class MeasurementDatabase {
public:
    void add(ScheduleMeasurement measurement);
    const std::vector<ScheduleMeasurement>& records() const;
    std::optional<ScheduleMeasurement> bestMatch(const Problem& problem,
                                                 int max_threads) const;
    void saveCsv(const std::filesystem::path& path) const;
    static MeasurementDatabase loadCsv(const std::filesystem::path& path);
private:
    std::vector<ScheduleMeasurement> records_;
};

class LearnedCostModel {
public:
    void fit(const MeasurementDatabase& database);
    double predictMilliseconds(const Problem& problem, const Schedule& schedule,
                               const SimulationResult& simulation) const;
    double hybridScore(const Problem& problem, const Schedule& schedule,
                       const SimulationResult& simulation,
                       double analytical_weight = 0.35) const;
    bool trained() const;
private:
    std::vector<double> weights_;
};

class ExecutableRuntime {
public:
    GraphBenchmarkResult runMLP(const ExecutablePlan& plan, const MLPConfig& config,
                                const MLPData& data, int warmup = 1,
                                int repetitions = 5) const;
};

TensorGraph build_transformer_mlp_graph(const MLPConfig& config,
                                        bool dynamic_batch = false);
TensorGraph build_mini_attention_graph(int sequence, int hidden,
                                       bool dynamic_sequence = false);
MLPData make_mlp_data(const MLPConfig& config, std::uint32_t seed = 7);
std::vector<float> reference_mlp(const MLPConfig& config, const MLPData& data);
GraphBenchmarkResult execute_mlp(const ExecutablePlan& plan, const MLPConfig& config,
                                 const MLPData& data, int warmup, int repetitions);

class StableHLOImporter {
public:
    TensorGraph importFile(const std::filesystem::path& path) const;
    TensorGraph importText(const std::string& text) const;
};

struct GraphCompileOptions {
    std::string target = "native-cpu";
    bool autotune = false;
    int max_threads = 8;
    std::size_t top_k = 8;
    int warmup = 1;
    int repetitions = 5;
    std::filesystem::path measurement_database;
};

class GraphCompiler {
public:
    explicit GraphCompiler(TargetInfo target = TargetInfo::detect());
    ExecutablePlan compile(TensorGraph graph, const GraphCompileOptions& options,
                           const MLPConfig* mlp_config = nullptr,
                           const MLPData* mlp_data = nullptr) const;
private:
    TargetInfo target_;
};

}  // namespace schedforge
