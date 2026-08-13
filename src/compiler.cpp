#include "schedforge/compiler.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <cstring>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <chrono>

namespace schedforge {
namespace {

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, delimiter)) if (!part.empty()) parts.push_back(part);
    return parts;
}

int parse_int(const std::string& text) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{}) throw std::invalid_argument("invalid integer: " + text);
    return value;
}

}  // namespace

std::vector<std::uint16_t> convert_to_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
        result[index] = static_cast<std::uint16_t>((bits + rounding) >> 16U);
    }
    return result;
}

std::vector<float> convert_from_bf16(const std::vector<std::uint16_t>& values) {
    std::vector<float> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(values[index]) << 16U;
        std::memcpy(&result[index], &bits, sizeof(bits));
    }
    return result;
}

QuantizedTensor quantize_int8(const std::vector<float>& values) {
    QuantizedTensor result;
    float maximum = 0.0F;
    for (float value : values) maximum = std::max(maximum, std::abs(value));
    result.scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
    result.values.resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float quantized = std::round(values[index] / result.scale);
        result.values[index] = static_cast<std::int8_t>(std::clamp(quantized, -127.0F, 127.0F));
    }
    return result;
}

std::vector<float> dequantize_int8(const QuantizedTensor& tensor) {
    std::vector<float> result(tensor.values.size());
    for (std::size_t index = 0; index < tensor.values.size(); ++index) {
        result[index] = static_cast<float>(static_cast<int>(tensor.values[index]) - tensor.zero_point) * tensor.scale;
    }
    return result;
}

Problem DynamicProblem::specialize(int runtime_m, int runtime_n, int runtime_k) const {
    return {static_cast<int>(m < 0 ? runtime_m : m), static_cast<int>(n < 0 ? runtime_n : n),
            static_cast<int>(k < 0 ? runtime_k : k), true, true};
}

std::string DynamicProblem::signature() const {
    auto extent = [](std::int64_t value) { return value < 0 ? std::string("?") : std::to_string(value); };
    return extent(m) + "x" + extent(n) + "x" + extent(k);
}

Schedule ScheduleDSL::parse(const std::string& text) {
    Schedule schedule;
    for (const auto& command : split(text, ';')) {
        const auto separator = command.find('=');
        if (separator == std::string::npos || separator == 0) {
            throw std::invalid_argument("invalid schedule command: " + command);
        }
        const std::string key = command.substr(0, separator);
        const std::string value = command.substr(separator + 1);
        if (key == "order") schedule.order = value == "ijk" ? LoopOrder::IJK : LoopOrder::IKJ;
        else if (key == "tile") {
            const auto values = split(value, ',');
            if (values.size() != 3) throw std::invalid_argument("tile requires three values");
            schedule.bm = parse_int(values[0]); schedule.bn = parse_int(values[1]); schedule.bk = parse_int(values[2]);
        } else if (key == "outer") {
            const auto values = split(value, ',');
            if (values.size() != 3) throw std::invalid_argument("outer requires three values");
            schedule.mc = parse_int(values[0]); schedule.nc = parse_int(values[1]); schedule.kc = parse_int(values[2]);
        } else if (key == "micro") {
            const auto values = split(value, ',');
            if (values.size() != 2) throw std::invalid_argument("micro requires two values");
            schedule.mr = parse_int(values[0]); schedule.nr = parse_int(values[1]);
        } else if (key == "vector") schedule.vector_width = parse_int(value);
        else if (key == "unroll") schedule.unroll_k = parse_int(value);
        else if (key == "threads") schedule.threads = parse_int(value);
        else if (key == "prefetch") schedule.prefetch_distance = parse_int(value);
        else if (key == "pack") { schedule.pack_a = value.find('a') != std::string::npos; schedule.pack_b = value.find('b') != std::string::npos; }
        else if (key == "fuse") schedule.fused = value == "1" || value == "true";
        else if (key == "pin") schedule.pin_threads = value == "1" || value == "true";
        else throw std::invalid_argument("unknown schedule key: " + key);
    }
    return schedule;
}

std::string ScheduleDSL::print(const Schedule& schedule) {
    std::ostringstream out;
    out << "order=" << (schedule.order == LoopOrder::IKJ ? "ikj" : "ijk")
        << ";outer=" << schedule.mc << ',' << schedule.nc << ',' << schedule.kc
        << ";tile=" << schedule.bm << ',' << schedule.bn << ',' << schedule.bk
        << ";micro=" << schedule.mr << ',' << schedule.nr
        << ";vector=" << schedule.vector_width << ";unroll=" << schedule.unroll_k
        << ";threads=" << schedule.threads << ";pack="
        << (schedule.pack_a ? "a" : "") << (schedule.pack_b ? "b" : "")
        << ";prefetch=" << schedule.prefetch_distance << ";fuse=" << (schedule.fused ? "true" : "false")
        << ";pin=" << (schedule.pin_threads ? "true" : "false");
    return out.str();
}

std::vector<TensorSpec> LayoutPropagation::run(const GraphIR& graph, const Schedule& schedule) const {
    return {
        {"A", {graph.problem.m, graph.problem.k}, DataType::F32,
         schedule.pack_a ? Layout::PackedA : Layout::RowMajor},
        {"B", {graph.problem.k, graph.problem.n}, DataType::F32,
         schedule.pack_b ? Layout::PackedB : Layout::RowMajor},
        {"bias", {graph.problem.n}, DataType::F32, Layout::RowMajor},
        {"output", {graph.problem.m, graph.problem.n}, DataType::F32, Layout::RowMajor}
    };
}

std::vector<BufferPlan> MemoryPlanner::plan(const GraphIR& graph, const Schedule& schedule) const {
    std::vector<BufferPlan> buffers;
    if (schedule.pack_a) buffers.push_back({"packed_A", 4ULL * schedule.mc * schedule.kc, 64, 1, 2, true, 0});
    if (schedule.pack_b) buffers.push_back({"packed_B", 4ULL * schedule.kc * schedule.nc, 64, 0, 3, true, 0});
    if (!schedule.fused && graph.problem.bias) {
        buffers.push_back({"matmul_intermediate", 4ULL * graph.problem.m * graph.problem.n, 64, 2, 3, true, 0});
    }
    if (!schedule.fused && graph.problem.relu) {
        buffers.push_back({"bias_intermediate", 4ULL * graph.problem.m * graph.problem.n, 64, 3, 4, true, 0});
    }
    std::size_t arena_end = 0;
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        bool reused = false;
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (buffers[previous].last_use < buffers[index].first_use &&
                buffers[previous].bytes >= buffers[index].bytes) {
                buffers[index].offset = buffers[previous].offset;
                reused = true;
                break;
            }
        }
        if (!reused) {
            arena_end = (arena_end + buffers[index].alignment - 1) / buffers[index].alignment * buffers[index].alignment;
            buffers[index].offset = arena_end;
            arena_end += buffers[index].bytes;
        }
    }
    return buffers;
}

std::size_t MemoryPlanner::peakBytes(const std::vector<BufferPlan>& buffers) const {
    std::size_t peak = 0;
    for (const auto& buffer : buffers) peak = std::max(peak, buffer.offset + buffer.bytes);
    return peak;
}

LoopIR TensorToLoopLowering::lower(const GraphIR& graph) const { return LowerToLoopsPass{}.run(graph); }
void LoopPassManager::run(LoopIR& loop) { for (const auto& pass : passes_) pass->run(loop); }
ApplySchedulePass::ApplySchedulePass(Schedule schedule) : schedule_(std::move(schedule)) {}
void ApplySchedulePass::run(LoopIR& loop) { loop.schedule = schedule_; }

std::string LLVMTextCodeGen::lower(const LoopIR& loop, const TargetInfo& target) const {
    const int width = std::max(1, loop.schedule.vector_width);
    std::ostringstream out;
    out << "; SchedForge LLVM-compatible textual IR\n"
        << "; target = " << target.str() << "\n"
        << "; schedule = " << ScheduleDSL::print(loop.schedule) << "\n"
        << "define void @matmul(ptr noalias %A, ptr noalias %B, ptr noalias %bias, ptr noalias %C, i64 %M, i64 %N, i64 %K) {\n"
        << "entry:\n  br label %i.loop\n"
        << "i.loop:\n  %i = phi i64 [ 0, %entry ], [ %i.next, %i.latch ]\n  br label %k.loop\n"
        << "k.loop:\n  %k = phi i64 [ 0, %i.loop ], [ %k.next, %k.latch ]\n  br label %j.loop\n"
        << "j.loop:\n  %j = phi i64 [ 0, %k.loop ], [ %j.next, %j.loop ]\n"
        << "  %avec = load float, ptr %A, align 4\n"
        << "  %bvec = load <" << width << " x float>, ptr %B, align 4\n"
        << "  %cvec = load <" << width << " x float>, ptr %C, align 4\n"
        << "  %ab = insertelement <" << width << " x float> poison, float %avec, i64 0\n"
        << "  %splat = shufflevector <" << width << " x float> %ab, <" << width << " x float> poison, <"
        << width << " x i32> zeroinitializer\n"
        << "  %mul = fmul <" << width << " x float> %splat, %bvec\n"
        << "  %sum = fadd <" << width << " x float> %cvec, %mul\n"
        << "  store <" << width << " x float> %sum, ptr %C, align 4\n"
        << "  %j.next = add i64 %j, " << width << "\n  %j.done = icmp uge i64 %j.next, %N\n  br i1 %j.done, label %k.latch, label %j.loop\n"
        << "k.latch:\n  %k.next = add i64 %k, 1\n  %k.done = icmp uge i64 %k.next, %K\n  br i1 %k.done, label %i.latch, label %k.loop\n"
        << "i.latch:\n  %i.next = add i64 %i, 1\n  %i.done = icmp uge i64 %i.next, %M\n  br i1 %i.done, label %exit, label %i.loop\n"
        << "exit:\n  ret void\n}\n";
    return out.str();
}

CostBreakdown CostModel::evaluate(const LoopIR& loop, const SimulationResult& simulation,
                                  const TargetInfo& target) const {
    CostBreakdown cost;
    const double lanes = static_cast<double>(std::max(1, loop.schedule.vector_width));
    const double threads = static_cast<double>(std::max(1, loop.schedule.threads));
    cost.compute_cycles = static_cast<double>(loop.problem.m) * loop.problem.n * loop.problem.k / (2.0 * lanes * threads);
    cost.memory_cycles = static_cast<double>(simulation.l1.hits) * 4.0 +
                         static_cast<double>(simulation.l2.hits) * 12.0 +
                         static_cast<double>(simulation.l3.hits) * 40.0 +
                         static_cast<double>(simulation.dram_accesses) * 200.0;
    cost.tlb_cycles = static_cast<double>(simulation.dtlb_misses) * 30.0;
    cost.packing_cycles = ((loop.schedule.pack_a ? 4.0 * loop.problem.m * loop.problem.k : 0.0) +
                           (loop.schedule.pack_b ? 4.0 * loop.problem.k * loop.problem.n : 0.0)) / 32.0;
    cost.spill_cycles = estimate_register_pressure(loop.schedule, target).spills ? cost.compute_cycles * 0.75 : 0.0;
    cost.total_cycles = (cost.compute_cycles + cost.memory_cycles + cost.tlb_cycles +
                         cost.packing_cycles + cost.spill_cycles) * calibration_factor_;
    return cost;
}
void CostModel::calibrate(double predicted_cycles, double measured_nanoseconds, double nominal_ghz) {
    if (predicted_cycles > 0.0 && measured_nanoseconds > 0.0)
        calibration_factor_ = measured_nanoseconds * nominal_ghz / predicted_cycles;
}
double CostModel::calibrationFactor() const { return calibration_factor_; }

KernelCache::KernelCache(std::filesystem::path root) : root_(std::move(root)) {}
std::string KernelCache::key(const Problem& problem, const Schedule& schedule, const TargetInfo& target) const {
    constexpr const char* cache_version = "runtime-v2-register-blocked";
    return std::string(cache_version) + "-" + std::to_string(problem.m) + "x" +
           std::to_string(problem.n) + "x" + std::to_string(problem.k) +
           "-" + target.architecture + "-" + std::to_string(target.vector_width) + "-" +
           std::to_string(std::hash<std::string>{}(ScheduleDSL::print(schedule)));
}
std::optional<Schedule> KernelCache::lookup(const std::string& key_value) const {
    std::ifstream input(root_ / (key_value + ".schedule"));
    if (!input) return std::nullopt;
    std::string text; std::getline(input, text);
    return ScheduleDSL::parse(text);
}
void KernelCache::store(const std::string& key_value, const Schedule& schedule) const {
    std::filesystem::create_directories(root_);
    std::ofstream(root_ / (key_value + ".schedule")) << ScheduleDSL::print(schedule) << '\n';
}
std::optional<std::string> KernelCache::lookupArtifact(const std::string& key_value,
                                                       const std::string& extension) const {
    std::ifstream input(root_ / (key_value + extension));
    if (!input) return std::nullopt;
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
void KernelCache::storeArtifact(const std::string& key_value, const std::string& extension,
                                const std::string& content) const {
    std::filesystem::create_directories(root_);
    std::ofstream(root_ / (key_value + extension)) << content;
}

RuntimeInfo RuntimeInfo::detect() {
    RuntimeInfo info;
    info.logical_cpus = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
    for (int cpu = 0; cpu < info.logical_cpus; ++cpu) info.allowed_cpus.push_back(cpu);
    info.numa_nodes = 0;
    const std::filesystem::path nodes("/sys/devices/system/node");
    if (std::filesystem::exists(nodes)) {
        for (const auto& entry : std::filesystem::directory_iterator(nodes)) {
            if (entry.is_directory() && entry.path().filename().string().starts_with("node")) ++info.numa_nodes;
        }
    }
    info.numa_nodes = std::max(1, info.numa_nodes);
    return info;
}

std::string RuntimeInfo::str() const {
    return "cpus=" + std::to_string(logical_cpus) + " numa_nodes=" + std::to_string(numa_nodes);
}

DynamicDispatcher::DynamicDispatcher(Compiler& compiler, std::vector<int> buckets)
    : compiler_(compiler), buckets_(std::move(buckets)) { std::sort(buckets_.begin(), buckets_.end()); }
int DynamicDispatcher::bucket(int extent) const {
    const auto found = std::lower_bound(buckets_.begin(), buckets_.end(), extent);
    return found == buckets_.end() ? extent : *found;
}
SearchResult DynamicDispatcher::dispatch(const DynamicProblem& problem, int runtime_m, int runtime_n, int runtime_k,
                                         int max_threads, std::size_t top_k, int warmup, int repetitions) {
    const auto specialized = problem.specialize(bucket(runtime_m), bucket(runtime_n), bucket(runtime_k));
    GraphIR graph{specialized};
    const auto data = make_data(specialized, 17);
    return compiler_.compileAndTune(graph, data, max_threads, top_k, warmup, repetitions);
}

AssemblyReport AssemblyAnalyzer::analyze(const std::string& assembly) const {
    AssemblyReport report;
    std::stringstream input(assembly);
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("ymm") != std::string::npos || line.find("xmm") != std::string::npos) ++report.vector_instructions;
        if (line.find("vfmadd") != std::string::npos || line.find("fmadd") != std::string::npos) ++report.fma_instructions;
        if (line.find("vmov") != std::string::npos || line.find(" mov") != std::string::npos) ++report.loads;
        if (line.find("(%rsp)") != std::string::npos && (line.find("ymm") != std::string::npos || line.find("xmm") != std::string::npos))
            report.has_spill_pattern = true;
        if (line.find("store") != std::string::npos || line.find("vmovups") != std::string::npos) ++report.stores;
    }
    return report;
}

HardwareCalibration HardwareCalibrator::run(std::size_t elements, int repetitions) const {
    std::vector<float> source(elements, 1.0F);
    std::vector<float> destination(elements, 0.0F);
    std::vector<double> timings;
    volatile float sink = 0.0F;
    for (int repetition = 0; repetition < std::max(1, repetitions); ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < elements; ++index) destination[index] = source[index] * 1.0001F;
        const auto end = std::chrono::steady_clock::now();
        sink = sink + destination[static_cast<std::size_t>(repetition) % elements];
        timings.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double nanoseconds = timings[timings.size() / 2];
    const double bytes = static_cast<double>(elements) * sizeof(float) * 2.0;
    HardwareCalibration calibration;
    calibration.stream_nanoseconds = nanoseconds;
    calibration.memory_bandwidth_gbps = bytes / nanoseconds;
    calibration.fma_gflops = static_cast<double>(elements) * 2.0 / nanoseconds;
    (void)sink;
    return calibration;
}
TargetInfo HardwareCalibrator::apply(TargetInfo target, const HardwareCalibration& calibration) const {
    target.memory_bandwidth_gbps = calibration.memory_bandwidth_gbps;
    target.fma_gflops = calibration.fma_gflops;
    return target;
}

Compiler::Compiler(TargetInfo target, std::filesystem::path cache_root)
    : target_(std::move(target)), cache_(std::move(cache_root)) {}
Module Compiler::buildTensorModule(const GraphIR& graph) const {
    Module module("tensor_program");
    IRBuilder builder(module.body());
    auto* a = builder.createInput("A", TensorType::get({graph.problem.m, graph.problem.k}));
    auto* b = builder.createInput("B", TensorType::get({graph.problem.k, graph.problem.n}));
    auto* result = builder.createMatMul(a, b, TensorType::get({graph.problem.m, graph.problem.n}));
    result->setAttribute("layout", "row_major");
    result->setAttribute("epilogue", graph.problem.bias && graph.problem.relu ? "bias_relu" : "none");
    builder.createReturn(result->result(0));
    return module;
}
CompilationResult Compiler::compile(const GraphIR& graph, const Schedule& schedule) const {
    CompilationResult result;
    result.tensor_module = buildTensorModule(graph);
    result.loop = TensorToLoopLowering{}.lower(graph);
    LoopPassManager passes;
    passes.addPass<ApplySchedulePass>(schedule);
    passes.run(result.loop);
    result.llvm_ir = LLVMTextCodeGen{}.lower(result.loop, target_);
    result.schedule = schedule;
    result.layouts = LayoutPropagation{}.run(graph, schedule);
    result.buffers = MemoryPlanner{}.plan(graph, schedule);
    const auto artifact_key = cache_.key(graph.problem, schedule, target_);
    cache_.store(artifact_key, schedule);
    cache_.storeArtifact(artifact_key, ".ll", result.llvm_ir);
    cache_.storeArtifact(artifact_key, ".loopir", result.loop.dump());
    return result;
}
SearchResult Compiler::compileAndTune(const GraphIR& graph, const TensorData& data,
                                      int max_threads, std::size_t top_k,
                                      int warmup, int repetitions) {
    Schedule tuning_key_schedule;
    tuning_key_schedule.threads = max_threads;
    tuning_key_schedule.bm = static_cast<int>(top_k);
    const auto cache_key = cache_.key(graph.problem, tuning_key_schedule, target_);
    if (const auto cached = cache_.lookup(cache_key)) {
        LoopIR loop{graph.problem, *cached};
        return {*cached, simulate(loop, target_), benchmark(loop, data, warmup, repetitions),
                1, 1, 1, 1, true};
    }
    auto result = autoschedule(graph, data, target_, max_threads, top_k, warmup, repetitions);
    cache_.store(cache_key, result.schedule);
    return result;
}
const TargetInfo& Compiler::target() const { return target_; }

std::string data_type_name(DataType dtype) {
    if (dtype == DataType::BF16) return "bf16";
    if (dtype == DataType::I8) return "i8";
    return "f32";
}
std::string layout_name(Layout layout) {
    if (layout == Layout::ColumnMajor) return "column_major";
    if (layout == Layout::PackedA) return "packed_a";
    if (layout == Layout::PackedB) return "packed_b";
    return "row_major";
}

}  // namespace schedforge
