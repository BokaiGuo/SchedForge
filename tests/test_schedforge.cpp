#include "schedforge/schedforge.h"
#include "schedforge/ir.h"
#include "schedforge/compiler.h"
#include "schedforge/graph_compiler.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_ir_passes() {
    schedforge::GraphIR graph{{17, 19, 13, true, true}};
    schedforge::LowerToLoopsPass lower;
    auto loop = lower.run(graph);
    require(loop.schedule.order == schedforge::LoopOrder::IJK, "lowering order");
    schedforge::LoopInterchangePass{}.run(loop);
    schedforge::LoopTilingPass{}.run(loop, 8, 16, 4);
    schedforge::VectorizePass{}.run(loop, 8);
    schedforge::FusionPass{}.run(loop);
    require(loop.schedule.order == schedforge::LoopOrder::IKJ, "interchange");
    require(loop.schedule.tiled && loop.schedule.bn == 16, "tiling");
    require(loop.schedule.vector_width == 8 && loop.schedule.fused, "vector/fusion");
}

class RenamePass final : public schedforge::OperationPass {
public:
    void run(schedforge::Module& module) override {
        module.body().operations().front()->setAttribute("stage", "canonical");
    }
};

void test_ssa_ir_and_pass_manager() {
    schedforge::Module module("matmul_module");
    schedforge::IRBuilder builder(module.body());
    const auto matrix_a = schedforge::TensorType::get({8, 16});
    const auto matrix_b = schedforge::TensorType::get({16, 4});
    auto* input_a = builder.createInput("A", matrix_a);
    auto* input_b = builder.createInput("B", matrix_b);
    auto* matmul = builder.createMatMul(input_a, input_b, schedforge::TensorType::get({8, 4}));
    require(matmul->result(0)->definingOp() == matmul, "ssa defining op");
    require(input_a->users().size() == 1 && input_a->users().front() == matmul, "ssa users");

    auto* loop = builder.createFor("i", 0, 8, 1);
    schedforge::IRBuilder loop_builder(loop->body());
    loop_builder.createLoad(input_a, {loop->inductionVariable()});
    loop_builder.createVectorLoad(input_a, {loop->inductionVariable()}, 8);
    loop_builder.createPrefetch(input_a, {loop->inductionVariable()}, 4);
    loop_builder.createPack(input_b, schedforge::MemRefType::get({16, 4}), "nr8");
    require(loop->body().operations().size() == 4, "nested loop block");
    require(loop->body().dump().find("memref.prefetch") != std::string::npos, "prefetch op");

    schedforge::OperationPassManager passes;
    passes.addPass<RenamePass>();
    passes.run(module);
    require(module.body().operations().front()->attribute("stage") == "canonical", "pass manager");
    require(module.dump().find("tensor.matmul") != std::string::npos, "generic ir dump");
}

void test_correctness_non_multiple() {
    schedforge::GraphIR graph{{31, 29, 27, true, true}};
    const auto data = schedforge::make_data(graph.problem, 11);
    const auto expected = schedforge::reference(graph.problem, data);
    schedforge::Schedule schedule;
    schedule.bm = 16; schedule.bn = 16; schedule.bk = 8;
    schedule.mr = 4; schedule.vector_width = 8; schedule.threads = 3;
    schedforge::LoopIR loop{graph.problem, schedule};
    std::vector<float> actual;
    schedforge::execute(loop, data, actual);
    require(schedforge::max_abs_error(expected, actual) < 1.0e-3, "optimized correctness");

    schedule.pack_b = true;
    schedule.pack_a = true;
    schedule.unroll_k = 4;
    schedule.mc = 24; schedule.nc = 17; schedule.kc = 11;
    schedforge::LoopIR packed_loop{graph.problem, schedule};
    schedforge::execute(packed_loop, data, actual);
    require(schedforge::max_abs_error(expected, actual) < 1.0e-3, "packed correctness");
    require(packed_loop.dump().find("pack_a,pack_b") != std::string::npos, "packing in loop ir");

    schedule.pack_a = false;
    schedule.pack_b = false;
    schedule.mr = 6;
    schedule.threads = 4;
    schedule.pin_threads = true;
    schedforge::execute({graph.problem, schedule}, data, actual);
    require(schedforge::max_abs_error(expected, actual) < 1.0e-3,
            "dual-vector kernel correctness");

    schedule.pin_threads = false;
    schedule.fused = false;
    schedforge::execute({graph.problem, schedule}, data, actual);
    require(schedforge::max_abs_error(expected, actual) < 1.0e-3,
            "thread affinity reset correctness");
}

void test_simulator_and_search() {
    schedforge::GraphIR graph{{32, 32, 32, true, true}};
    const auto data = schedforge::make_data(graph.problem, 13);
    schedforge::LoopIR loop{graph.problem, {}};
    const auto simulation = schedforge::simulate(loop);
    require(simulation.memory_accesses > 0 && simulation.estimated_cycles > 0.0, "simulation stats");
    const auto result = schedforge::autoschedule(graph, data, 2, 3, 0, 1);
    require(result.search_space > result.benchmarked, "search pruning");
    require(result.benchmark.max_error < 1.0e-3, "search correctness");
}

void test_compiler_architecture() {
    const schedforge::GraphIR graph{{17, 19, 13, true, true}};
    const auto schedule = schedforge::ScheduleDSL::parse(
        "order=ikj;outer=64,128,32;tile=16,32,16;micro=4,8;vector=8;unroll=4;threads=2;pack=b;prefetch=4;fuse=true");
    require(schedule.pack_b && schedule.prefetch_distance == 4, "schedule dsl");
    schedforge::Schedule unpacked;
    const auto round_trip = schedforge::ScheduleDSL::parse(
        schedforge::ScheduleDSL::print(unpacked));
    require(!round_trip.pack_a && !round_trip.pack_b, "empty packing round trip");
    schedforge::Compiler compiler(schedforge::TargetInfo::detect(), "build/test-kernel-cache");
    const auto compiled = compiler.compile(graph, schedule);
    require(compiled.tensor_module.dump().find("tensor.matmul") != std::string::npos, "tensor module");
    require(compiled.loop.dump().find("pack_b") != std::string::npos, "scheduled loop module");
    require(compiled.llvm_ir.find("define void @matmul") != std::string::npos, "llvm text lowering");
    require(compiled.layouts.at(1).layout == schedforge::Layout::PackedB, "layout propagation");
    require(!compiled.buffers.empty() && compiled.buffers.front().alignment == 64, "memory planning");
    require(schedforge::MemoryPlanner{}.peakBytes(compiled.buffers) > 0, "peak memory");
    const schedforge::DynamicProblem dynamic{-1, 64, -1};
    const auto specialized = dynamic.specialize(31, 99, 27);
    require(specialized.m == 31 && specialized.n == 64 && specialized.k == 27, "dynamic specialization");
    const auto pressure = schedforge::estimate_register_pressure(schedule, compiler.target());
    require(pressure.total > 0, "register pressure model");

    const auto runtime = schedforge::RuntimeInfo::detect();
    require(runtime.logical_cpus > 0 && runtime.numa_nodes > 0, "runtime topology");
    const auto assembly = schedforge::AssemblyAnalyzer{}.analyze(
        "vfmadd231ps %ymm0, %ymm1, %ymm2\nvmovups %ymm2, 32(%rsp)\n");
    require(assembly.fma_instructions == 1 && assembly.has_spill_pattern, "assembly analysis");

    schedforge::CostModel cost_model;
    const auto simulation = schedforge::simulate(compiled.loop, compiler.target());
    const auto before = cost_model.evaluate(compiled.loop, simulation, compiler.target()).total_cycles;
    cost_model.calibrate(before, 1000.0);
    require(cost_model.calibrationFactor() > 0.0, "cost calibration");

    schedforge::KernelCache cache("build/test-explicit-cache");
    const auto key = cache.key(graph.problem, schedule, compiler.target());
    cache.store(key, schedule);
    const auto cached = cache.lookup(key);
    require(cached.has_value() && cached->pack_b, "kernel cache");

    const schedforge::GraphIR jit_graph{{9, 11, 7, true, true}};
    const auto jit_data = schedforge::make_data(jit_graph.problem, 23);
    const auto jit_result = schedforge::LLVMJITBackend{}.benchmark(jit_graph, jit_data, schedule, 0, 1);
    require(jit_result.max_error < 1.0e-3 && jit_result.llvm_ir.find("<8 x float>") != std::string::npos &&
            !jit_result.assembly.empty(),
            "llvm orc jit");
    const auto jit_cached = schedforge::LLVMJITBackend{}.benchmark(jit_graph, jit_data, schedule, 0, 1);
    require(jit_cached.compile_milliseconds < jit_result.compile_milliseconds, "jit kernel cache");

    schedforge::Schedule register_schedule;
    register_schedule.mr = 4;
    register_schedule.nr = 8;
    register_schedule.vector_width = 8;
    const schedforge::GraphIR register_graph{{8, 16, 8, true, false}};
    const auto register_data = schedforge::make_data(register_graph.problem, 29);
    const auto register_jit = schedforge::LLVMJITBackend{}.benchmark(
        register_graph, register_data, register_schedule, 0, 1);
    require(register_jit.max_error < 1.0e-3 &&
            register_jit.assembly_report.fma_instructions > 0 &&
            !register_jit.assembly_report.has_spill_pattern,
            "llvm generated register microkernel");

    const std::vector<float> dtype_values{-1.25F, 0.0F, 0.75F, 3.5F};
    const auto bf16 = schedforge::convert_from_bf16(schedforge::convert_to_bf16(dtype_values));
    require(schedforge::max_abs_error(dtype_values, bf16) < 0.02, "bf16 conversion");
    const auto int8 = schedforge::dequantize_int8(schedforge::quantize_int8(dtype_values));
    require(schedforge::max_abs_error(dtype_values, int8) < 0.04, "int8 quantization");
    const auto bf16_benchmark = schedforge::benchmark_bf16(jit_graph.problem, jit_data, 1);
    const auto int8_benchmark = schedforge::benchmark_int8(jit_graph.problem, jit_data, 1);
    require(bf16_benchmark.max_error < 0.05 && int8_benchmark.max_error < 0.2, "dtype kernels");

    const auto calibration = schedforge::HardwareCalibrator{}.run(1024, 1);
    require(calibration.memory_bandwidth_gbps > 0.0, "hardware calibration");
    const auto search = schedforge::compare_search_strategy(jit_graph, jit_data,
        schedforge::SearchStrategy::Random, 1, 2, 3);
    require(search.hardware_measurements == 2 && search.best_gflops > 0.0, "search strategies");
}

void test_graph_compiler() {
    const schedforge::MLPConfig config{1, 4, 8, 16};
    auto graph = schedforge::build_transformer_mlp_graph(config, true);
    const auto constraints = schedforge::ShapeInferencePass{}.run(graph);
    require(graph.operations().size() == 12, "multi-op tensor graph");
    require(!constraints.empty() && constraints.front().runtime_guard, "symbolic shape constraint");

    const auto structured = schedforge::StructuredComputeLowering{}.run(graph);
    require(structured.size() >= 4 && structured.front().dump().find("reduction") != std::string::npos,
            "structured tensor compute");

    auto dispatches = schedforge::FusionPlanner{}.run(graph);
    require(dispatches.size() == 2, "mlp dispatch formation");
    require(dispatches.front().operations.size() == 3, "matmul bias gelu fusion");
    require(dispatches.back().operations.size() == 3, "matmul bias residual fusion");

    auto static_graph = schedforge::build_transformer_mlp_graph(config, false);
    schedforge::ShapeInferencePass{}.run(static_graph);
    auto static_dispatches = schedforge::FusionPlanner{}.run(static_graph);
    const auto removed = schedforge::GraphLayoutPlanner{}.run(static_graph, static_dispatches);
    const auto memory = schedforge::GraphBufferizer{}.run(static_graph, static_dispatches);
    require(removed == 1, "layout propagation");
    require(memory.workspace_bytes < memory.naive_bytes && memory.workspace_bytes > 0,
            "dispatch boundary workspace reuse");
    require(memory.buffers.size() < static_graph.values().size(), "fused values stay virtual");

    const auto data = schedforge::make_mlp_data(config, 31);
    schedforge::GraphCompileOptions options;
    options.max_threads = 2;
    const auto plan = schedforge::GraphCompiler{}.compile(graph, options, &config, &data);
    require(plan.dispatches.size() == 2 && plan.llvm_ir.size() == 2 &&
            plan.llvm_ir.front().find("llvm.fma") != std::string::npos,
            "graph llvm kernel compilation");
    require(plan.layout_conversions_removed == 1 &&
            plan.llvm_compile_milliseconds >= 0.0 && !plan.hardware.empty(),
            "graph compilation statistics");
    require(plan.dump().find("shape_guards") != std::string::npos, "executable plan");
    const auto measured = schedforge::execute_mlp(plan, config, data, 0, 1);
    require(measured.max_error < 1.0e-3, "mlp runtime correctness");
    require(!plan.dispatches.front().transforms.operations().empty() &&
            plan.dispatches.front().intrinsic.has_value(), "transform ir tensorization");
    const auto replayed = schedforge::TransformProgram::parse(
        plan.dispatches.front().transforms.dump()).replay();
    require(replayed.mr == plan.dispatches.front().schedule.mr &&
            replayed.nr == plan.dispatches.front().schedule.nr,
            "transform ir replay");

    auto quantized = schedforge::build_transformer_mlp_graph(config);
    quantized.values()[0].type.dtype = schedforge::DataType::I8;
    quantized.values()[0].type.quant_scale = 0.03125F;
    const auto quantized_ops = schedforge::QuantizationPropagationPass{}.run(quantized);
    require(quantized_ops > 0, "quantization propagation");

    auto attention = schedforge::build_mini_attention_graph(8, 16, true);
    const auto attention_constraints = schedforge::ShapeInferencePass{}.run(attention);
    require(attention.operations().size() >= 15 && !attention_constraints.empty(),
            "mini attention graph");

    schedforge::MeasurementDatabase database;
    schedforge::Schedule learned_schedule;
    learned_schedule.mr = 4;
    learned_schedule.nr = 8;
    learned_schedule.vector_width = 8;
    const schedforge::LoopIR learned_loop{{8, 8, 8, true, true}, learned_schedule};
    const auto simulated = schedforge::simulate(learned_loop);
    database.add(schedforge::ScheduleMeasurement{
        {8, 8, 8, true, true}, learned_schedule, simulated, 0.01});
    database.add(schedforge::ScheduleMeasurement{
        {16, 16, 16, true, true}, learned_schedule, simulated, 0.03});
    database.saveCsv("build/test-measurements.csv");
    const auto loaded = schedforge::MeasurementDatabase::loadCsv("build/test-measurements.csv");
    schedforge::LearnedCostModel learned;
    learned.fit(loaded);
    require(learned.trained() && std::isfinite(learned.predictMilliseconds(
        {8, 8, 8, true, true}, learned_schedule, simulated)), "learned cost model");

    const auto imported = schedforge::StableHLOImporter{}.importText(
        "func.func @main(%x: tensor<4x8xf32>, %w: tensor<8x16xf32>, %b: tensor<16xf32>) {\n"
        "  %0 = stablehlo.dot_general %x, %w : tensor<4x16xf32>\n"
        "  %1 = stablehlo.add %0, %b : tensor<4x16xf32>\n"
        "  return %1\n}\n");
    require(imported.operations().size() >= 6 && imported.returnValue() >= 0,
            "stablehlo subset importer");
    bool rejected_unsupported = false;
    try {
        schedforge::StableHLOImporter{}.importText(
            "func.func @main(%x: tensor<4x8xf32>) {\n"
            "  %0 = stablehlo.sine %x : tensor<4x8xf32>\n"
            "  return %0\n}\n");
    } catch (const std::invalid_argument&) {
        rejected_unsupported = true;
    }
    require(rejected_unsupported, "reject unsupported stablehlo operations");
}

}  // namespace

int main() {
    try {
        test_ir_passes();
        test_ssa_ir_and_pass_manager();
        test_correctness_non_multiple();
        test_simulator_and_search();
        test_compiler_architecture();
        test_graph_compiler();
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
