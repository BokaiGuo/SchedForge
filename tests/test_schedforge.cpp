#include "schedforge/schedforge.h"
#include "schedforge/ir.h"
#include "schedforge/compiler.h"
#include "schedforge/graph_compiler.h"
#include "schedforge/moe_compiler.h"
#include "schedforge/attention_compiler.h"
#include "schedforge/decoder_compiler.h"
#include "schedforge/next_milestones.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_ir_passes() {
    schedforge::GraphIR graph{{17, 19, 13, true, true}};
    schedforge::LowerToLoopsPass lower;
    auto loop = lower.run(graph);
    require(schedforge::analyze_loop_ir(loop).order == schedforge::LoopOrder::IJK,
            "lowering order");
    schedforge::LoopInterchangePass{}.run(loop);
    require(schedforge::analyze_loop_ir(loop).order == schedforge::LoopOrder::IKJ,
            "interchange");
    schedforge::LoopTilingPass{}.run(loop, 8, 16, 4);
    schedforge::VectorizePass{}.run(loop, 8);
    schedforge::FusionPass{}.run(loop);
    const auto execution = schedforge::analyze_loop_ir(loop);
    require(execution.tiled && execution.bn == 16, "tiling");
    require(execution.vector_width == 8 && execution.fused, "vector/fusion");
    const auto text = loop.dump();
    require(text.find("scf.for %ii") != std::string::npos &&
            text.find("vector.accumulator.init") != std::string::npos &&
            text.find("vector.load") != std::string::npos &&
            text.find("vector.fma") != std::string::npos &&
            text.find("vector.store") != std::string::npos,
            "explicit scheduled loop operations");
    require(text.find("vector.fma") < text.find("epilogue.add_bias") &&
            text.find("epilogue.add_bias") < text.find("vector.store"),
            "reduction epilogue store scope");
    schedforge::LoopIR invalid;
    invalid.problem = graph.problem;
    bool rejected_invalid = false;
    try {
        schedforge::verify_loop_ir(invalid);
    } catch (const std::invalid_argument&) {
        rejected_invalid = true;
    }
    require(rejected_invalid, "loop ir verifier");
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
    require(packed_loop.dump().find("buffer.pack %A -> %packedA") != std::string::npos &&
            packed_loop.dump().find("buffer.pack %B -> %packedB") != std::string::npos,
            "packing in loop ir");

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
    require(simulation.memory_accesses > 0 && simulation.estimated_cycles > 0.0 &&
            simulation.sampled_m == graph.problem.m && simulation.sampled_n == graph.problem.n &&
            simulation.sampled_k == graph.problem.k, "full-size simulation stats");
    const auto sampled = schedforge::simulate(loop, schedforge::TargetInfo::detect(), {16});
    require(sampled.sampled_m == 16 && sampled.sampled_n == 16 && sampled.sampled_k == 16,
            "explicit simulation sampling");
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
    require(compiled.loop.dump().find("buffer.pack %B -> %packedB") != std::string::npos &&
            compiled.loop.dump().find("memref.prefetch") != std::string::npos &&
            compiled.loop.dump().find("vector.fma") != std::string::npos,
            "scheduled loop module");
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
    const auto jit_result = schedforge::LLVMJITBackend{}.benchmark(
        schedforge::apply_schedule(jit_graph.problem, schedule), jit_data, 0, 1);
    require(jit_result.max_error < 1.0e-3 && jit_result.llvm_ir.find("<8 x float>") != std::string::npos &&
            !jit_result.assembly.empty(),
            "llvm orc jit");
    std::vector<float> native_output;
    const auto shared_loop = schedforge::apply_schedule(jit_graph.problem, schedule);
    schedforge::execute(shared_loop, jit_data, native_output);
    require(schedforge::max_abs_error(native_output, schedforge::reference(jit_graph.problem, jit_data)) < 1.0e-3 &&
            schedforge::simulate(shared_loop, compiler.target()).memory_accesses > 0,
            "single loop ir native simulator agreement");
    const auto jit_cached = schedforge::LLVMJITBackend{}.benchmark(
        schedforge::apply_schedule(jit_graph.problem, schedule), jit_data, 0, 1);
    require(jit_cached.compile_milliseconds < jit_result.compile_milliseconds, "jit kernel cache");

    auto parallel_schedule = schedule;
    parallel_schedule.threads = 3;
    const schedforge::Problem parallel_problem{17, 19, 13, true, true};
    const auto parallel_data = schedforge::make_data(parallel_problem, 27);
    const auto parallel_jit = schedforge::LLVMJITBackend{}.benchmark(
        schedforge::apply_schedule(parallel_problem, parallel_schedule),
        parallel_data, 1, 2);
    require(parallel_jit.threads == 3 && parallel_jit.max_error < 1.0e-3 &&
            parallel_jit.assembly_report.instructions > 0 &&
            parallel_jit.assembly_report.branches > 0,
            "parallel llvm loopir execution and assembly metrics");

    schedforge::Schedule register_schedule;
    register_schedule.mr = 4;
    register_schedule.nr = 8;
    register_schedule.vector_width = 8;
    const schedforge::GraphIR register_graph{{8, 16, 8, true, false}};
    const auto register_data = schedforge::make_data(register_graph.problem, 29);
    const auto register_jit = schedforge::LLVMJITBackend{}.benchmark(
        schedforge::apply_schedule(register_graph.problem, register_schedule), register_data, 0, 1);
    require(register_jit.max_error < 1.0e-3 &&
            register_jit.assembly_report.fma_instructions > 0 &&
            !register_jit.assembly_report.has_spill_pattern,
            "llvm generated register microkernel");

    auto graph_epilogue_loop = schedforge::apply_schedule(
        {8, 16, 8, true, false}, register_schedule);
    schedforge::LoopOperation gelu;
    gelu.kind = schedforge::LoopOpKind::Gelu;
    graph_epilogue_loop.insertEpilogueBeforeStores(std::move(gelu));
    schedforge::LoopOperation residual;
    residual.kind = schedforge::LoopOpKind::AddResidual;
    graph_epilogue_loop.insertEpilogueBeforeStores(std::move(residual));
    auto graph_epilogue_data = register_data;
    graph_epilogue_data.residual.resize(8 * 16, 0.125F);
    const auto graph_epilogue_jit = schedforge::LLVMJITBackend{}.benchmark(
        graph_epilogue_loop, graph_epilogue_data, 0, 1);
    require(graph_epilogue_jit.max_error < 1.0e-3 &&
            graph_epilogue_jit.llvm_ir.find("llvm.exp") != std::string::npos &&
            graph_epilogue_jit.llvm_ir.find("residual") != std::string::npos,
            "llvm consumes explicit graph epilogues");

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

void test_aot_deployment() {
    schedforge::Schedule schedule;
    schedule.mr = 4;
    schedule.nr = 8;
    schedule.vector_width = 8;
    schedule.threads = 1;
    const schedforge::Problem problem{13, 17, 11, true, true};
    const auto loop = schedforge::apply_schedule(problem, schedule);
    const auto object = schedforge::LLVMAOTBackend{}.compile(loop);
    require(object.object_code.size() > 4 && object.object_code[0] == 0x7f &&
            object.object_code[1] == 'E' && object.object_code[2] == 'L' &&
            object.object_code[3] == 'F' &&
            object.llvm_ir.find("schedforge_matmul_v1") != std::string::npos,
            "llvm aot elf object");

    const auto package = std::filesystem::temp_directory_path() / "schedforge_test_aot.sfe";
    std::filesystem::remove_all(package);
    const auto created = schedforge::create_aot_package(loop, package);
    const auto inspected = schedforge::inspect_aot_package(package);
    require(created.manifest.object_checksum == inspected.object_checksum &&
            inspected.problem.m == problem.m &&
            std::filesystem::file_size(package / "kernel.so") > 0,
            "aot package round trip");
    const auto measured = schedforge::benchmark_aot_package(
        package, schedforge::make_data(problem, 103), 0, 2);
    require(measured.max_error < 1.0e-3 && measured.gflops > 0.0,
            "aot dlopen execution");

    {
        std::ofstream corrupt(package / "kernel.so", std::ios::binary | std::ios::app);
        corrupt.put('\0');
    }
    bool rejected = false;
    try {
        (void)schedforge::inspect_aot_package(package);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "aot checksum rejects tampering");
    std::filesystem::remove_all(package);

    (void)schedforge::create_aot_package(loop, package);
    const auto manifest_path = package / "manifest.sfe";
    std::ifstream manifest_input(manifest_path);
    std::string manifest_text((std::istreambuf_iterator<char>(manifest_input)),
                              std::istreambuf_iterator<char>());
    const auto cpu_field = manifest_text.find("target_cpu=");
    const auto cpu_end = manifest_text.find('\n', cpu_field);
    manifest_text.replace(cpu_field, cpu_end - cpu_field,
                          "target_cpu=incompatible-test-cpu");
    std::ofstream(manifest_path, std::ios::trunc) << manifest_text;
    rejected = false;
    try {
        (void)schedforge::inspect_aot_package(package);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "aot target guard rejects incompatible cpu");
    std::filesystem::remove_all(package);

    (void)schedforge::create_aot_package(loop, package);
    std::ifstream shape_manifest_input(manifest_path);
    manifest_text.assign(std::istreambuf_iterator<char>(shape_manifest_input),
                         std::istreambuf_iterator<char>());
    const auto shape_field = manifest_text.find("m=13");
    manifest_text.replace(shape_field, 4, "m=14");
    std::ofstream(manifest_path, std::ios::trunc) << manifest_text;
    rejected = false;
    try {
        (void)schedforge::inspect_aot_package(package);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "aot embedded metadata rejects manifest shape tampering");
    std::filesystem::remove_all(package);

    auto parallel = schedule;
    parallel.threads = 2;
    rejected = false;
    try {
        (void)schedforge::LLVMAOTBackend{}.compile(
            schedforge::apply_schedule(problem, parallel));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "aot v1 rejects unsupported thread dispatch");

    auto residual_loop = loop;
    schedforge::LoopOperation residual;
    residual.kind = schedforge::LoopOpKind::AddResidual;
    residual_loop.insertEpilogueBeforeStores(std::move(residual));
    rejected = false;
    try {
        (void)schedforge::LLVMAOTBackend{}.compile(residual_loop);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "aot v1 rejects unmanifested graph epilogues");
}

void test_next_milestones() {
    schedforge::PagedKVConfig paged_config{1, 2, 4, 4, 4, 12};
    auto paged = schedforge::make_paged_kv_cache(paged_config);
    std::vector<float> keys(2 * 7 * 4), values(2 * 7 * 4);
    std::iota(keys.begin(), keys.end(), 0.0F);
    std::iota(values.begin(), values.end(), 1.0F);
    schedforge::append_paged_kv(paged, keys, values, 7);
    require(paged.page_table.size() == 2 && paged.length == 7,
            "paged kv allocation and append");
    const auto flat = schedforge::gather_paged_kv(paged);
    require(flat.length == 7 && flat.keys == keys && flat.values == values,
            "paged kv gather preserves logical order");
    bool rejected = false;
    try {
        schedforge::release_paged_kv_pages(paged, {paged.page_table.front()});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "paged kv active page guard");
    const auto pages_before_truncate = paged.page_table.size();
    schedforge::truncate_paged_kv(paged, 3);
    require(paged.length == 3 && paged.page_table.size() < pages_before_truncate &&
            !paged.free_pages.empty(), "paged kv truncate releases tail pages");

    const schedforge::AttentionConfig paged_attention_config{
        1, 4, 2, 1, 7, 4, 4, true, 0.0F};
    schedforge::AttentionPlanOptions paged_options;
    paged_options.strategy = schedforge::AttentionLoweringStrategy::SplitKVDecode;
    const auto paged_plan = schedforge::AttentionCompiler{}.compile(
        paged_attention_config, paged_options);
    std::vector<float> decode_query(16, 0.125F);
    auto full_paged = schedforge::make_paged_kv_cache(paged_config);
    schedforge::append_paged_kv(full_paged, keys, values, 7);
    const auto paged_result = schedforge::execute_paged_decode_attention(
        paged_plan, decode_query, full_paged, 0, 2, true);
    auto contiguous = schedforge::make_kv_cache(paged_attention_config, 7);
    schedforge::append_kv(contiguous, keys, values, 7);
    const auto contiguous_result = schedforge::execute_decode_attention(
        paged_plan, decode_query, contiguous, 0, 2, false);
    require(paged_result.max_error < 1.0e-5 &&
            schedforge::max_abs_error(paged_result.output, contiguous_result.output) < 1.0e-5,
            "paged attention direct traversal agrees with contiguous decode");

    schedforge::QuantizedMatMulConfig quant_config{7, 9, 5, true, true, true};
    std::vector<float> input(35, 0.25F), weights(45, 0.2F), bias(9, 0.1F);
    const auto quantized = schedforge::quantize_matmul_weights(weights, quant_config);
    const auto quantized_result = schedforge::execute_quantized_matmul(
        quant_config, input, quantized, bias, 0, 2);
    require(quantized_result.max_error < 0.01 && quantized_result.gflops > 0.0,
            "int8 per-channel matmul");
    quant_config.per_channel = false;
    const auto tensor_quantized = schedforge::quantize_matmul_weights(weights, quant_config);
    require(tensor_quantized.scales.size() == 1 && tensor_quantized.scales.front() < 1.0F,
            "int8 per-tensor scale");

    std::vector<std::uint8_t> transfer(1U << 20U, 7);
    const auto schedule = schedforge::tune_transfer_schedule(transfer, 2, 1);
    const auto transfer_result = schedforge::benchmark_transfer(transfer, schedule, 2);
    require(transfer_result.bytes == transfer.size() &&
            transfer_result.gigabytes_per_second > 0.0,
            "transfer tuning executes and validates");

    const auto neon = schedforge::inspect_neon_codegen();
    require(neon.source.find("arm_neon.h") != std::string::npos &&
            neon.source.find("vfmaq") != std::string::npos &&
            neon.source.find("b + kk * 4") != std::string::npos,
            "neon source generation");
    require(neon.runtime_max_error < 1.0e-6 &&
            (neon.runtime_succeeded || !neon.host_supports_neon),
            "neon runtime correctness boundary");
    const auto fuzz = schedforge::run_schedforge_fuzz(19, 128);
    require(fuzz.failures == 0 && fuzz.passed > 0, "compiler fuzz invariants");
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
    require(plan.dispatches.size() == 2 && plan.scheduled_loops.size() == 2 &&
            plan.llvm_ir.size() == 2 &&
            plan.llvm_ir.front().find("llvm.fma") != std::string::npos,
            "graph llvm kernel compilation");
    require(plan.scheduled_loops.front().dump().find("vector.accumulator.init") != std::string::npos &&
            plan.scheduled_loops.front().dump().find("epilogue.gelu") != std::string::npos &&
            plan.scheduled_loops.back().dump().find("epilogue.add_residual") != std::string::npos &&
            plan.dump().find("scheduled_loop @dispatch_0") != std::string::npos,
            "graph executable embeds scheduled loop ir");
    const auto first_loop_dump = plan.scheduled_loops.front().dump();
    const auto second_loop_dump = plan.scheduled_loops.back().dump();
    require(first_loop_dump.find("vector.fma") < first_loop_dump.find("epilogue.gelu") &&
            first_loop_dump.find("epilogue.gelu") < first_loop_dump.find("vector.store") &&
            second_loop_dump.find("vector.fma") < second_loop_dump.find("epilogue.add_residual") &&
            second_loop_dump.find("epilogue.add_residual") < second_loop_dump.find("vector.store"),
            "graph epilogues execute before stores");
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
    const auto transformed_loop = schedforge::TransformProgram::parse(
        plan.dispatches.front().transforms.dump()).apply(
            plan.dispatches.front().kernel_problem);
    const auto transformed_execution = schedforge::analyze_loop_ir(transformed_loop);
    require(transformed_execution.bm == plan.dispatches.front().schedule.bm &&
            transformed_execution.vector_width == plan.dispatches.front().schedule.vector_width,
            "transform ir directly lowers loop ir");

    const auto dynamic_plan = schedforge::GraphCompiler{}.compile(
        schedforge::build_transformer_mlp_graph(config, true), options);
    const auto specialized_plan = dynamic_plan.specializeMLP(config, &data);
    require(specialized_plan.scheduled_loops.size() == 2 &&
            specialized_plan.scheduled_loops.front().problem.m == config.batch * config.sequence &&
            specialized_plan.llvm_ir.size() == 2 &&
            specialized_plan.guards.size() == 1,
            "dynamic executable specialization");
    const auto specialized_result = schedforge::execute_mlp(
        specialized_plan, config, data, 0, 1);
    require(specialized_result.max_error < 1.0e-3,
            "specialized executable runtime correctness");

    auto quantized = schedforge::build_transformer_mlp_graph(config);
    quantized.values()[0].type.dtype = schedforge::DataType::I8;
    quantized.values()[0].type.quant_scale = 0.03125F;
    const auto quantized_ops = schedforge::QuantizationPropagationPass{}.run(quantized);
    require(quantized_ops > 0, "quantization propagation");

    auto attention = schedforge::build_mini_attention_graph(8, 16, true);
    const auto attention_constraints = schedforge::ShapeInferencePass{}.run(attention);
    require(attention.operations().size() >= 15 && !attention_constraints.empty(),
            "mini attention graph");
    schedforge::TensorGraph fused_attention;
    const auto attention_type = schedforge::GraphTensorType{{
        schedforge::Dimension::fixed(1), schedforge::Dimension::fixed(4),
        schedforge::Dimension::fixed(8), schedforge::Dimension::fixed(16)}};
    const int query = fused_attention.addInput("q", attention_type);
    const int key = fused_attention.addInput("k", attention_type);
    const int value = fused_attention.addInput("v", attention_type);
    const int sdpa = fused_attention.addOperation(
        schedforge::GraphOpKind::AttentionSdpa, "sdpa", {query, key, value});
    fused_attention.setReturn(sdpa);
    schedforge::ShapeInferencePass{}.run(fused_attention);
    const auto attention_compute = schedforge::StructuredComputeLowering{}.run(fused_attention);
    require(attention_compute.size() == 1 && attention_compute.front().iterators.size() == 6 &&
            attention_compute.front().body.find("attention.qk") != std::string::npos,
            "structured attention compute lowering");

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

    schedforge::MeasurementDatabase tuning_database;
    schedforge::Schedule measured_first = plan.dispatches.front().schedule;
    measured_first.bm = 16;
    measured_first.threads = 1;
    tuning_database.add({plan.dispatches.front().kernel_problem, measured_first,
                         schedforge::simulate(schedforge::LoopIR{
                             plan.dispatches.front().kernel_problem, measured_first}), 0.005});
    schedforge::Schedule measured_second = plan.dispatches.back().schedule;
    measured_second.bm = 16;
    measured_second.threads = 1;
    tuning_database.add({plan.dispatches.back().kernel_problem, measured_second,
                         schedforge::simulate(schedforge::LoopIR{
                             plan.dispatches.back().kernel_problem, measured_second}), 0.006});
    tuning_database.saveCsv("build/test-tuning-database.csv");
    auto database_options = options;
    database_options.measurement_database = "build/test-tuning-database.csv";
    const auto database_plan = schedforge::GraphCompiler{}.compile(
        schedforge::build_transformer_mlp_graph(config), database_options);
    require(database_plan.dispatches.front().schedule.bm == 16 &&
            database_plan.dispatches.front().tuning_source == "measurement_database" &&
            database_plan.dump().find("tuning<source=measurement_database") != std::string::npos,
            "measurement database drives transform selection");

    const auto imported = schedforge::StableHLOImporter{}.importText(
        "func.func @main(%x: tensor<4x8xf32>, %w: tensor<8x16xf32>, %b: tensor<16xf32>) {\n"
        "  %0 = stablehlo.dot_general %x, %w : tensor<4x16xf32>\n"
        "  %1 = stablehlo.add %0, %b : tensor<4x16xf32>\n"
        "  return %1\n}\n");
    require(imported.operations().size() >= 6 && imported.returnValue() >= 0,
            "stablehlo subset importer");
    const auto imported_constant = schedforge::StableHLOImporter{}.importText(
        "func.func @main(%x: tensor<4x8xf32>) {\n"
        "  %c = stablehlo.constant dense<1.0> : tensor<4x8xf32>\n"
        "  %0 = stablehlo.add %x, %c : tensor<4x8xf32>\n"
        "  return %0\n}\n");
    require(imported_constant.dump().find("tensor.constant") != std::string::npos &&
            imported_constant.dump().find("value=dense<1.0>") != std::string::npos,
            "stablehlo constant importer");
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

void test_moe_compiler() {
    const schedforge::MoeConfig config{7, 8, 12, 4, 2};
    const auto graph = schedforge::build_moe_mlp_graph(config, true);
    require(graph.operations().size() >= 17 &&
            graph.dump().find("moe.grouped_matmul") != std::string::npos &&
            graph.dump().find("moe.combine") != std::string::npos,
            "moe tensor graph decomposition");
    const auto program = schedforge::lower_moe_program(config);
    require(program.operations.size() == 11 &&
            program.dump().find("moe.histogram") != std::string::npos &&
            program.dump().find("tensor.swiglu") != std::string::npos,
            "moe routing program lowering");

    const auto data = schedforge::make_moe_data(config, 41);
    const auto routing = schedforge::route_topk(config, data);
    require(std::accumulate(routing.counts.begin(), routing.counts.end(), 0) ==
                config.tokens * config.top_k,
            "top-k routing assignment count");
    for (int token = 0; token < config.tokens; ++token) {
        float weight_sum = 0.0F;
        for (int slot = 0; slot < config.top_k; ++slot)
            weight_sum += routing.expert_weights[
                static_cast<std::size_t>(token) * config.top_k + slot];
        require(std::abs(weight_sum - 1.0F) < 1.0e-5F, "top-k normalized weights");
    }
    const auto segmented = schedforge::dispatch_tokens(config, data, routing);
    require(segmented.offsets.size() == static_cast<std::size_t>(config.experts + 1) &&
            segmented.offsets.back() == config.tokens * config.top_k &&
            segmented.values.size() == static_cast<std::size_t>(config.tokens * config.top_k * config.hidden),
            "segmented token dispatch");

    schedforge::MoeExecutionSchedule schedule;
    schedule.threads = 3;
    schedule.split_threshold = 2;
    schedule.token_buckets = {2, 4, 8};
    const auto tasks = schedforge::plan_moe_tasks(routing, schedule);
    require(tasks.size() >= static_cast<std::size_t>(config.experts) &&
            std::all_of(tasks.begin(), tasks.end(), [](const auto& task) {
                return task.end > task.begin && task.end - task.begin <= 2;
            }), "load-aware expert splitting");

    const auto uniform = schedforge::make_routing_trace(
        {64, 8, 16, 8, 2}, schedforge::RoutingDistribution::Uniform, 5);
    const auto skewed = schedforge::make_routing_trace(
        {64, 8, 16, 8, 2}, schedforge::RoutingDistribution::HeavySkew, 5);
    const auto uniform_simulation = schedforge::simulate_moe(
        {64, 8, 16, 8, 2}, uniform, schedule);
    const auto skewed_simulation = schedforge::simulate_moe(
        {64, 8, 16, 8, 2}, skewed, schedule);
    require(skewed_simulation.expert_counts.front() > uniform_simulation.expert_counts.front() &&
            skewed_simulation.dispatch_bytes > 0.0 && skewed_simulation.weight_bytes > 0.0,
            "routing skew simulation");
    const auto selected = schedforge::select_moe_schedule(
        {64, 8, 16, 8, 2}, skewed, schedforge::TargetInfo::detect(), 4);
    require(selected.threads == 4 &&
            selected.strategy != schedforge::MoeExecutionStrategy::IndependentExperts,
            "routing-aware moe strategy selection");

    const auto plan = schedforge::MoeCompiler{}.compile(config, schedule);
    require(plan.kernels.size() == 3 &&
            plan.dump().find("!sfg.segmented_tensor") != std::string::npos &&
            plan.dump().find("moe.schedule") != std::string::npos &&
            plan.dump().find("T <= 7") != std::string::npos &&
            plan.memory.workspace_bytes < plan.memory.naive_bytes,
            "moe executable plan");
    const auto measured = schedforge::execute_moe(plan, data, routing, 0, 1);
    require(measured.max_error < 1.0e-3 &&
            measured.p95_milliseconds >= measured.p50_milliseconds &&
            measured.output.size() == static_cast<std::size_t>(config.tokens * config.hidden),
            "moe end-to-end execution");
}

void test_attention_compiler() {
    const schedforge::AttentionConfig config{1, 4, 2, 9, 9, 8, 8, true};
    auto graph = schedforge::build_sdpa_graph(config, true);
    const auto unfused_constraints = schedforge::ShapeInferencePass{}.run(graph);
    require(graph.dump().find("tensor.softmax") != std::string::npos &&
            graph.dump().find("tensor.mask") != std::string::npos &&
            !unfused_constraints.empty(),
            "unfused sdpa tensor graph");
    auto fused = schedforge::AttentionFusionPass{}.run(graph);
    require(fused.dump().find("attention.sdpa") != std::string::npos,
            "attention fusion pass");
    const auto constraints = schedforge::ShapeInferencePass{}.run(fused);
    require(!constraints.empty() &&
            fused.values().at(static_cast<std::size_t>(fused.returnValue())).type.shape.size() == 4,
            "attention shape inference");

    const auto data = schedforge::make_attention_data(config, 43);
    schedforge::AttentionSchedule schedule;
    schedule.query_tile = 4;
    schedule.kv_tile = 3;
    schedule.threads = 2;
    const auto materialized = schedforge::AttentionCompiler{}.compile(
        config, {schedforge::AttentionLoweringStrategy::Materialized, schedule});
    const auto io_aware = schedforge::AttentionCompiler{}.compile(
        config, {schedforge::AttentionLoweringStrategy::IOAware, schedule});
    const auto baseline = schedforge::execute_attention(materialized, data, 0, 1);
    const auto streaming = schedforge::execute_attention(io_aware, data, 0, 1);
    require(baseline.max_error < 1.0e-4 && streaming.max_error < 1.0e-4 &&
            schedforge::max_abs_error(baseline.output, streaming.output) < 1.0e-4,
            "exact io-aware online softmax");
    require(io_aware.memory.temporary_bytes < materialized.memory.temporary_bytes &&
            io_aware.memory.temporary_bytes == io_aware.simulation.temporary_bytes &&
            io_aware.memory.temporary_bytes > 0 &&
            io_aware.simulation.estimated_l2_traffic > io_aware.simulation.bytes_read &&
            io_aware.pipeline.dump().find("attention.online_softmax") != std::string::npos &&
            io_aware.pipeline.dump().find("reduce.max") != std::string::npos,
            "attention tile pipeline and memory reduction");
    auto dynamic_config = config;
    dynamic_config.sequence_query = 5;
    dynamic_config.sequence_kv = 7;
    const auto dynamic_data = schedforge::make_attention_data(dynamic_config, 45);
    const auto dynamic_result = schedforge::execute_attention(
        io_aware, dynamic_data, 5, 7, 0, 1);
    require(dynamic_result.max_error < 1.0e-4,
            "dynamic attention sequence specialization");

    auto prefill_config = config;
    prefill_config.sequence_query = 128;
    prefill_config.sequence_kv = 128;
    const auto selected = schedforge::select_attention_plan(
        prefill_config, schedforge::TargetInfo::detect(), 4);
    require(selected.strategy == schedforge::AttentionLoweringStrategy::AutoScheduledIOAware,
            "prefill attention strategy selection");
    auto small_config = config;
    small_config.sequence_query = 8;
    small_config.sequence_kv = 8;
    small_config.causal = false;
    const auto small_selected = schedforge::select_attention_plan(
        small_config, schedforge::TargetInfo::detect(), 4);
    require(small_selected.strategy == schedforge::AttentionLoweringStrategy::Materialized,
            "small attention materialized selection");
    const auto small_data = schedforge::make_attention_data(small_config, 46);
    auto small_io_options = small_selected;
    small_io_options.strategy = schedforge::AttentionLoweringStrategy::IOAware;
    small_io_options.schedule.query_tile = 4;
    small_io_options.schedule.kv_tile = 4;
    const auto small_io_plan = schedforge::AttentionCompiler{}.compile(
        small_config, small_io_options);
    require(schedforge::execute_attention(small_io_plan, small_data, 0, 1).max_error < 1.0e-4,
            "non-causal io-aware attention execution");

    schedforge::AttentionConfig decode_config{1, 4, 2, 1, 11, 8, 8, true};
    const auto decode_data = schedforge::make_attention_data(decode_config, 47);
    auto cache = schedforge::make_kv_cache(decode_config);
    const int first_tokens = 5;
    std::vector<float> first_keys;
    std::vector<float> first_values;
    std::vector<float> second_keys;
    std::vector<float> second_values;
    for (int head = 0; head < decode_config.kv_heads; ++head) {
        const auto key_head = decode_data.key.begin() +
            static_cast<std::ptrdiff_t>(head * decode_config.sequence_kv * decode_config.head_dim);
        first_keys.insert(first_keys.end(), key_head,
            key_head + first_tokens * decode_config.head_dim);
        second_keys.insert(second_keys.end(), key_head + first_tokens * decode_config.head_dim,
            key_head + decode_config.sequence_kv * decode_config.head_dim);
        const auto value_head = decode_data.value.begin() +
            static_cast<std::ptrdiff_t>(head * decode_config.sequence_kv * decode_config.head_dim_value);
        first_values.insert(first_values.end(), value_head,
            value_head + first_tokens * decode_config.head_dim_value);
        second_values.insert(second_values.end(), value_head + first_tokens * decode_config.head_dim_value,
            value_head + decode_config.sequence_kv * decode_config.head_dim_value);
    }
    schedforge::append_kv(cache, first_keys, first_values, first_tokens);
    schedforge::append_kv(cache, second_keys, second_values,
                         decode_config.sequence_kv - first_tokens);
    const auto decode_plan = schedforge::AttentionCompiler{}.compile(
        decode_config, schedforge::select_attention_plan(
            decode_config, schedforge::TargetInfo::detect(), 4));
    require(decode_plan.plan.strategy == schedforge::AttentionLoweringStrategy::SplitKVDecode &&
            decode_plan.plan.split_kv > 1,
            "split-kv decode strategy");
    const auto decoded = schedforge::execute_decode_attention(
        decode_plan, decode_data.query, cache, 0, 1);
    require(decoded.max_error < 1.0e-4 &&
            decoded.output.size() == static_cast<std::size_t>(
                decode_config.batch * decode_config.query_heads * decode_config.head_dim_value),
            "gqa kv-cache decode execution");
    auto mqa_config = decode_config;
    mqa_config.kv_heads = 1;
    const auto mqa_data = schedforge::make_attention_data(mqa_config, 49);
    const auto mqa_plan = schedforge::AttentionCompiler{}.compile(
        mqa_config, schedforge::select_attention_plan(
            mqa_config, schedforge::TargetInfo::detect(), 4));
    const auto mqa_result = schedforge::execute_attention(mqa_plan, mqa_data, 0, 1);
    require(mqa_result.max_error < 1.0e-4, "mqa attention execution");

    const auto fused_gqa = schedforge::execute_fused_attention_llvm(
        small_io_plan, small_data, 0, 1);
    require(fused_gqa.max_error < 1.0e-3 &&
            fused_gqa.llvm_ir.find("schedforge_fused_attention") != std::string::npos &&
            fused_gqa.llvm_ir.find("llvm.exp") != std::string::npos &&
            fused_gqa.assembly_report.instructions > 0,
            "fused llvm gqa attention execution");
    auto fused_mha_config = small_config;
    fused_mha_config.kv_heads = fused_mha_config.query_heads;
    const auto fused_mha_data = schedforge::make_attention_data(fused_mha_config, 51);
    const auto fused_mha_plan = schedforge::AttentionCompiler{}.compile(
        fused_mha_config, small_io_options);
    require(schedforge::execute_fused_attention_llvm(
                fused_mha_plan, fused_mha_data, 0, 1).max_error < 1.0e-3,
            "fused llvm mha attention execution");
}

void test_decoder_compiler() {
    const schedforge::DecoderConfig dense_config{
        1, 4, 16, 32, 4, 2, 4, 1.0e-5F, 10000.0F, true,
        schedforge::DecoderFFNKind::Dense, 4, 2};
    const auto imported = schedforge::StableHLOImporter{}.importFile(
        std::filesystem::path(SCHEDFORGE_SOURCE_DIR) / "examples/decoder_layer.mlir");
    require(imported.operations().size() >= 20 &&
            imported.dump().find("tensor.rms_norm") != std::string::npos &&
            imported.dump().find("tensor.rope") != std::string::npos,
            "decoder StableHLO import");
    const auto dense_data = schedforge::make_decoder_data(dense_config, 81);
    schedforge::DecoderCompileOptions options;
    options.max_threads = 2;
    const auto dense_plan = schedforge::DecoderCompiler{}.compile(
        imported, dense_config, dense_data, options);
    require(dense_plan.fusion.fused_qkv && dense_plan.fusion.fused_gate_up &&
            dense_plan.fusion.fused_rope && dense_plan.constants.size() == 2 &&
            dense_plan.llvm_ir.size() == 4 &&
            dense_plan.dump().find("tensor.fused_qkv") != std::string::npos &&
            dense_plan.dump().find("tensor.fused_gate_up") != std::string::npos &&
            dense_plan.dump().find("attention.online_softmax") != std::string::npos,
            "dense decoder compilation");
    const auto dense_result = schedforge::execute_decoder_layer(
        dense_plan, dense_data, 0, 1);
    require(dense_result.max_error < 1.0e-3 &&
            dense_result.output.size() == dense_data.input.size(),
            "dense decoder execution");
    const auto quantized_decoder_weights = schedforge::quantize_decoder_weights(
        dense_config, dense_data);
    const auto quantized_decoder_result = schedforge::execute_quantized_decoder_layer(
        dense_config, dense_data, quantized_decoder_weights, 0, 1);
    require(quantized_decoder_result.max_error < 0.15 &&
            quantized_decoder_result.tokens_per_second > 0.0 &&
            quantized_decoder_result.output.size() == dense_data.input.size(),
            "dense decoder INT8 execution");

    auto moe_config = dense_config;
    moe_config.ffn = schedforge::DecoderFFNKind::MoE;
    const auto moe_data = schedforge::make_decoder_data(moe_config, 83);
    const auto moe_plan = schedforge::DecoderCompiler{}.compile(
        imported, moe_config, moe_data, options);
    require(moe_plan.moe.has_value() && moe_plan.constants.size() == 1 &&
            moe_plan.dump().find("moe.histogram") != std::string::npos,
            "moe decoder compilation");
    const auto moe_result = schedforge::execute_decoder_layer(
        moe_plan, moe_data, 0, 1);
    require(moe_result.max_error < 1.0e-3 &&
            moe_result.output.size() == moe_data.input.size(),
            "moe decoder execution");

    const auto imported_moe = schedforge::StableHLOImporter{}.importFile(
        std::filesystem::path(SCHEDFORGE_SOURCE_DIR) / "examples/decoder_layer_moe.mlir");
    require(imported_moe.dump().find("moe.combine") != std::string::npos,
            "decoder MoE StableHLO import");
    const auto explicit_moe_plan = schedforge::DecoderCompiler{}.compile(
        imported_moe, moe_config, moe_data, options);
    require(schedforge::execute_decoder_layer(
                explicit_moe_plan, moe_data, 0, 1).max_error < 1.0e-3,
            "explicit decoder MoE graph execution");

    auto decode_config = dense_config;
    decode_config.sequence = 1;
    decode_config.context_sequence = 16;
    const auto decode_data = schedforge::make_decoder_data(decode_config, 87);
    const auto decode_plan = schedforge::DecoderCompiler{}.compile(
        imported, decode_config, decode_data, options);
    const auto decode_result = schedforge::execute_decoder_layer(
        decode_plan, decode_data, 0, 1);
    require(decode_plan.policy.attention_strategy ==
                schedforge::AttentionLoweringStrategy::SplitKVDecode &&
            decode_result.max_error < 1.0e-3 && decode_result.tokens_per_second > 0.0,
            "decoder KV-cache decode execution");
    const auto quantized_decode_weights = schedforge::quantize_decoder_weights(
        decode_config, decode_data);
    const auto quantized_decode_result = schedforge::execute_quantized_decoder_layer(
        decode_config, decode_data, quantized_decode_weights, 0, 1);
    require(quantized_decode_result.max_error < 0.15 &&
            quantized_decode_result.output.size() == decode_data.input.size(),
            "decoder INT8 KV-cache decode execution");

    const schedforge::MoeConfig quantized_moe_config{8, 16, 24, 4, 2};
    const auto quantized_moe_data = schedforge::make_moe_data(quantized_moe_config, 91);
    const auto quantized_moe_routing = schedforge::route_topk(
        quantized_moe_config, quantized_moe_data);
    const auto quantized_moe_weights = schedforge::quantize_moe_weights(
        quantized_moe_config, quantized_moe_data);
    const auto quantized_moe_result = schedforge::execute_quantized_moe(
        quantized_moe_config, quantized_moe_data, quantized_moe_routing,
        quantized_moe_weights, 0, 1);
    require(quantized_moe_result.max_error < 0.2 &&
            quantized_moe_result.tokens_per_second > 0.0 &&
            quantized_moe_result.output.size() == quantized_moe_data.input.size(),
            "MoE Expert INT8 execution");

    const auto optimized = schedforge::ExecutablePlanOptimizer{}.optimize(
        imported, dense_config, dense_data, 2, 4, 1);
    require(optimized.candidates.size() >= 16 && optimized.hardware_measurements >= 5 &&
            optimized.winner.measured && optimized.speedup > 0.0 &&
            optimized.plan.memory.workspace_bytes > 0,
            "whole-graph decoder plan optimizer");

    const auto profiles = schedforge::realistic_decoder_profiles();
    require(profiles.size() == 24 && profiles.front().config.hidden == 512 &&
            profiles[14].config.hidden == 4096,
            "realistic decoder profile matrix");
    const auto compile_only = schedforge::benchmark_decoder_profile(
        profiles[14], imported, 2, 1, 1, false, 1);
    require(compile_only.evidence == schedforge::DecoderEvidenceKind::CompileOnly &&
            compile_only.measured.milliseconds == 0.0 &&
            compile_only.profile.estimated_weight_bytes > 0,
            "decoder compile-only evidence boundary");

    const auto csv_path = std::filesystem::temp_directory_path() / "schedforge_decoder_test.csv";
    schedforge::write_decoder_benchmark_csv(csv_path, {compile_only});
    std::ifstream csv(csv_path);
    const std::string csv_text((std::istreambuf_iterator<char>(csv)),
                               std::istreambuf_iterator<char>());
    require(csv_text.find("profile,evidence") != std::string::npos &&
            csv_text.find("compile-only") != std::string::npos &&
            csv_text.find("tokens_per_second") != std::string::npos,
            "decoder benchmark CSV schema");
}

}  // namespace

int main() {
    try {
        test_ir_passes();
        test_ssa_ir_and_pass_manager();
        test_correctness_non_multiple();
        test_simulator_and_search();
        test_compiler_architecture();
        test_aot_deployment();
        test_next_milestones();
        test_graph_compiler();
        test_moe_compiler();
        test_attention_compiler();
        test_decoder_compiler();
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
