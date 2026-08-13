#include "schedforge/attention_compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace schedforge {
namespace {

using AttentionFunction = void (*)(const float*, const float*, const float*, float*,
                                   std::int64_t, std::int64_t);

std::string error_string(llvm::Error error) {
    return llvm::toString(std::move(error));
}

void initialize_llvm() {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

void optimize_module(llvm::Module& module) {
    llvm::LoopAnalysisManager loop_analyses;
    llvm::FunctionAnalysisManager function_analyses;
    llvm::CGSCCAnalysisManager cgscc_analyses;
    llvm::ModuleAnalysisManager module_analyses;
    llvm::PassBuilder pass_builder;
    pass_builder.registerModuleAnalyses(module_analyses);
    pass_builder.registerCGSCCAnalyses(cgscc_analyses);
    pass_builder.registerFunctionAnalyses(function_analyses);
    pass_builder.registerLoopAnalyses(loop_analyses);
    pass_builder.crossRegisterProxies(loop_analyses, function_analyses,
                                      cgscc_analyses, module_analyses);
    auto pipeline = pass_builder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    pipeline.run(module, module_analyses);
}

std::unique_ptr<llvm::TargetMachine> target_machine() {
    std::string error;
    const std::string triple = llvm::sys::getDefaultTargetTriple();
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) throw std::runtime_error("LLVM target lookup failed: " + error);
    llvm::TargetOptions options;
    auto machine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
        triple, llvm::sys::getHostCPUName(), "", options, std::nullopt));
    if (!machine) throw std::runtime_error("LLVM target machine creation failed");
    return machine;
}

std::string emit_assembly(const llvm::Module& source) {
    auto machine = target_machine();
    auto module = llvm::CloneModule(source);
    module->setTargetTriple(machine->getTargetTriple().str());
    module->setDataLayout(machine->createDataLayout());
    llvm::SmallVector<char, 0> storage;
    llvm::raw_svector_ostream output(storage);
    llvm::legacy::PassManager passes;
    if (machine->addPassesToEmitFile(
            passes, output, nullptr, llvm::CodeGenFileType::AssemblyFile))
        throw std::runtime_error("LLVM assembly emission is unsupported");
    passes.run(*module);
    return std::string(storage.begin(), storage.end());
}

llvm::Value* pointer_at(llvm::IRBuilder<>& builder, llvm::Value* base,
                        llvm::Value* index) {
    return builder.CreateGEP(builder.getFloatTy(), base, index);
}

void build_fused_attention(llvm::Module& module, const AttentionConfig& config) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* pointer = builder.getPtrTy();
    auto* function_type = llvm::FunctionType::get(
        builder.getVoidTy(), {pointer, pointer, pointer, pointer,
                              builder.getInt64Ty(), builder.getInt64Ty()}, false);
    auto* function = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                                            "schedforge_fused_attention", module);
    function->addFnAttr(llvm::Attribute::NoUnwind);
    auto argument = function->arg_begin();
    llvm::Value* query = argument++; query->setName("query");
    llvm::Value* key = argument++; key->setName("key");
    llvm::Value* value = argument++; value->setName("value");
    llvm::Value* output = argument++; output->setName("output");
    llvm::Value* row_begin = argument++; row_begin->setName("row_begin");
    llvm::Value* row_end = argument++; row_end->setName("row_end");
    for (auto* input : {query, key, value}) {
        if (auto* arg = llvm::dyn_cast<llvm::Argument>(input)) {
            arg->addAttr(llvm::Attribute::NoAlias);
            arg->addAttr(llvm::Attribute::ReadOnly);
        }
    }
    llvm::cast<llvm::Argument>(output)->addAttr(llvm::Attribute::NoAlias);

    const int heads_per_kv = config.query_heads / config.kv_heads;
    const float scale = config.scale > 0.0F ? config.scale :
        1.0F / std::sqrt(static_cast<float>(config.head_dim));
    auto* entry = llvm::BasicBlock::Create(context, "entry", function);
    auto* row_header = llvm::BasicBlock::Create(context, "row.header", function);
    auto* row_body = llvm::BasicBlock::Create(context, "row.body", function);
    auto* init_header = llvm::BasicBlock::Create(context, "init.header", function);
    auto* init_body = llvm::BasicBlock::Create(context, "init.body", function);
    auto* key_header = llvm::BasicBlock::Create(context, "key.header", function);
    auto* dot_header = llvm::BasicBlock::Create(context, "dot.header", function);
    auto* dot_body = llvm::BasicBlock::Create(context, "dot.body", function);
    auto* online = llvm::BasicBlock::Create(context, "online", function);
    auto* value_header = llvm::BasicBlock::Create(context, "value.header", function);
    auto* value_body = llvm::BasicBlock::Create(context, "value.body", function);
    auto* key_latch = llvm::BasicBlock::Create(context, "key.latch", function);
    auto* normalize_header = llvm::BasicBlock::Create(context, "normalize.header", function);
    auto* normalize_body = llvm::BasicBlock::Create(context, "normalize.body", function);
    auto* row_latch = llvm::BasicBlock::Create(context, "row.latch", function);
    auto* exit = llvm::BasicBlock::Create(context, "exit", function);

    builder.SetInsertPoint(entry);
    builder.CreateBr(row_header);
    builder.SetInsertPoint(row_header);
    auto* row = builder.CreatePHI(builder.getInt64Ty(), 2, "row");
    row->addIncoming(row_begin, entry);
    builder.CreateCondBr(builder.CreateICmpULT(row, row_end), row_body, exit);

    builder.SetInsertPoint(row_body);
    auto* query_token = builder.CreateURem(row, builder.getInt64(config.sequence_query));
    auto* head_batch = builder.CreateUDiv(row, builder.getInt64(config.sequence_query));
    auto* query_head = builder.CreateURem(head_batch, builder.getInt64(config.query_heads));
    auto* batch = builder.CreateUDiv(head_batch, builder.getInt64(config.query_heads));
    auto* kv_head = builder.CreateUDiv(query_head, builder.getInt64(heads_per_kv));
    auto* query_base = builder.CreateMul(row, builder.getInt64(config.head_dim));
    auto* kv_head_batch = builder.CreateAdd(
        builder.CreateMul(batch, builder.getInt64(config.kv_heads)), kv_head);
    auto* key_base = builder.CreateMul(
        builder.CreateMul(kv_head_batch, builder.getInt64(config.sequence_kv)),
        builder.getInt64(config.head_dim));
    auto* value_base = builder.CreateMul(
        builder.CreateMul(kv_head_batch, builder.getInt64(config.sequence_kv)),
        builder.getInt64(config.head_dim_value));
    auto* output_base = builder.CreateMul(row, builder.getInt64(config.head_dim_value));
    llvm::Value* key_limit = builder.getInt64(config.sequence_kv);
    if (config.causal)
        key_limit = builder.CreateAdd(query_token,
            builder.getInt64(config.sequence_kv - config.sequence_query + 1));
    builder.CreateBr(init_header);

    builder.SetInsertPoint(init_header);
    auto* init_dim = builder.CreatePHI(builder.getInt64Ty(), 2, "init.dim");
    init_dim->addIncoming(builder.getInt64(0), row_body);
    builder.CreateCondBr(builder.CreateICmpULT(
        init_dim, builder.getInt64(config.head_dim_value)), init_body, key_header);
    builder.SetInsertPoint(init_body);
    builder.CreateStore(llvm::ConstantFP::get(builder.getFloatTy(), 0.0), pointer_at(
        builder, output, builder.CreateAdd(output_base, init_dim)));
    auto* init_next = builder.CreateAdd(init_dim, builder.getInt64(1));
    builder.CreateBr(init_header);
    init_dim->addIncoming(init_next, init_body);

    builder.SetInsertPoint(key_header);
    auto* key_token = builder.CreatePHI(builder.getInt64Ty(), 2, "key.token");
    auto* maximum = builder.CreatePHI(builder.getFloatTy(), 2, "maximum");
    auto* denominator = builder.CreatePHI(builder.getFloatTy(), 2, "denominator");
    key_token->addIncoming(builder.getInt64(0), init_header);
    maximum->addIncoming(llvm::ConstantFP::getInfinity(builder.getFloatTy(), true), init_header);
    denominator->addIncoming(llvm::ConstantFP::get(builder.getFloatTy(), 0.0), init_header);
    builder.CreateCondBr(builder.CreateICmpULT(key_token, key_limit), dot_header,
                         normalize_header);

    builder.SetInsertPoint(dot_header);
    auto* dot_dim = builder.CreatePHI(builder.getInt64Ty(), 2, "dot.dim");
    auto* dot = builder.CreatePHI(builder.getFloatTy(), 2, "dot");
    dot_dim->addIncoming(builder.getInt64(0), key_header);
    dot->addIncoming(llvm::ConstantFP::get(builder.getFloatTy(), 0.0), key_header);
    builder.CreateCondBr(builder.CreateICmpULT(dot_dim, builder.getInt64(config.head_dim)),
                         dot_body, online);
    builder.SetInsertPoint(dot_body);
    auto* q = builder.CreateLoad(builder.getFloatTy(), pointer_at(
        builder, query, builder.CreateAdd(query_base, dot_dim)));
    auto* key_offset = builder.CreateAdd(
        key_base, builder.CreateAdd(builder.CreateMul(key_token,
            builder.getInt64(config.head_dim)), dot_dim));
    auto* k = builder.CreateLoad(builder.getFloatTy(), pointer_at(builder, key, key_offset));
    auto* dot_next = builder.CreateFAdd(dot, builder.CreateFMul(q, k));
    auto* dot_dim_next = builder.CreateAdd(dot_dim, builder.getInt64(1));
    builder.CreateBr(dot_header);
    dot_dim->addIncoming(dot_dim_next, dot_body);
    dot->addIncoming(dot_next, dot_body);

    builder.SetInsertPoint(online);
    auto* score = builder.CreateFMul(dot, llvm::ConstantFP::get(builder.getFloatTy(), scale));
    auto* new_maximum = builder.CreateSelect(builder.CreateFCmpOGT(score, maximum),
                                              score, maximum);
    auto* exp = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::exp,
                                                 {builder.getFloatTy()});
    llvm::Value* old_scale_argument = builder.CreateFSub(maximum, new_maximum);
    llvm::Value* probability_argument = builder.CreateFSub(score, new_maximum);
    auto* old_scale = builder.CreateCall(exp, llvm::ArrayRef<llvm::Value*>{old_scale_argument});
    auto* probability = builder.CreateCall(exp, llvm::ArrayRef<llvm::Value*>{probability_argument});
    auto* new_denominator = builder.CreateFAdd(
        builder.CreateFMul(denominator, old_scale), probability);
    builder.CreateBr(value_header);

    builder.SetInsertPoint(value_header);
    auto* value_dim = builder.CreatePHI(builder.getInt64Ty(), 2, "value.dim");
    value_dim->addIncoming(builder.getInt64(0), online);
    builder.CreateCondBr(builder.CreateICmpULT(
        value_dim, builder.getInt64(config.head_dim_value)), value_body, key_latch);
    builder.SetInsertPoint(value_body);
    auto* output_offset = builder.CreateAdd(output_base, value_dim);
    auto* current = builder.CreateLoad(builder.getFloatTy(),
                                        pointer_at(builder, output, output_offset));
    auto* value_offset = builder.CreateAdd(
        value_base, builder.CreateAdd(builder.CreateMul(key_token,
            builder.getInt64(config.head_dim_value)), value_dim));
    auto* v = builder.CreateLoad(builder.getFloatTy(), pointer_at(builder, value, value_offset));
    auto* updated = builder.CreateFAdd(builder.CreateFMul(current, old_scale),
                                       builder.CreateFMul(probability, v));
    builder.CreateStore(updated, pointer_at(builder, output, output_offset));
    auto* value_dim_next = builder.CreateAdd(value_dim, builder.getInt64(1));
    builder.CreateBr(value_header);
    value_dim->addIncoming(value_dim_next, value_body);

    builder.SetInsertPoint(key_latch);
    auto* key_next = builder.CreateAdd(key_token, builder.getInt64(1));
    builder.CreateBr(key_header);
    key_token->addIncoming(key_next, key_latch);
    maximum->addIncoming(new_maximum, key_latch);
    denominator->addIncoming(new_denominator, key_latch);

    builder.SetInsertPoint(normalize_header);
    auto* normalize_dim = builder.CreatePHI(builder.getInt64Ty(), 2, "normalize.dim");
    normalize_dim->addIncoming(builder.getInt64(0), key_header);
    builder.CreateCondBr(builder.CreateICmpULT(
        normalize_dim, builder.getInt64(config.head_dim_value)), normalize_body, row_latch);
    builder.SetInsertPoint(normalize_body);
    auto* normalize_offset = builder.CreateAdd(output_base, normalize_dim);
    auto* numerator = builder.CreateLoad(builder.getFloatTy(),
                                          pointer_at(builder, output, normalize_offset));
    builder.CreateStore(builder.CreateFDiv(numerator, denominator),
                        pointer_at(builder, output, normalize_offset));
    auto* normalize_next = builder.CreateAdd(normalize_dim, builder.getInt64(1));
    builder.CreateBr(normalize_header);
    normalize_dim->addIncoming(normalize_next, normalize_body);

    builder.SetInsertPoint(row_latch);
    auto* row_next = builder.CreateAdd(row, builder.getInt64(1));
    builder.CreateBr(row_header);
    row->addIncoming(row_next, row_latch);
    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
}

}  // namespace

FusedAttentionLLVMResult execute_fused_attention_llvm(
    const AttentionExecutablePlan& plan, const AttentionData& data,
    int warmup, int repetitions) {
    initialize_llvm();
    const std::string pipeline = plan.pipeline.dump();
    for (const std::string_view operation : {
             "attention.qk", "attention.online_softmax", "attention.pv", "vector.div"})
        if (pipeline.find(operation) == std::string::npos)
            throw std::invalid_argument(
                "fused Attention LLVM requires a complete online-softmax TilePipelineIR");
    const std::size_t query_values = static_cast<std::size_t>(plan.config.batch) *
        plan.config.query_heads * plan.config.sequence_query * plan.config.head_dim;
    const std::size_t key_values = static_cast<std::size_t>(plan.config.batch) *
        plan.config.kv_heads * plan.config.sequence_kv * plan.config.head_dim;
    const std::size_t value_values = static_cast<std::size_t>(plan.config.batch) *
        plan.config.kv_heads * plan.config.sequence_kv * plan.config.head_dim_value;
    if (data.query.size() != query_values || data.key.size() != key_values ||
        data.value.size() != value_values)
        throw std::invalid_argument("fused Attention LLVM data shape mismatch");
    const auto compile_start = std::chrono::steady_clock::now();
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("schedforge_fused_attention", *context);
    build_fused_attention(*module, plan.config);
    if (llvm::verifyModule(*module, &llvm::errs()))
        throw std::runtime_error("fused Attention LLVM verification failed");
    optimize_module(*module);
    FusedAttentionLLVMResult result;
    llvm::raw_string_ostream stream(result.llvm_ir);
    module->print(stream, nullptr);
    stream.flush();
    result.assembly = emit_assembly(*module);
    result.assembly_report = AssemblyAnalyzer{}.analyze(result.assembly);
    auto jit_expected = llvm::orc::LLJITBuilder().create();
    if (!jit_expected) throw std::runtime_error(error_string(jit_expected.takeError()));
    auto jit = std::move(*jit_expected);
    if (auto error = jit->addIRModule(
            llvm::orc::ThreadSafeModule(std::move(module), std::move(context))))
        throw std::runtime_error(error_string(std::move(error)));
    auto symbol = jit->lookup("schedforge_fused_attention");
    if (!symbol) throw std::runtime_error(error_string(symbol.takeError()));
    auto function = symbol->toPtr<AttentionFunction>();
    result.compile_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - compile_start).count();
    result.output.resize(static_cast<std::size_t>(plan.config.batch) *
        plan.config.query_heads * plan.config.sequence_query * plan.config.head_dim_value);
    const int rows = plan.config.batch * plan.config.query_heads * plan.config.sequence_query;
    result.threads = std::clamp(plan.plan.schedule.threads, 1, rows);
    const int rows_per_thread = (rows + result.threads - 1) / result.threads;
    const auto invoke = [&] {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(result.threads));
        for (int thread = 0; thread < result.threads; ++thread) {
            const int begin = thread * rows_per_thread;
            const int end = std::min(begin + rows_per_thread, rows);
            if (begin >= end) continue;
            workers.emplace_back([&, begin, end] {
                function(data.query.data(), data.key.data(), data.value.data(),
                         result.output.data(), begin, end);
            });
        }
        for (auto& worker : workers) worker.join();
    };
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration)
        invoke();
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(std::max(1, repetitions)));
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        invoke();
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    result.execution_milliseconds = timings[timings.size() / 2];
    result.p95_milliseconds = timings[std::min(timings.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(timings.size()))) - 1)];
    result.max_error = max_abs_error(reference_attention(plan.config, data), result.output);
    return result;
}

}  // namespace schedforge
