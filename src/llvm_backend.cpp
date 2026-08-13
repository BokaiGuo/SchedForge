#include "schedforge/compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace schedforge {
namespace {

using MatMulFunction = void (*)(const float*, const float*, const float*, const float*, float*,
                                std::int64_t, std::int64_t, std::int64_t);

struct CachedKernel {
    std::unique_ptr<llvm::orc::LLJIT> jit;
    MatMulFunction function = nullptr;
    std::string llvm_ir;
    std::string assembly;
    AssemblyReport assembly_report;
};

std::mutex kernel_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<CachedKernel>> kernel_cache;

std::string kernel_key(const LoopIR& loop) {
    return "llvm-loopir-v3-" + std::to_string(loop.problem.m) + "x" +
           std::to_string(loop.problem.n) + "x" + std::to_string(loop.problem.k) +
           "-bias" + std::to_string(loop.problem.bias) + "-relu" +
           std::to_string(loop.problem.relu) + "-" +
           std::to_string(std::hash<std::string>{}(loop.dump()));
}

std::string error_string(llvm::Error error) {
    return llvm::toString(std::move(error));
}

llvm::Value* gep2d(llvm::IRBuilder<>& builder, llvm::Type* float_type,
                   llvm::Value* base, llvm::Value* row, llvm::Value* columns,
                   llvm::Value* column) {
    auto* index = builder.CreateAdd(builder.CreateMul(row, columns), column);
    return builder.CreateGEP(float_type, base, index);
}

llvm::Constant* constant_like(llvm::Type* type, double value) {
    if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        return llvm::ConstantVector::getSplat(
            vector_type->getElementCount(),
            llvm::ConstantFP::get(vector_type->getElementType(), value));
    }
    return llvm::ConstantFP::get(type, value);
}

llvm::Value* apply_graph_epilogue(llvm::Module& module, llvm::IRBuilder<>& builder,
                                  llvm::Value* value, llvm::Value* residual,
                                  llvm::Value* index, const LoopExecutionPlan& execution) {
    if (execution.gelu) {
        auto* type = value->getType();
        auto* square = builder.CreateFMul(value, value);
        auto* cube = builder.CreateFMul(square, value);
        auto* inner = builder.CreateFAdd(value,
            builder.CreateFMul(constant_like(type, 0.044715), cube));
        auto* scaled = builder.CreateFMul(constant_like(type, 0.7978845608), inner);
        auto* exp_intrinsic = llvm::Intrinsic::getDeclaration(
            &module, llvm::Intrinsic::exp, {type});
        auto* fabs_intrinsic = llvm::Intrinsic::getDeclaration(
            &module, llvm::Intrinsic::fabs, {type});
        auto* absolute = builder.CreateCall(fabs_intrinsic, {scaled});
        auto* exponential = builder.CreateCall(
            exp_intrinsic, {builder.CreateFMul(constant_like(type, -2.0), absolute)});
        auto* magnitude = builder.CreateFDiv(
            builder.CreateFSub(constant_like(type, 1.0), exponential),
            builder.CreateFAdd(constant_like(type, 1.0), exponential));
        auto* activated = builder.CreateSelect(
            builder.CreateFCmpOLT(scaled, constant_like(type, 0.0)),
            builder.CreateFNeg(magnitude), magnitude);
        value = builder.CreateFMul(
            builder.CreateFMul(constant_like(type, 0.5), value),
            builder.CreateFAdd(constant_like(type, 1.0), activated));
    }
    if (execution.residual) {
        auto* residual_ptr = builder.CreateGEP(builder.getFloatTy(), residual, index);
        auto* residual_value = builder.CreateLoad(value->getType(), residual_ptr);
        residual_value->setAlignment(llvm::Align(1));
        value = builder.CreateFAdd(value, residual_value);
    }
    return value;
}

void build_ijk_function(llvm::Module& module, const Problem& problem,
                        const LoopExecutionPlan& execution) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* void_type = builder.getVoidTy();
    auto* float_type = builder.getFloatTy();
    auto* pointer_type = builder.getPtrTy();
    auto* i64_type = builder.getInt64Ty();
    auto* function_type = llvm::FunctionType::get(
        void_type, {pointer_type, pointer_type, pointer_type, pointer_type, pointer_type,
                    i64_type, i64_type, i64_type}, false);
    auto* function = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                                            "schedforge_matmul", module);
    auto arguments = function->arg_begin();
    llvm::Value* a = arguments++; a->setName("A");
    llvm::Value* b = arguments++; b->setName("B");
    llvm::Value* bias = arguments++; bias->setName("bias");
    llvm::Value* residual = arguments++; residual->setName("residual");
    llvm::Value* c = arguments++; c->setName("C");
    llvm::Value* m = arguments++; m->setName("M");
    llvm::Value* n = arguments++; n->setName("N");
    llvm::Value* k_size = arguments++; k_size->setName("K");

    auto* entry = llvm::BasicBlock::Create(context, "entry", function);
    auto* i_header = llvm::BasicBlock::Create(context, "i.header", function);
    auto* j_header = llvm::BasicBlock::Create(context, "j.header", function);
    auto* k_header = llvm::BasicBlock::Create(context, "k.header", function);
    auto* k_body = llvm::BasicBlock::Create(context, "k.body", function);
    auto* k_exit = llvm::BasicBlock::Create(context, "k.exit", function);
    auto* j_latch = llvm::BasicBlock::Create(context, "j.latch", function);
    auto* i_latch = llvm::BasicBlock::Create(context, "i.latch", function);
    auto* exit = llvm::BasicBlock::Create(context, "exit", function);

    builder.SetInsertPoint(entry);
    builder.CreateBr(i_header);

    builder.SetInsertPoint(i_header);
    auto* i = builder.CreatePHI(i64_type, 2, "i");
    i->addIncoming(builder.getInt64(0), entry);
    builder.CreateBr(j_header);

    builder.SetInsertPoint(j_header);
    auto* j = builder.CreatePHI(i64_type, 2, "j");
    j->addIncoming(builder.getInt64(0), i_header);
    builder.CreateBr(k_header);

    builder.SetInsertPoint(k_header);
    auto* k = builder.CreatePHI(i64_type, 2, "k");
    auto* accumulator = builder.CreatePHI(float_type, 2, "acc");
    k->addIncoming(builder.getInt64(0), j_header);
    accumulator->addIncoming(llvm::ConstantFP::get(float_type, 0.0), j_header);
    auto* k_condition = builder.CreateICmpULT(k, k_size);
    builder.CreateCondBr(k_condition, k_body, k_exit);

    builder.SetInsertPoint(k_body);
    auto* a_ptr = gep2d(builder, float_type, a, i, k_size, k);
    auto* b_ptr = gep2d(builder, float_type, b, k, n, j);
    auto* a_value = builder.CreateLoad(float_type, a_ptr);
    auto* b_value = builder.CreateLoad(float_type, b_ptr);
    auto* product = builder.CreateFMul(a_value, b_value);
    auto* next_accumulator = builder.CreateFAdd(accumulator, product);
    auto* next_k = builder.CreateAdd(k, builder.getInt64(1));
    builder.CreateBr(k_header);
    k->addIncoming(next_k, k_body);
    accumulator->addIncoming(next_accumulator, k_body);

    builder.SetInsertPoint(k_exit);
    llvm::Value* result = accumulator;
    if (problem.bias) {
        auto* bias_ptr = builder.CreateGEP(float_type, bias, j);
        result = builder.CreateFAdd(result, builder.CreateLoad(float_type, bias_ptr));
    }
    if (problem.relu) {
        auto* positive = builder.CreateFCmpOGT(result, llvm::ConstantFP::get(float_type, 0.0));
        result = builder.CreateSelect(positive, result, llvm::ConstantFP::get(float_type, 0.0));
    }
    auto* c_ptr = gep2d(builder, float_type, c, i, n, j);
    auto* output_index = builder.CreateAdd(builder.CreateMul(i, n), j);
    result = apply_graph_epilogue(module, builder, result, residual, output_index, execution);
    builder.CreateStore(result, c_ptr);
    builder.CreateBr(j_latch);

    builder.SetInsertPoint(j_latch);
    auto* next_j = builder.CreateAdd(j, builder.getInt64(1));
    auto* j_done = builder.CreateICmpUGE(next_j, n);
    builder.CreateCondBr(j_done, i_latch, j_header);
    j->addIncoming(next_j, j_latch);

    builder.SetInsertPoint(i_latch);
    auto* next_i = builder.CreateAdd(i, builder.getInt64(1));
    auto* i_done = builder.CreateICmpUGE(next_i, m);
    builder.CreateCondBr(i_done, exit, i_header);
    i->addIncoming(next_i, i_latch);

    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
}

bool can_build_register_kernel(const Problem& problem, const LoopExecutionPlan& execution) {
    const int width = std::max(1, execution.vector_width);
    const int mr = std::max(1, execution.mr);
    const int nr = std::max(width, execution.nr);
    const int vectors = nr / width;
    const std::size_t blocks = static_cast<std::size_t>(problem.m / mr) *
                               static_cast<std::size_t>(problem.n / nr);
    return width > 1 && nr % width == 0 && problem.m % mr == 0 && problem.n % nr == 0 &&
           mr * vectors <= 12 && blocks <= 256;
}

void build_register_tiled_function(llvm::Module& module, const Problem& problem,
                                   const LoopExecutionPlan& execution) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* float_type = builder.getFloatTy();
    auto* i64_type = builder.getInt64Ty();
    auto* pointer_type = builder.getPtrTy();
    auto* function_type = llvm::FunctionType::get(
        builder.getVoidTy(), {pointer_type, pointer_type, pointer_type, pointer_type, pointer_type,
                              i64_type, i64_type, i64_type}, false);
    auto* function = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                                            "schedforge_matmul", module);
    auto arguments = function->arg_begin();
    llvm::Value* a = arguments++; a->setName("A");
    llvm::Value* b = arguments++; b->setName("B");
    llvm::Value* bias = arguments++; bias->setName("bias");
    llvm::Value* residual = arguments++; residual->setName("residual");
    llvm::Value* c = arguments++; c->setName("C");
    llvm::Value* m = arguments++; m->setName("M");
    llvm::Value* n = arguments++; n->setName("N");
    llvm::Value* k_size = arguments++; k_size->setName("K");
    (void)m;

    const int width = execution.vector_width;
    const int mr = execution.mr;
    const int nr = execution.nr;
    const int vector_count = nr / width;
    auto* vector_type = llvm::FixedVectorType::get(float_type, static_cast<unsigned>(width));
    auto* vector_fma = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::fma, {vector_type});
    auto* entry = llvm::BasicBlock::Create(context, "entry", function);
    builder.SetInsertPoint(entry);
    llvm::BasicBlock* predecessor = entry;

    int block_index = 0;
    for (int row = 0; row < problem.m; row += mr) {
        for (int column = 0; column < problem.n; column += nr) {
            auto* k_header = llvm::BasicBlock::Create(context,
                "micro." + std::to_string(block_index) + ".k", function);
            auto* k_body = llvm::BasicBlock::Create(context,
                "micro." + std::to_string(block_index) + ".body", function);
            auto* store = llvm::BasicBlock::Create(context,
                "micro." + std::to_string(block_index) + ".store", function);
            builder.SetInsertPoint(predecessor);
            builder.CreateBr(k_header);

            builder.SetInsertPoint(k_header);
            auto* k = builder.CreatePHI(i64_type, 2, "k");
            k->addIncoming(builder.getInt64(0), predecessor);
            std::vector<llvm::PHINode*> accumulators;
            accumulators.reserve(static_cast<std::size_t>(mr * vector_count));
            for (int index = 0; index < mr * vector_count; ++index) {
                auto* accumulator = builder.CreatePHI(vector_type, 2, "acc");
                accumulator->addIncoming(llvm::Constant::getNullValue(vector_type), predecessor);
                accumulators.push_back(accumulator);
            }
            builder.CreateCondBr(builder.CreateICmpULT(k, k_size), k_body, store);

            builder.SetInsertPoint(k_body);
            std::vector<llvm::Value*> b_vectors;
            for (int vector_index = 0; vector_index < vector_count; ++vector_index) {
                auto* b_ptr = gep2d(builder, float_type, b, k, n,
                    builder.getInt64(column + vector_index * width));
                auto* load = builder.CreateLoad(vector_type, b_ptr);
                load->setAlignment(llvm::Align(1));
                b_vectors.push_back(load);
            }
            std::vector<llvm::Value*> updated;
            updated.reserve(accumulators.size());
            for (int row_index = 0; row_index < mr; ++row_index) {
                auto* a_ptr = gep2d(builder, float_type, a, builder.getInt64(row + row_index),
                                    k_size, k);
                auto* a_value = builder.CreateLoad(float_type, a_ptr);
                auto* inserted = builder.CreateInsertElement(
                    llvm::PoisonValue::get(vector_type), a_value, builder.getInt64(0));
                auto* splat = builder.CreateShuffleVector(inserted,
                    llvm::PoisonValue::get(vector_type),
                    llvm::SmallVector<int, 16>(static_cast<std::size_t>(width), 0));
                for (int vector_index = 0; vector_index < vector_count; ++vector_index) {
                    const int accumulator_index = row_index * vector_count + vector_index;
                    updated.push_back(builder.CreateCall(vector_fma,
                        {splat, b_vectors[static_cast<std::size_t>(vector_index)],
                         accumulators[static_cast<std::size_t>(accumulator_index)]}));
                }
            }
            auto* k_next = builder.CreateAdd(k, builder.getInt64(1));
            builder.CreateBr(k_header);
            k->addIncoming(k_next, k_body);
            for (std::size_t index = 0; index < accumulators.size(); ++index)
                accumulators[index]->addIncoming(updated[index], k_body);

            builder.SetInsertPoint(store);
            for (int row_index = 0; row_index < mr; ++row_index) {
                for (int vector_index = 0; vector_index < vector_count; ++vector_index) {
                    const int accumulator_index = row_index * vector_count + vector_index;
                    llvm::Value* value = accumulators[static_cast<std::size_t>(accumulator_index)];
                    if (problem.bias) {
                        auto* bias_ptr = builder.CreateGEP(float_type, bias,
                            builder.getInt64(column + vector_index * width));
                        auto* bias_vector = builder.CreateLoad(vector_type, bias_ptr);
                        bias_vector->setAlignment(llvm::Align(1));
                        value = builder.CreateFAdd(value, bias_vector);
                    }
                    if (problem.relu) {
                        auto* zero = llvm::Constant::getNullValue(vector_type);
                        value = builder.CreateSelect(builder.CreateFCmpOGT(value, zero), value, zero);
                    }
                    auto* output_index = builder.getInt64(
                        (row + row_index) * problem.n + column + vector_index * width);
                    value = apply_graph_epilogue(
                        module, builder, value, residual, output_index, execution);
                    auto* c_ptr = gep2d(builder, float_type, c,
                        builder.getInt64(row + row_index), n,
                        builder.getInt64(column + vector_index * width));
                    auto* output = builder.CreateStore(value, c_ptr);
                    output->setAlignment(llvm::Align(1));
                }
            }
            predecessor = store;
            ++block_index;
        }
    }
    builder.SetInsertPoint(predecessor);
    builder.CreateRetVoid();
}

void build_ikj_function(llvm::Module& module, const Problem& problem,
                        const LoopExecutionPlan& execution) {
    auto& context = module.getContext();
    llvm::IRBuilder<> builder(context);
    auto* float_type = builder.getFloatTy();
    auto* i64_type = builder.getInt64Ty();
    auto* pointer_type = builder.getPtrTy();
    auto* function_type = llvm::FunctionType::get(
        builder.getVoidTy(), {pointer_type, pointer_type, pointer_type, pointer_type, pointer_type,
                              i64_type, i64_type, i64_type}, false);
    auto* function = llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                                            "schedforge_matmul", module);
    auto arguments = function->arg_begin();
    llvm::Value* a = arguments++; a->setName("A");
    llvm::Value* b = arguments++; b->setName("B");
    llvm::Value* bias = arguments++; bias->setName("bias");
    llvm::Value* residual = arguments++; residual->setName("residual");
    llvm::Value* c = arguments++; c->setName("C");
    llvm::Value* m = arguments++; m->setName("M");
    llvm::Value* n = arguments++; n->setName("N");
    llvm::Value* k_size = arguments++; k_size->setName("K");
    const int width = std::max(1, execution.vector_width);
    auto* vector_type = llvm::FixedVectorType::get(float_type, static_cast<unsigned>(width));

    auto* entry = llvm::BasicBlock::Create(context, "entry", function);
    auto* zero_i_header = llvm::BasicBlock::Create(context, "zero.i", function);
    auto* zero_j_header = llvm::BasicBlock::Create(context, "zero.j", function);
    auto* zero_j_body = llvm::BasicBlock::Create(context, "zero.j.body", function);
    auto* zero_i_latch = llvm::BasicBlock::Create(context, "zero.i.latch", function);
    auto* compute_i_header = llvm::BasicBlock::Create(context, "compute.i", function);
    auto* compute_k_header = llvm::BasicBlock::Create(context, "compute.k", function);
    auto* vector_j_header = llvm::BasicBlock::Create(context, "vector.j", function);
    auto* vector_j_body = llvm::BasicBlock::Create(context, "vector.j.body", function);
    auto* scalar_j_header = llvm::BasicBlock::Create(context, "scalar.j", function);
    auto* scalar_j_body = llvm::BasicBlock::Create(context, "scalar.j.body", function);
    auto* compute_i_latch = llvm::BasicBlock::Create(context, "compute.i.latch", function);
    auto* epilogue_i_header = llvm::BasicBlock::Create(context, "epilogue.i", function);
    auto* epilogue_j_header = llvm::BasicBlock::Create(context, "epilogue.j", function);
    auto* epilogue_j_body = llvm::BasicBlock::Create(context, "epilogue.j.body", function);
    auto* epilogue_i_latch = llvm::BasicBlock::Create(context, "epilogue.i.latch", function);
    auto* exit = llvm::BasicBlock::Create(context, "exit", function);

    builder.SetInsertPoint(entry);
    builder.CreateBr(zero_i_header);

    builder.SetInsertPoint(zero_i_header);
    auto* zero_i = builder.CreatePHI(i64_type, 2, "zero.i.iv");
    zero_i->addIncoming(builder.getInt64(0), entry);
    builder.CreateBr(zero_j_header);

    builder.SetInsertPoint(zero_j_header);
    auto* zero_j = builder.CreatePHI(i64_type, 2, "zero.j.iv");
    zero_j->addIncoming(builder.getInt64(0), zero_i_header);
    auto* zero_j_condition = builder.CreateICmpULT(zero_j, n);
    builder.CreateCondBr(zero_j_condition, zero_j_body, zero_i_latch);

    builder.SetInsertPoint(zero_j_body);
    builder.CreateStore(llvm::ConstantFP::get(float_type, 0.0),
                        gep2d(builder, float_type, c, zero_i, n, zero_j));
    auto* zero_j_next = builder.CreateAdd(zero_j, builder.getInt64(1));
    builder.CreateBr(zero_j_header);
    zero_j->addIncoming(zero_j_next, zero_j_body);

    builder.SetInsertPoint(zero_i_latch);
    auto* zero_i_next = builder.CreateAdd(zero_i, builder.getInt64(1));
    auto* zero_i_done = builder.CreateICmpUGE(zero_i_next, m);
    builder.CreateCondBr(zero_i_done, compute_i_header, zero_i_header);
    zero_i->addIncoming(zero_i_next, zero_i_latch);

    builder.SetInsertPoint(compute_i_header);
    auto* compute_i = builder.CreatePHI(i64_type, 2, "i");
    compute_i->addIncoming(builder.getInt64(0), zero_i_latch);
    builder.CreateBr(compute_k_header);

    builder.SetInsertPoint(compute_k_header);
    auto* compute_k = builder.CreatePHI(i64_type, 2, "k");
    compute_k->addIncoming(builder.getInt64(0), compute_i_header);
    auto* a_ptr = gep2d(builder, float_type, a, compute_i, k_size, compute_k);
    auto* a_value = builder.CreateLoad(float_type, a_ptr);
    builder.CreateBr(vector_j_header);

    builder.SetInsertPoint(vector_j_header);
    auto* vector_j = builder.CreatePHI(i64_type, 2, "j.vec");
    vector_j->addIncoming(builder.getInt64(0), compute_k_header);
    auto* vector_end = builder.CreateAdd(vector_j, builder.getInt64(width));
    auto* can_vectorize = builder.CreateICmpULE(vector_end, n);
    builder.CreateCondBr(can_vectorize, vector_j_body, scalar_j_header);

    builder.SetInsertPoint(vector_j_body);
    auto* b_ptr = gep2d(builder, float_type, b, compute_k, n, vector_j);
    auto* c_ptr = gep2d(builder, float_type, c, compute_i, n, vector_j);
    auto* b_vector_ptr = builder.CreatePointerCast(b_ptr, builder.getPtrTy());
    auto* c_vector_ptr = builder.CreatePointerCast(c_ptr, builder.getPtrTy());
    auto* b_vector_load = builder.CreateLoad(vector_type, b_vector_ptr);
    b_vector_load->setAlignment(llvm::Align(1));
    llvm::Value* b_vector = b_vector_load;
    auto* c_vector_load = builder.CreateLoad(vector_type, c_vector_ptr);
    c_vector_load->setAlignment(llvm::Align(1));
    llvm::Value* c_vector = c_vector_load;
    auto* a_insert = builder.CreateInsertElement(llvm::PoisonValue::get(vector_type), a_value,
                                                  builder.getInt64(0));
    auto* a_vector = builder.CreateShuffleVector(a_insert, llvm::PoisonValue::get(vector_type),
                                                  llvm::SmallVector<int, 16>(static_cast<std::size_t>(width), 0));
    auto* vector_fma = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::fma, {vector_type});
    auto* updated = builder.CreateCall(vector_fma, {a_vector, b_vector, c_vector});
    auto* vector_store = builder.CreateStore(updated, c_vector_ptr);
    vector_store->setAlignment(llvm::Align(1));
    builder.CreateBr(vector_j_header);
    vector_j->addIncoming(vector_end, vector_j_body);

    builder.SetInsertPoint(scalar_j_header);
    auto* scalar_j = builder.CreatePHI(i64_type, 2, "j.scalar");
    scalar_j->addIncoming(vector_j, vector_j_header);
    auto* scalar_condition = builder.CreateICmpULT(scalar_j, n);
    auto* k_latch = llvm::BasicBlock::Create(context, "compute.k.latch", function);
    builder.CreateCondBr(scalar_condition, scalar_j_body, k_latch);

    builder.SetInsertPoint(scalar_j_body);
    auto* scalar_b_ptr = gep2d(builder, float_type, b, compute_k, n, scalar_j);
    auto* scalar_c_ptr = gep2d(builder, float_type, c, compute_i, n, scalar_j);
    auto* scalar_fma = llvm::Intrinsic::getDeclaration(&module, llvm::Intrinsic::fma, {float_type});
    auto* scalar_updated = builder.CreateCall(scalar_fma,
        {a_value, builder.CreateLoad(float_type, scalar_b_ptr), builder.CreateLoad(float_type, scalar_c_ptr)});
    builder.CreateStore(scalar_updated, scalar_c_ptr);
    auto* scalar_j_next = builder.CreateAdd(scalar_j, builder.getInt64(1));
    builder.CreateBr(scalar_j_header);
    scalar_j->addIncoming(scalar_j_next, scalar_j_body);

    builder.SetInsertPoint(k_latch);
    auto* compute_k_next = builder.CreateAdd(compute_k, builder.getInt64(1));
    auto* compute_k_done = builder.CreateICmpUGE(compute_k_next, k_size);
    builder.CreateCondBr(compute_k_done, compute_i_latch, compute_k_header);
    compute_k->addIncoming(compute_k_next, k_latch);

    builder.SetInsertPoint(compute_i_latch);
    auto* compute_i_next = builder.CreateAdd(compute_i, builder.getInt64(1));
    auto* compute_i_done = builder.CreateICmpUGE(compute_i_next, m);
    builder.CreateCondBr(compute_i_done, epilogue_i_header, compute_i_header);
    compute_i->addIncoming(compute_i_next, compute_i_latch);

    builder.SetInsertPoint(epilogue_i_header);
    auto* epilogue_i = builder.CreatePHI(i64_type, 2, "epilogue.i.iv");
    epilogue_i->addIncoming(builder.getInt64(0), compute_i_latch);
    builder.CreateBr(epilogue_j_header);

    builder.SetInsertPoint(epilogue_j_header);
    auto* epilogue_j = builder.CreatePHI(i64_type, 2, "epilogue.j.iv");
    epilogue_j->addIncoming(builder.getInt64(0), epilogue_i_header);
    auto* epilogue_condition = builder.CreateICmpULT(epilogue_j, n);
    builder.CreateCondBr(epilogue_condition, epilogue_j_body, epilogue_i_latch);

    builder.SetInsertPoint(epilogue_j_body);
    auto* result_ptr = gep2d(builder, float_type, c, epilogue_i, n, epilogue_j);
    llvm::Value* result = builder.CreateLoad(float_type, result_ptr);
    if (problem.bias) {
        result = builder.CreateFAdd(result, builder.CreateLoad(float_type,
            builder.CreateGEP(float_type, bias, epilogue_j)));
    }
    if (problem.relu) {
        result = builder.CreateSelect(builder.CreateFCmpOGT(result, llvm::ConstantFP::get(float_type, 0.0)),
                                      result, llvm::ConstantFP::get(float_type, 0.0));
    }
    auto* output_index = builder.CreateAdd(builder.CreateMul(epilogue_i, n), epilogue_j);
    result = apply_graph_epilogue(module, builder, result, residual, output_index, execution);
    builder.CreateStore(result, result_ptr);
    auto* epilogue_j_next = builder.CreateAdd(epilogue_j, builder.getInt64(1));
    builder.CreateBr(epilogue_j_header);
    epilogue_j->addIncoming(epilogue_j_next, epilogue_j_body);

    builder.SetInsertPoint(epilogue_i_latch);
    auto* epilogue_i_next = builder.CreateAdd(epilogue_i, builder.getInt64(1));
    auto* epilogue_i_done = builder.CreateICmpUGE(epilogue_i_next, m);
    builder.CreateCondBr(epilogue_i_done, exit, epilogue_i_header);
    epilogue_i->addIncoming(epilogue_i_next, epilogue_i_latch);

    builder.SetInsertPoint(exit);
    builder.CreateRetVoid();
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

void initialize_llvm() {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

std::string emit_assembly(const llvm::Module& source) {
    std::string lookup_error;
    const std::string triple = llvm::sys::getDefaultTargetTriple();
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, lookup_error);
    if (!target) throw std::runtime_error("LLVM target lookup failed: " + lookup_error);
    llvm::TargetOptions options;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        triple, llvm::sys::getHostCPUName(), "", options, std::nullopt));
    if (!machine) throw std::runtime_error("LLVM target machine creation failed");
    auto module = llvm::CloneModule(source);
    module->setTargetTriple(triple);
    module->setDataLayout(machine->createDataLayout());
    llvm::SmallVector<char, 0> storage;
    llvm::raw_svector_ostream output(storage);
    llvm::legacy::PassManager passes;
    if (machine->addPassesToEmitFile(passes, output, nullptr, llvm::CodeGenFileType::AssemblyFile))
        throw std::runtime_error("LLVM assembly emission is unsupported");
    passes.run(*module);
    return std::string(storage.begin(), storage.end());
}

}  // namespace

bool LLVMJITBackend::available() const { return true; }

LLVMJITResult LLVMJITBackend::benchmark(const LoopIR& loop, const TensorData& data,
                                        int warmup,
                                        int repetitions) const {
    initialize_llvm();
    verify_loop_ir(loop);
    const auto execution = analyze_loop_ir(loop);
    const std::string cache_key = kernel_key(loop);
    std::shared_ptr<CachedKernel> cached;
    bool cache_hit = false;
    {
        std::lock_guard lock(kernel_cache_mutex);
        const auto found = kernel_cache.find(cache_key);
        if (found != kernel_cache.end()) {
            cached = found->second;
            cache_hit = true;
        }
    }
    const auto compile_start = std::chrono::steady_clock::now();
    if (!cached) {
        auto context = std::make_unique<llvm::LLVMContext>();
        auto module = std::make_unique<llvm::Module>("schedforge_jit", *context);
        if (can_build_register_kernel(loop.problem, execution))
            build_register_tiled_function(*module, loop.problem, execution);
        else if (execution.vector_width > 1 || execution.order == LoopOrder::IKJ)
            build_ikj_function(*module, loop.problem, execution);
        else build_ijk_function(*module, loop.problem, execution);
        if (llvm::verifyModule(*module, &llvm::errs())) throw std::runtime_error("LLVM module verification failed");
        optimize_module(*module);
        auto kernel = std::make_shared<CachedKernel>();
        llvm::raw_string_ostream stream(kernel->llvm_ir);
        module->print(stream, nullptr);
        stream.flush();
        kernel->assembly = emit_assembly(*module);
        kernel->assembly_report = AssemblyAnalyzer{}.analyze(kernel->assembly);
        if (const char* dump_path = std::getenv("SCHEDFORGE_JIT_ASM")) {
            std::ofstream(dump_path) << kernel->assembly;
        }
        auto jit_expected = llvm::orc::LLJITBuilder().create();
        if (!jit_expected) throw std::runtime_error(error_string(jit_expected.takeError()));
        kernel->jit = std::move(*jit_expected);
        if (auto error = kernel->jit->addIRModule(
                llvm::orc::ThreadSafeModule(std::move(module), std::move(context))))
            throw std::runtime_error(error_string(std::move(error)));
        auto symbol = kernel->jit->lookup("schedforge_matmul");
        if (!symbol) throw std::runtime_error(error_string(symbol.takeError()));
        kernel->function = symbol->toPtr<MatMulFunction>();
        {
            std::lock_guard lock(kernel_cache_mutex);
            const auto [iterator, inserted] = kernel_cache.emplace(cache_key, kernel);
            cached = inserted ? kernel : iterator->second;
        }
    }
    const auto compile_end = std::chrono::steady_clock::now();

    std::vector<float> output(static_cast<std::size_t>(loop.problem.m) * loop.problem.n);
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) {
        cached->function(data.a.data(), data.b.data(), data.bias.data(), data.residual.data(),
                         output.data(), loop.problem.m, loop.problem.n, loop.problem.k);
    }
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        cached->function(data.a.data(), data.b.data(), data.bias.data(), data.residual.data(),
                         output.data(), loop.problem.m, loop.problem.n, loop.problem.k);
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double execution_ms = timings[timings.size() / 2];
    std::vector<float> expected;
    execute(loop, data, expected);
    const double operations = 2.0 * loop.problem.m * loop.problem.n * loop.problem.k;
    const double compile_milliseconds = cache_hit ? 0.0
        : std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
    return {compile_milliseconds,
            execution_ms, operations / (execution_ms * 1.0e6),
            max_abs_error(expected, output), cached->llvm_ir, cached->assembly,
            cached->assembly_report};
}

}  // namespace schedforge
