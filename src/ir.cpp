#include "schedforge/schedforge.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace schedforge {
namespace {

std::string op_name(LoopOpKind kind) {
    switch (kind) {
        case LoopOpKind::For: return "scf.for";
        case LoopOpKind::ParallelFor: return "scf.parallel";
        case LoopOpKind::BufferAlloc: return "buffer.alloc";
        case LoopOpKind::Pack: return "buffer.pack";
        case LoopOpKind::Prefetch: return "memref.prefetch";
        case LoopOpKind::ScalarLoad: return "memref.load";
        case LoopOpKind::VectorLoad: return "vector.load";
        case LoopOpKind::Broadcast: return "vector.broadcast";
        case LoopOpKind::AccumulatorInit: return "vector.accumulator.init";
        case LoopOpKind::Fma: return "vector.fma";
        case LoopOpKind::AddBias: return "epilogue.add_bias";
        case LoopOpKind::Relu: return "epilogue.relu";
        case LoopOpKind::Gelu: return "epilogue.gelu";
        case LoopOpKind::AddResidual: return "epilogue.add_residual";
        case LoopOpKind::ScalarStore: return "memref.store";
        case LoopOpKind::VectorStore: return "vector.store";
    }
    return "unknown";
}

LoopOperation make_loop(LoopOpKind kind, std::string induction, std::string upper,
                        int step, std::vector<LoopOperation> body = {}) {
    LoopOperation operation;
    operation.kind = kind;
    operation.induction = std::move(induction);
    operation.lower = "0";
    operation.upper = std::move(upper);
    operation.step = std::max(1, step);
    operation.body = std::move(body);
    return operation;
}

void visit(const std::vector<LoopOperation>& operations,
           const std::function<void(const LoopOperation&)>& callback) {
    for (const auto& operation : operations) {
        callback(operation);
        visit(operation.body, callback);
    }
}

const LoopOperation* find_loop(const LoopIR& loop, const std::string& induction) {
    const LoopOperation* result = nullptr;
    visit(loop.operations(), [&](const LoopOperation& operation) {
        if (!result && (operation.kind == LoopOpKind::For || operation.kind == LoopOpKind::ParallelFor) &&
            operation.induction == induction) result = &operation;
    });
    return result;
}

}  // namespace

std::string GraphIR::dump() const {
    std::ostringstream out;
    out << "module {\n"
        << "  %A = input tensor<" << problem.m << "x" << problem.k << "xf32>\n"
        << "  %B = input tensor<" << problem.k << "x" << problem.n << "xf32>\n";
    if (problem.bias) out << "  %bias = input tensor<" << problem.n << "xf32>\n";
    out << "  %0 = matmul %A, %B\n";
    if (problem.bias) out << "  %1 = add %0, %bias\n";
    if (problem.relu) out << "  %2 = relu %" << (problem.bias ? "1" : "0") << "\n";
    out << "  return %" << (problem.relu ? "2" : (problem.bias ? "1" : "0")) << "\n}\n";
    return out.str();
}

std::string LoopOperation::dump(int indent) const {
    const std::string padding(static_cast<std::size_t>(indent), ' ');
    std::ostringstream out;
    if (!result.empty()) out << padding << '%' << result << " = ";
    else out << padding;
    out << op_name(kind);
    if (kind == LoopOpKind::For || kind == LoopOpKind::ParallelFor) {
        out << " %" << induction << " = " << lower << " to " << upper << " step " << step;
        if (kind == LoopOpKind::ParallelFor) out << " threads=" << threads << " pinned=" << pinned;
        out << " {\n";
        for (const auto& child : body) out << child.dump(indent + 2) << '\n';
        out << padding << '}';
        return out.str();
    }
    if (!source.empty()) out << ' ' << source;
    if (!destination.empty()) out << " -> " << destination;
    if (!indices.empty()) {
        out << '[';
        for (std::size_t index = 0; index < indices.size(); ++index) {
            if (index) out << ',';
            out << '%' << indices[index];
        }
        out << ']';
    }
    if (width > 1) out << " : vector<" << width << "xf32>";
    if (distance > 0) out << " distance=" << distance;
    return out.str();
}

std::string LoopIR::dump() const {
    std::ostringstream out;
    out << "func @matmul(%A, %B, %bias, %C) {\n";
    for (const auto& operation : operations_) out << operation.dump(2) << '\n';
    out << "}\n";
    return out.str();
}

const std::vector<LoopOperation>& LoopIR::operations() const { return operations_; }

void LoopIR::appendOperation(LoopOperation operation) {
    operations_.push_back(std::move(operation));
    execution_plan_.reset();
}

void LoopIR::insertEpilogueBeforeStores(LoopOperation operation) {
    std::size_t inserted = 0;
    std::function<void(std::vector<LoopOperation>&)> rewrite =
        [&](std::vector<LoopOperation>& operations) {
            for (std::size_t index = 0; index < operations.size(); ++index) {
                rewrite(operations[index].body);
                if (operations[index].kind != LoopOpKind::ScalarStore &&
                    operations[index].kind != LoopOpKind::VectorStore) continue;
                operations.insert(operations.begin() + static_cast<std::ptrdiff_t>(index), operation);
                ++inserted;
                ++index;
            }
        };
    rewrite(operations_);
    if (inserted == 0) throw std::invalid_argument("LoopIR epilogue requires a store");
    execution_plan_.reset();
}

LoopIR::LoopIR(Problem input_problem, const Schedule& schedule) {
    *this = apply_schedule(input_problem, schedule);
}

LoopIR apply_schedule(const Problem& problem, const Schedule& schedule) {
    LoopIR loop;
    loop.problem = problem;
    auto append_buffer_ops = [&](bool enabled, const std::string& source,
                                 const std::string& destination, const std::string& result) {
        if (!enabled) return;
        LoopOperation allocation;
        allocation.kind = LoopOpKind::BufferAlloc;
        allocation.result = result;
        allocation.destination = destination;
        loop.operations_.push_back(std::move(allocation));
        LoopOperation pack;
        pack.kind = LoopOpKind::Pack;
        pack.source = source;
        pack.destination = destination;
        loop.operations_.push_back(std::move(pack));
    };
    append_buffer_ops(schedule.pack_a, "%A", "%packedA", "packed_a");
    append_buffer_ops(schedule.pack_b, "%B", "%packedB", "packed_b");

    const int bm = schedule.tiled ? std::max(1, schedule.bm) : problem.m;
    const int bn = schedule.tiled ? std::max(1, schedule.bn) : problem.n;
    const int bk = schedule.tiled ? std::max(1, schedule.bk) : problem.k;
    const int width = std::max(1, schedule.vector_width);

    auto make_reduction_body = [&] {
        std::vector<LoopOperation> body;
        if (schedule.prefetch_distance > 0) {
            LoopOperation prefetch;
            prefetch.kind = LoopOpKind::Prefetch;
            prefetch.source = schedule.pack_b ? "%packedB" : "%B";
            prefetch.indices = {"k", "j"};
            prefetch.distance = schedule.prefetch_distance;
            body.push_back(std::move(prefetch));
        }
        LoopOperation load_a;
        load_a.kind = LoopOpKind::ScalarLoad;
        load_a.result = "a";
        load_a.source = schedule.pack_a ? "%packedA" : "%A";
        load_a.indices = {"i", "k"};
        body.push_back(std::move(load_a));
        LoopOperation load_b;
        load_b.kind = width > 1 ? LoopOpKind::VectorLoad : LoopOpKind::ScalarLoad;
        load_b.result = "b";
        load_b.source = schedule.pack_b ? "%packedB" : "%B";
        load_b.indices = {"k", "j"};
        load_b.width = width;
        body.push_back(std::move(load_b));
        if (width > 1) {
            LoopOperation broadcast;
            broadcast.kind = LoopOpKind::Broadcast;
            broadcast.result = "a_vec";
            broadcast.source = "%a";
            broadcast.width = width;
            body.push_back(std::move(broadcast));
        }
        LoopOperation fma;
        fma.kind = LoopOpKind::Fma;
        fma.result = "acc";
        fma.source = width > 1 ? "%a_vec,%b,%acc" : "%a,%b,%acc";
        fma.width = width;
        body.push_back(std::move(fma));
        return body;
    };

    auto append_epilogue = [&](std::vector<LoopOperation>& body) {
        if (problem.bias) {
            LoopOperation bias;
            bias.kind = LoopOpKind::AddBias;
            bias.result = "acc";
            bias.source = "%bias";
            bias.indices = {"j"};
            bias.width = width;
            body.push_back(std::move(bias));
        }
        if (problem.relu) {
            LoopOperation relu;
            relu.kind = LoopOpKind::Relu;
            relu.result = "acc";
            relu.source = "%acc";
            relu.width = width;
            body.push_back(std::move(relu));
        }
    };

    if (schedule.order == LoopOrder::IKJ && width == 1 && !schedule.pack_a && !schedule.pack_b) {
        std::vector<LoopOperation> update_body = make_reduction_body();
        LoopOperation store;
        store.kind = LoopOpKind::ScalarStore;
        store.source = "%acc";
        store.destination = "%C";
        store.indices = {"i", "j"};
        update_body.push_back(std::move(store));
        auto j = make_loop(LoopOpKind::For, "j", "min(jj+" + std::to_string(bn) + ",N)", 1,
                           std::move(update_body));
        auto k = make_loop(LoopOpKind::For, "k", "min(kk+" + std::to_string(bk) + ",K)",
                           std::max(1, schedule.unroll_k), {std::move(j)});
        auto i = make_loop(LoopOpKind::For, "i", "min(ii+" + std::to_string(bm) + ",M)", 1,
                           {std::move(k)});
        auto jj = make_loop(LoopOpKind::For, "jj", "N", bn, {std::move(i)});
        auto kk = make_loop(LoopOpKind::For, "kk", "K", bk, {std::move(jj)});
        auto ii = make_loop(schedule.threads > 1 ? LoopOpKind::ParallelFor : LoopOpKind::For,
                            "ii", "M", bm, {std::move(kk)});
        ii.threads = std::max(1, schedule.threads);
        ii.pinned = schedule.pin_threads;
        loop.operations_.push_back(std::move(ii));
    } else {
        std::vector<LoopOperation> element_body;
        LoopOperation init;
        init.kind = LoopOpKind::AccumulatorInit;
        init.result = "acc";
        init.width = width;
        element_body.push_back(std::move(init));
        auto k = make_loop(LoopOpKind::For, "k", "min(kk+" + std::to_string(bk) + ",K)",
                           std::max(1, schedule.unroll_k), make_reduction_body());
        element_body.push_back(make_loop(LoopOpKind::For, "kk", "K", bk, {std::move(k)}));
        if (schedule.fused) append_epilogue(element_body);
        LoopOperation store;
        store.kind = width > 1 ? LoopOpKind::VectorStore : LoopOpKind::ScalarStore;
        store.source = "%acc";
        store.destination = "%C";
        store.indices = {"i", "j"};
        store.width = width;
        element_body.push_back(std::move(store));
        auto j = make_loop(LoopOpKind::For, "j", "min(jj+" + std::to_string(bn) + ",N)",
                           width > 1 ? std::max(width, schedule.nr) : 1, std::move(element_body));
        auto i = make_loop(LoopOpKind::For, "i", "min(ii+" + std::to_string(bm) + ",M)",
                           width > 1 ? std::max(1, schedule.mr) : 1, {std::move(j)});
        auto jj = make_loop(LoopOpKind::For, "jj", "N", bn, {std::move(i)});
        auto ii = make_loop(schedule.threads > 1 ? LoopOpKind::ParallelFor : LoopOpKind::For,
                            "ii", "M", bm, {std::move(jj)});
        ii.threads = std::max(1, schedule.threads);
        ii.pinned = schedule.pin_threads;
        loop.operations_.push_back(std::move(ii));
    }

    if (!schedule.fused || (schedule.order == LoopOrder::IKJ && width == 1)) {
        if (problem.bias) {
            LoopOperation bias;
            bias.kind = LoopOpKind::AddBias;
            bias.source = "%bias";
            bias.destination = "%C";
            loop.operations_.push_back(std::move(bias));
        }
        if (problem.relu) {
            LoopOperation relu;
            relu.kind = LoopOpKind::Relu;
            relu.source = "%C";
            relu.destination = "%C";
            loop.operations_.push_back(std::move(relu));
        }
    }
    verify_loop_ir(loop);
    (void)analyze_loop_ir(loop);
    return loop;
}

LoopExecutionPlan analyze_loop_ir(const LoopIR& loop) {
    if (loop.execution_plan_) return *loop.execution_plan_;
    LoopExecutionPlan plan;
    plan.bm = loop.problem.m;
    plan.bn = loop.problem.n;
    plan.bk = loop.problem.k;
    plan.mc = plan.bm;
    plan.nc = plan.bn;
    plan.kc = plan.bk;
    const auto* ii = find_loop(loop, "ii");
    const auto* jj = find_loop(loop, "jj");
    const auto* kk = find_loop(loop, "kk");
    const auto* i = find_loop(loop, "i");
    const auto* j = find_loop(loop, "j");
    const auto* k = find_loop(loop, "k");
    if (ii) {
        plan.bm = ii->step;
        plan.mc = ii->step;
        plan.threads = ii->kind == LoopOpKind::ParallelFor ? ii->threads : 1;
        plan.pin_threads = ii->pinned;
    }
    if (jj) { plan.bn = jj->step; plan.nc = jj->step; }
    if (kk) { plan.bk = kk->step; plan.kc = kk->step; }
    if (i) plan.mr = i->step;
    if (j) plan.nr = j->step;
    if (k) plan.unroll_k = k->step;
    plan.tiled = plan.bm < loop.problem.m || plan.bn < loop.problem.n || plan.bk < loop.problem.k;
    std::size_t traversal_index = 0;
    std::size_t kk_position = std::numeric_limits<std::size_t>::max();
    std::size_t jj_position = std::numeric_limits<std::size_t>::max();
    visit(loop.operations_, [&](const LoopOperation& operation) {
        if (operation.induction == "kk" && kk_position == std::numeric_limits<std::size_t>::max())
            kk_position = traversal_index;
        if (operation.induction == "jj" && jj_position == std::numeric_limits<std::size_t>::max())
            jj_position = traversal_index;
        ++traversal_index;
        if (operation.kind == LoopOpKind::Pack && operation.source == "%A") plan.pack_a = true;
        if (operation.kind == LoopOpKind::Pack && operation.source == "%B") plan.pack_b = true;
        if (operation.kind == LoopOpKind::Prefetch) plan.prefetch_distance = operation.distance;
        if (operation.kind == LoopOpKind::Gelu) plan.gelu = true;
        if (operation.kind == LoopOpKind::AddResidual) plan.residual = true;
        if (operation.kind == LoopOpKind::VectorLoad || operation.kind == LoopOpKind::VectorStore ||
            operation.kind == LoopOpKind::Fma) plan.vector_width = std::max(plan.vector_width, operation.width);
    });
    plan.order = kk_position < jj_position ? LoopOrder::IKJ : LoopOrder::IJK;
    const auto top_level_epilogue = std::any_of(loop.operations_.begin(), loop.operations_.end(),
        [](const LoopOperation& operation) {
            return operation.kind == LoopOpKind::AddBias || operation.kind == LoopOpKind::Relu;
        });
    plan.fused = !top_level_epilogue;
    loop.execution_plan_ = plan;
    return plan;
}

void verify_loop_ir(const LoopIR& loop) {
    if (loop.operations_.empty()) throw std::invalid_argument("LoopIR has no operations");
    bool has_fma = false;
    bool has_store = false;
    visit(loop.operations_, [&](const LoopOperation& operation) {
        if ((operation.kind == LoopOpKind::For || operation.kind == LoopOpKind::ParallelFor) && operation.step <= 0)
            throw std::invalid_argument("LoopIR loop step must be positive");
        if (operation.kind == LoopOpKind::Fma) has_fma = true;
        if (operation.kind == LoopOpKind::ScalarStore || operation.kind == LoopOpKind::VectorStore) has_store = true;
    });
    if (!has_fma || !has_store) throw std::invalid_argument("LoopIR requires explicit FMA and store operations");
}

LoopIR LowerToLoopsPass::run(const GraphIR& graph) const {
    Schedule schedule;
    schedule.order = LoopOrder::IJK;
    schedule.tiled = false;
    schedule.vector_width = 1;
    schedule.mr = 1;
    schedule.nr = 1;
    schedule.fused = false;
    return apply_schedule(graph.problem, schedule);
}

void LoopInterchangePass::run(LoopIR& loop) const {
    auto plan = analyze_loop_ir(loop);
    Schedule schedule;
    schedule.order = LoopOrder::IKJ;
    schedule.tiled = plan.tiled;
    schedule.bm = plan.bm; schedule.bn = plan.bn; schedule.bk = plan.bk;
    schedule.mr = plan.mr; schedule.nr = plan.nr; schedule.vector_width = plan.vector_width;
    schedule.threads = plan.threads; schedule.unroll_k = plan.unroll_k;
    schedule.pack_a = plan.pack_a; schedule.pack_b = plan.pack_b;
    schedule.prefetch_distance = plan.prefetch_distance; schedule.fused = plan.fused;
    schedule.pin_threads = plan.pin_threads;
    loop = apply_schedule(loop.problem, schedule);
}

void LoopTilingPass::run(LoopIR& loop, int bm, int bn, int bk) const {
    auto plan = analyze_loop_ir(loop);
    Schedule schedule;
    schedule.order = plan.order; schedule.tiled = true;
    schedule.bm = std::max(1, bm); schedule.bn = std::max(1, bn); schedule.bk = std::max(1, bk);
    schedule.mr = plan.mr; schedule.nr = plan.nr; schedule.vector_width = plan.vector_width;
    schedule.threads = plan.threads; schedule.unroll_k = plan.unroll_k;
    schedule.pack_a = plan.pack_a; schedule.pack_b = plan.pack_b;
    schedule.prefetch_distance = plan.prefetch_distance; schedule.fused = plan.fused;
    schedule.pin_threads = plan.pin_threads;
    loop = apply_schedule(loop.problem, schedule);
}

void VectorizePass::run(LoopIR& loop, int width) const {
    auto plan = analyze_loop_ir(loop);
    Schedule schedule;
    schedule.order = plan.order; schedule.tiled = plan.tiled;
    schedule.bm = plan.bm; schedule.bn = plan.bn; schedule.bk = plan.bk;
    schedule.mr = plan.mr; schedule.nr = std::max(1, width); schedule.vector_width = std::max(1, width);
    schedule.threads = plan.threads; schedule.unroll_k = plan.unroll_k;
    schedule.pack_a = plan.pack_a; schedule.pack_b = plan.pack_b;
    schedule.prefetch_distance = plan.prefetch_distance; schedule.fused = plan.fused;
    schedule.pin_threads = plan.pin_threads;
    loop = apply_schedule(loop.problem, schedule);
}

void FusionPass::run(LoopIR& loop) const {
    auto plan = analyze_loop_ir(loop);
    Schedule schedule;
    schedule.order = plan.order; schedule.tiled = plan.tiled;
    schedule.bm = plan.bm; schedule.bn = plan.bn; schedule.bk = plan.bk;
    schedule.mr = plan.mr; schedule.nr = plan.nr; schedule.vector_width = plan.vector_width;
    schedule.threads = plan.threads; schedule.unroll_k = plan.unroll_k;
    schedule.pack_a = plan.pack_a; schedule.pack_b = plan.pack_b;
    schedule.prefetch_distance = plan.prefetch_distance; schedule.fused = true;
    schedule.pin_threads = plan.pin_threads;
    loop = apply_schedule(loop.problem, schedule);
}

std::string schedule_name(const Schedule& schedule) {
    std::ostringstream out;
    out << (schedule.order == LoopOrder::IKJ ? "ikj" : "ijk");
    if (schedule.tiled) out << "-tile" << schedule.bm << 'x' << schedule.bn << 'x' << schedule.bk;
    if (schedule.pack_a || schedule.pack_b) out << "-pack" << (schedule.pack_a ? "a" : "") << (schedule.pack_b ? "b" : "");
    if (schedule.vector_width > 1) out << "-v" << schedule.vector_width;
    if (schedule.mr > 1) out << "-mr" << schedule.mr;
    if (schedule.threads > 1) out << "-t" << schedule.threads;
    if (schedule.fused) out << "-fused";
    return out.str();
}

}  // namespace schedforge
