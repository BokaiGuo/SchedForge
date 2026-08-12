#include "schedforge/schedforge.h"

#include <algorithm>
#include <sstream>

namespace schedforge {

std::string GraphIR::dump() const {
    std::ostringstream out;
    out << "module {\n"
        << "  %A = input tensor<" << problem.m << "x" << problem.k << "xf32>\n"
        << "  %B = input tensor<" << problem.k << "x" << problem.n << "xf32>\n";
    if (problem.bias) {
        out << "  %bias = input tensor<" << problem.n << "xf32>\n";
    }
    out << "  %0 = matmul %A, %B\n";
    if (problem.bias) {
        out << "  %1 = add %0, %bias\n";
    }
    if (problem.relu) {
        out << "  %2 = relu %" << (problem.bias ? "1" : "0") << "\n";
    }
    out << "  return %" << (problem.relu ? "2" : (problem.bias ? "1" : "0")) << "\n}\n";
    return out.str();
}

std::string LoopIR::dump() const {
    std::ostringstream out;
    out << "func @matmul_fused {\n";
    if (schedule.tiled) {
        out << "  tile<L2=" << schedule.mc << ',' << schedule.nc << ',' << schedule.kc
            << ";L1=" << schedule.bm << ',' << schedule.bn << ',' << schedule.bk << "> ";
    } else {
        out << "  ";
    }
    out << "order<" << (schedule.order == LoopOrder::IKJ ? "i-k-j" : "i-j-k") << ">";
    if (schedule.vector_width > 1) {
        out << " vector<" << schedule.vector_width << "xf32>";
    }
    if (schedule.threads > 1) {
        out << " parallel<" << schedule.threads << ">";
    }
    if (schedule.pack_a || schedule.pack_b) {
        out << " packing<" << (schedule.pack_a ? "pack_a" : "")
            << (schedule.pack_a && schedule.pack_b ? "," : "")
            << (schedule.pack_b ? "pack_b" : "") << ">";
    }
    if (schedule.prefetch_distance > 0) out << " prefetch<" << schedule.prefetch_distance << ">";
    out << " {\n"
        << "    %acc = reduce.mul_add %A, %B\n";
    if (problem.bias) {
        out << "    %acc = add %acc, %bias[j]\n";
    }
    if (problem.relu) {
        out << "    %acc = max %acc, 0.0\n";
    }
    out << "    store %acc, %C[i,j]\n  }\n}\n";
    return out.str();
}

LoopIR LowerToLoopsPass::run(const GraphIR& graph) const {
    LoopIR loop;
    loop.problem = graph.problem;
    loop.schedule.order = LoopOrder::IJK;
    loop.schedule.tiled = false;
    loop.schedule.vector_width = 1;
    loop.schedule.mr = 1;
    loop.schedule.nr = 1;
    loop.schedule.fused = false;
    return loop;
}

void LoopInterchangePass::run(LoopIR& loop) const { loop.schedule.order = LoopOrder::IKJ; }

void LoopTilingPass::run(LoopIR& loop, int bm, int bn, int bk) const {
    loop.schedule.tiled = true;
    loop.schedule.bm = std::max(1, bm);
    loop.schedule.bn = std::max(1, bn);
    loop.schedule.bk = std::max(1, bk);
}

void VectorizePass::run(LoopIR& loop, int width) const {
    loop.schedule.vector_width = std::max(1, width);
    loop.schedule.nr = loop.schedule.vector_width;
}

void FusionPass::run(LoopIR& loop) const { loop.schedule.fused = true; }

std::string schedule_name(const Schedule& schedule) {
    std::ostringstream out;
    out << (schedule.order == LoopOrder::IKJ ? "ikj" : "ijk");
    if (schedule.tiled) {
        out << "-tile" << schedule.bm << 'x' << schedule.bn << 'x' << schedule.bk;
    }
    if (schedule.pack_a || schedule.pack_b) {
        out << "-pack" << (schedule.pack_a ? "a" : "") << (schedule.pack_b ? "b" : "");
    }
    if (schedule.vector_width > 1) {
        out << "-v" << schedule.vector_width;
    }
    if (schedule.mr > 1) {
        out << "-mr" << schedule.mr;
    }
    if (schedule.threads > 1) {
        out << "-t" << schedule.threads;
    }
    if (schedule.fused) {
        out << "-fused";
    }
    return out.str();
}

}  // namespace schedforge
