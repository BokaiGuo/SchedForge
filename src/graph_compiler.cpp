#include "schedforge/graph_compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace schedforge {
namespace {

std::string op_name(GraphOpKind kind) {
    switch (kind) {
        case GraphOpKind::Input: return "tensor.input";
        case GraphOpKind::Constant: return "tensor.constant";
        case GraphOpKind::MatMul: return "tensor.matmul";
        case GraphOpKind::Add: return "tensor.add";
        case GraphOpKind::Multiply: return "tensor.multiply";
        case GraphOpKind::Maximum: return "tensor.maximum";
        case GraphOpKind::Gelu: return "tensor.gelu";
        case GraphOpKind::Broadcast: return "tensor.broadcast";
        case GraphOpKind::Reshape: return "tensor.reshape";
        case GraphOpKind::Transpose: return "tensor.transpose";
        case GraphOpKind::Reduce: return "tensor.reduce";
        case GraphOpKind::Exp: return "tensor.exp";
        case GraphOpKind::Rsqrt: return "tensor.rsqrt";
        case GraphOpKind::Convert: return "tensor.convert";
        case GraphOpKind::Softmax: return "tensor.softmax";
        case GraphOpKind::Mask: return "tensor.mask";
        case GraphOpKind::AttentionSdpa: return "attention.sdpa";
        case GraphOpKind::TopK: return "tensor.topk";
        case GraphOpKind::MoeHistogram: return "moe.histogram";
        case GraphOpKind::MoePrefixSum: return "moe.prefix_sum";
        case GraphOpKind::MoeDispatch: return "moe.dispatch";
        case GraphOpKind::MoeGroupedMatMul: return "moe.grouped_matmul";
        case GraphOpKind::SwiGLU: return "tensor.swiglu";
        case GraphOpKind::MoeCombine: return "moe.combine";
        case GraphOpKind::Return: return "func.return";
    }
    return "tensor.unknown";
}

bool same_dimension(const Dimension& lhs, const Dimension& rhs) {
    if (lhs.kind == DimensionKind::Static && rhs.kind == DimensionKind::Static)
        return lhs.value == rhs.value;
    return lhs.str() == rhs.str() || lhs.kind == DimensionKind::Dynamic ||
           rhs.kind == DimensionKind::Dynamic;
}

std::string value_name(int value) { return "%v" + std::to_string(value); }

std::size_t align_to(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

GraphTensorType parse_tensor_type(const std::string& text) {
    const auto begin = text.find("tensor<");
    const auto end = text.find('>', begin);
    if (begin == std::string::npos || end == std::string::npos)
        throw std::invalid_argument("missing tensor type: " + text);
    std::string body = text.substr(begin + 7, end - begin - 7);
    if (body.ends_with("xf32")) body.resize(body.size() - 4);
    std::vector<Dimension> shape;
    std::stringstream stream(body);
    std::string extent;
    while (std::getline(stream, extent, 'x')) {
        if (extent.empty()) continue;
        if (extent == "?") shape.push_back(Dimension::dynamic());
        else if (std::all_of(extent.begin(), extent.end(), ::isdigit))
            shape.push_back(Dimension::fixed(std::stoll(extent)));
        else shape.push_back(Dimension::symbolic(extent));
    }
    return {shape, DataType::F32, {}, std::nullopt, 0, -1};
}

float gelu(float value) {
    constexpr float factor = 0.7978845608F;
    return 0.5F * value * (1.0F + std::tanh(factor * (value + 0.044715F * value * value * value)));
}

Problem dispatch_problem(const TensorGraph& graph, const Dispatch& dispatch) {
    for (const int operation_index : dispatch.operations) {
        const auto& operation = graph.operations().at(static_cast<std::size_t>(operation_index));
        if (operation.kind != GraphOpKind::MatMul) continue;
        const auto& lhs = graph.values().at(static_cast<std::size_t>(operation.inputs[0])).type.shape;
        const auto& rhs = graph.values().at(static_cast<std::size_t>(operation.inputs[1])).type.shape;
        if (lhs.size() != 2 || rhs.size() != 2) continue;
        const int m = lhs[0].kind == DimensionKind::Static ? static_cast<int>(lhs[0].value) : 1;
        const int k = lhs[1].kind == DimensionKind::Static ? static_cast<int>(lhs[1].value) : 1;
        const int n = rhs[1].kind == DimensionKind::Static ? static_cast<int>(rhs[1].value) : 1;
        return {m, n, k, dispatch.epilogue.find("bias") != std::string::npos, false};
    }
    return {};
}

}  // namespace

Dimension Dimension::fixed(std::int64_t value) { return {DimensionKind::Static, value, {}}; }
Dimension Dimension::dynamic(std::string symbol) { return {DimensionKind::Dynamic, -1, std::move(symbol)}; }
Dimension Dimension::symbolic(std::string symbol) { return {DimensionKind::Symbolic, -1, std::move(symbol)}; }
std::string Dimension::str() const {
    if (kind == DimensionKind::Static) return std::to_string(value);
    return symbol.empty() ? "?" : symbol;
}

std::string TensorLayout::str() const {
    if (kind == TensorLayoutKind::ColumnMajor) return "column_major";
    if (kind == TensorLayoutKind::Blocked)
        return "blocked<" + std::to_string(block_m) + "x" + std::to_string(block_n) + ">";
    if (kind == TensorLayoutKind::PackedA) return "packed_a";
    if (kind == TensorLayoutKind::PackedB) return "packed_b";
    return "row_major";
}

std::string GraphTensorType::str() const {
    std::ostringstream out;
    out << "tensor<";
    for (const auto& dimension : shape) out << dimension.str() << 'x';
    out << data_type_name(dtype);
    if (quant_scale) {
        out << ", quant<scale=" << *quant_scale << ", zero_point=" << quant_zero_point;
        if (quant_axis >= 0) out << ", axis=" << quant_axis;
        out << '>';
    }
    out << ", layout=" << layout.str() << '>';
    return out.str();
}

std::optional<std::size_t> GraphTensorType::staticBytes() const {
    std::size_t elements = 1;
    for (const auto& dimension : shape) {
        if (dimension.kind != DimensionKind::Static || dimension.value < 0) return std::nullopt;
        elements *= static_cast<std::size_t>(dimension.value);
    }
    return elements * (dtype == DataType::I8 ? 1U : dtype == DataType::BF16 ? 2U : 4U);
}

std::string GraphOperation::str(const std::vector<GraphValue>& values) const {
    std::ostringstream out;
    if (!outputs.empty()) out << value_name(outputs.front()) << " = ";
    out << op_name(kind);
    for (const int input : inputs) out << ' ' << value_name(input);
    if (!outputs.empty()) out << " : " << values.at(static_cast<std::size_t>(outputs.front())).type.str();
    return out.str();
}

int TensorGraph::addInput(std::string name, GraphTensorType type, bool constant) {
    const int value_index = static_cast<int>(values_.size());
    const int operation_index = static_cast<int>(operations_.size());
    values_.push_back({std::move(name), std::move(type), operation_index, {}, constant});
    operations_.push_back({constant ? GraphOpKind::Constant : GraphOpKind::Input,
                           values_.back().name, {}, {value_index}, {}});
    return value_index;
}

int TensorGraph::addOperation(GraphOpKind kind, std::string name, std::vector<int> inputs,
                              GraphTensorType output_type) {
    const int operation_index = static_cast<int>(operations_.size());
    const int value_index = static_cast<int>(values_.size());
    values_.push_back({name + ".result", std::move(output_type), operation_index, {}, false});
    for (const int input : inputs) values_.at(static_cast<std::size_t>(input)).users.push_back(operation_index);
    operations_.push_back({kind, std::move(name), std::move(inputs), {value_index}, {}});
    return value_index;
}

void TensorGraph::setReturn(int value) {
    return_value_ = value;
    const int operation_index = static_cast<int>(operations_.size());
    values_.at(static_cast<std::size_t>(value)).users.push_back(operation_index);
    operations_.push_back({GraphOpKind::Return, "return", {value}, {}, {}});
}
int TensorGraph::returnValue() const { return return_value_; }
std::vector<GraphValue>& TensorGraph::values() { return values_; }
const std::vector<GraphValue>& TensorGraph::values() const { return values_; }
std::vector<GraphOperation>& TensorGraph::operations() { return operations_; }
const std::vector<GraphOperation>& TensorGraph::operations() const { return operations_; }
std::string TensorGraph::dump() const {
    std::ostringstream out;
    out << "graph @model {\n";
    for (const auto& operation : operations_) out << "  " << operation.str(values_) << '\n';
    out << "}\n";
    return out.str();
}

std::vector<ShapeConstraint> ShapeInferencePass::run(TensorGraph& graph) const {
    std::vector<ShapeConstraint> constraints;
    for (const auto& value : graph.values()) {
        for (const auto& dimension : value.type.shape) {
            if (dimension.kind != DimensionKind::Static) {
                constraints.push_back({dimension.str() + " > 0", true});
            }
        }
    }
    for (auto& operation : graph.operations()) {
        if (operation.outputs.empty()) continue;
        auto& output = graph.values().at(static_cast<std::size_t>(operation.outputs[0])).type;
        if (operation.kind == GraphOpKind::MatMul) {
            const auto& lhs = graph.values().at(static_cast<std::size_t>(operation.inputs[0])).type;
            const auto& rhs = graph.values().at(static_cast<std::size_t>(operation.inputs[1])).type;
            if (lhs.shape.size() == 2 && rhs.shape.size() == 2) {
                if (!same_dimension(lhs.shape[1], rhs.shape[0]))
                    throw std::invalid_argument("matmul reduction dimensions do not match");
                constraints.push_back({lhs.shape[1].str() + " == " + rhs.shape[0].str(),
                                       lhs.shape[1].kind != DimensionKind::Static ||
                                       rhs.shape[0].kind != DimensionKind::Static});
                output = {{lhs.shape[0], rhs.shape[1]}, lhs.dtype, {}, std::nullopt, 0, -1};
            } else if (lhs.shape.size() == 4 && rhs.shape.size() == 4) {
                if (!same_dimension(lhs.shape[0], rhs.shape[0]) ||
                    !same_dimension(lhs.shape[3], rhs.shape[2]))
                    throw std::invalid_argument("batched matmul dimensions do not match");
                if (lhs.shape[1].kind == DimensionKind::Static &&
                    rhs.shape[1].kind == DimensionKind::Static &&
                    lhs.shape[1].value % rhs.shape[1].value != 0)
                    throw std::invalid_argument("batched matmul head dimensions do not broadcast");
                constraints.push_back({lhs.shape[3].str() + " == " + rhs.shape[2].str(),
                                       lhs.shape[3].kind != DimensionKind::Static ||
                                       rhs.shape[2].kind != DimensionKind::Static});
                constraints.push_back({lhs.shape[1].str() + " % " + rhs.shape[1].str() + " == 0",
                                       lhs.shape[1].kind != DimensionKind::Static ||
                                       rhs.shape[1].kind != DimensionKind::Static});
                output = {{lhs.shape[0], lhs.shape[1], lhs.shape[2], rhs.shape[3]},
                          lhs.dtype, {}, std::nullopt, 0, -1};
            } else {
                throw std::invalid_argument("matmul requires matching rank-2 or rank-4 tensors");
            }
        } else if (operation.kind == GraphOpKind::Add || operation.kind == GraphOpKind::Multiply ||
                   operation.kind == GraphOpKind::Maximum) {
            const auto& lhs = graph.values().at(static_cast<std::size_t>(operation.inputs[0])).type;
            const auto& rhs = graph.values().at(static_cast<std::size_t>(operation.inputs[1])).type;
            output = lhs.shape.size() >= rhs.shape.size() ? lhs : rhs;
        } else if (operation.kind == GraphOpKind::AttentionSdpa) {
            const auto& query = graph.values().at(static_cast<std::size_t>(operation.inputs[0])).type;
            const auto& key = graph.values().at(static_cast<std::size_t>(operation.inputs[1])).type;
            const auto& value = graph.values().at(static_cast<std::size_t>(operation.inputs[2])).type;
            if (query.shape.size() != 4 || key.shape.size() != 4 || value.shape.size() != 4)
                throw std::invalid_argument("attention.sdpa requires rank-4 Q/K/V tensors");
            if (!same_dimension(query.shape[0], key.shape[0]) ||
                !same_dimension(key.shape[0], value.shape[0]) ||
                !same_dimension(query.shape[3], key.shape[3]) ||
                !same_dimension(key.shape[2], value.shape[2]))
                throw std::invalid_argument("attention.sdpa Q/K/V dimensions do not match");
            output = {{query.shape[0], query.shape[1], query.shape[2], value.shape[3]},
                      query.dtype, {}, std::nullopt, 0, -1};
            constraints.push_back({query.shape[1].str() + " % " + key.shape[1].str() + " == 0",
                                   query.shape[1].kind != DimensionKind::Static ||
                                   key.shape[1].kind != DimensionKind::Static});
        } else if (operation.kind == GraphOpKind::Gelu || operation.kind == GraphOpKind::Exp ||
                   operation.kind == GraphOpKind::Rsqrt || operation.kind == GraphOpKind::Convert ||
                   operation.kind == GraphOpKind::Broadcast || operation.kind == GraphOpKind::Softmax ||
                   operation.kind == GraphOpKind::Mask ||
                   operation.kind == GraphOpKind::SwiGLU || operation.kind == GraphOpKind::MoeCombine) {
            if (output.shape.empty())
                output = graph.values().at(static_cast<std::size_t>(operation.inputs[0])).type;
        }
    }
    return constraints;
}

TensorGraph GraphCanonicalizationPass::run(const TensorGraph& graph) const {
    std::vector<int> matmuls;
    for (std::size_t index = 0; index < graph.operations().size(); ++index) {
        if (graph.operations()[index].kind == GraphOpKind::MatMul)
            matmuls.push_back(static_cast<int>(index));
    }
    if (matmuls.size() != 2 || graph.values().size() < 5) return graph;

    const auto& first = graph.operations().at(static_cast<std::size_t>(matmuls[0]));
    const auto& second = graph.operations().at(static_cast<std::size_t>(matmuls[1]));
    if (first.inputs.size() != 2 || second.inputs.size() != 2) return graph;
    const int first_output = first.outputs.front();
    int first_bias = -1;
    int first_bias_output = -1;
    int activation_output = -1;
    int second_bias = -1;
    int second_bias_output = -1;
    int residual_input = -1;
    for (const auto& operation : graph.operations()) {
        if (operation.kind != GraphOpKind::Add || operation.inputs.size() != 2) continue;
        if (std::find(operation.inputs.begin(), operation.inputs.end(), first_output) != operation.inputs.end()) {
            first_bias = operation.inputs[0] == first_output ? operation.inputs[1] : operation.inputs[0];
            first_bias_output = operation.outputs.front();
        }
        if (std::find(operation.inputs.begin(), operation.inputs.end(), second.outputs.front()) != operation.inputs.end()) {
            second_bias = operation.inputs[0] == second.outputs.front() ? operation.inputs[1] : operation.inputs[0];
            second_bias_output = operation.outputs.front();
        }
    }
    for (const auto& operation : graph.operations()) {
        if (operation.kind == GraphOpKind::Gelu && operation.inputs.size() == 1 &&
            operation.inputs.front() == first_bias_output && !operation.outputs.empty()) {
            activation_output = operation.outputs.front();
            break;
        }
    }
    for (const auto& operation : graph.operations()) {
        if (operation.kind != GraphOpKind::Add || operation.inputs.size() != 2 ||
            operation.outputs.empty() || operation.outputs.front() != graph.returnValue())
            continue;
        if (operation.inputs[0] == second_bias_output) residual_input = operation.inputs[1];
        else if (operation.inputs[1] == second_bias_output) residual_input = operation.inputs[0];
    }
    if (first_bias < 0 || activation_output < 0 || second_bias < 0 || residual_input < 0 ||
        second.inputs[0] != activation_output)
        return graph;

    TensorGraph canonical;
    std::unordered_map<int, int> remap;
    for (const int original : {first.inputs[0], first.inputs[1], first_bias,
                               second.inputs[1], second_bias}) {
        if (remap.contains(original)) continue;
        const auto& value = graph.values().at(static_cast<std::size_t>(original));
        remap[original] = canonical.addInput(value.name, value.type, value.constant);
    }
    const int mm1 = canonical.addOperation(GraphOpKind::MatMul, "linear1",
        {remap[first.inputs[0]], remap[first.inputs[1]]});
    const int add1 = canonical.addOperation(GraphOpKind::Add, "bias1", {mm1, remap[first_bias]});
    const int activation = canonical.addOperation(GraphOpKind::Gelu, "gelu", {add1});
    const int mm2 = canonical.addOperation(GraphOpKind::MatMul, "linear2",
        {activation, remap[second.inputs[1]]});
    const int add2 = canonical.addOperation(GraphOpKind::Add, "bias2", {mm2, remap[second_bias]});
    const int residual = canonical.addOperation(GraphOpKind::Add, "residual",
        {add2, remap.contains(residual_input) ? remap[residual_input] : remap[first.inputs[0]]});
    canonical.setReturn(residual);
    ShapeInferencePass{}.run(canonical);
    return canonical;
}

std::string StructuredCompute::dump() const {
    std::ostringstream out;
    out << "tensor.compute @" << name << " iterators(";
    for (std::size_t index = 0; index < iterators.size(); ++index) {
        if (index) out << ", ";
        out << iterators[index] << ':' << (iterator_kinds[index] == IteratorKind::Reduction ? "reduction" : "parallel");
    }
    out << ") maps(";
    for (std::size_t index = 0; index < indexing_maps.size(); ++index) {
        if (index) out << ", ";
        out << indexing_maps[index];
    }
    out << ") { " << body << " }";
    return out.str();
}

std::vector<StructuredCompute> StructuredComputeLowering::run(const TensorGraph& graph) const {
    std::vector<StructuredCompute> computes;
    for (const auto& operation : graph.operations()) {
        if (operation.kind == GraphOpKind::MatMul) {
            computes.push_back({operation.name, {"i", "j", "k"},
                {IteratorKind::Parallel, IteratorKind::Parallel, IteratorKind::Reduction},
                {"A(i,k)", "B(k,j)", "C(i,j)"}, "C(i,j) += A(i,k) * B(k,j)"});
        } else if (operation.kind == GraphOpKind::Add || operation.kind == GraphOpKind::Gelu ||
                   operation.kind == GraphOpKind::Softmax || operation.kind == GraphOpKind::Mask ||
                   operation.kind == GraphOpKind::AttentionSdpa || operation.kind == GraphOpKind::SwiGLU ||
                   operation.kind == GraphOpKind::MoeCombine) {
            computes.push_back({operation.name, {"i", "j"},
                {IteratorKind::Parallel, IteratorKind::Parallel},
                {"input(i,j)", "output(i,j)"}, op_name(operation.kind)});
        } else if (operation.kind == GraphOpKind::MoeGroupedMatMul) {
            computes.push_back({operation.name, {"e", "i", "j", "k"},
                {IteratorKind::Parallel, IteratorKind::Parallel,
                 IteratorKind::Parallel, IteratorKind::Reduction},
                {"X(offset[e]+i,k)", "W(e,k,j)", "Y(offset[e]+i,j)"},
                "Y_e(i,j) += X_e(i,k) * W_e(k,j)"});
        } else if (operation.kind == GraphOpKind::AttentionSdpa) {
            computes.push_back({operation.name, {"b", "h", "qi", "kj", "d", "dv"},
                {IteratorKind::Parallel, IteratorKind::Parallel, IteratorKind::Parallel,
                 IteratorKind::Reduction, IteratorKind::Reduction, IteratorKind::Parallel},
                {"Q(b,h,qi,d)", "K(b,hkv,kj,d)", "V(b,hkv,kj,dv)", "O(b,h,qi,dv)"},
                "attention.qk -> reduce.max -> vector.exp -> reduce.sum -> attention.pv"});
        }
    }
    return computes;
}

std::string TransformOperation::str() const {
    auto integer_list = [&] {
        std::ostringstream out;
        for (std::size_t index = 0; index < integers.size(); ++index) {
            if (index) out << ',';
            out << integers[index];
        }
        return out.str();
    };
    switch (kind) {
        case TransformKind::Match: return "match " + text;
        case TransformKind::Tile: return "tile [" + integer_list() + "]";
        case TransformKind::Interchange: return "interchange [" + integer_list() + "]";
        case TransformKind::RegisterTile: return "register_tile [" + integer_list() + "]";
        case TransformKind::Vectorize: return "vectorize width=" + integer_list();
        case TransformKind::Unroll: return "unroll factor=" + integer_list();
        case TransformKind::Pack: return "pack " + text;
        case TransformKind::Prefetch: return "prefetch distance=" + integer_list();
        case TransformKind::Parallelize: return "parallelize threads=" + integer_list();
        case TransformKind::Tensorize: return "tensorize " + text;
    }
    return {};
}

void TransformProgram::add(TransformOperation operation) {
    operations_.push_back(std::move(operation));
}
const std::vector<TransformOperation>& TransformProgram::operations() const { return operations_; }

Schedule TransformProgram::replay(Schedule schedule) const {
    for (const auto& operation : operations_) {
        switch (operation.kind) {
            case TransformKind::Tile:
                if (operation.integers.size() >= 3) {
                    schedule.bm = operation.integers[0];
                    schedule.bn = operation.integers[1];
                    schedule.bk = operation.integers[2];
                    schedule.tiled = true;
                }
                break;
            case TransformKind::Interchange:
                schedule.order = LoopOrder::IKJ;
                break;
            case TransformKind::RegisterTile:
                if (operation.integers.size() >= 2) {
                    schedule.mr = operation.integers[0];
                    schedule.nr = operation.integers[1];
                }
                break;
            case TransformKind::Vectorize:
                if (!operation.integers.empty()) schedule.vector_width = operation.integers[0];
                break;
            case TransformKind::Unroll:
                if (!operation.integers.empty()) schedule.unroll_k = operation.integers[0];
                break;
            case TransformKind::Pack:
                schedule.pack_a = operation.text.find('a') != std::string::npos;
                schedule.pack_b = operation.text.find('b') != std::string::npos;
                break;
            case TransformKind::Prefetch:
                if (!operation.integers.empty()) schedule.prefetch_distance = operation.integers[0];
                break;
            case TransformKind::Parallelize:
                if (!operation.integers.empty()) schedule.threads = operation.integers[0];
                schedule.pin_threads = schedule.threads > 1;
                break;
            case TransformKind::Tensorize:
                schedule.fused = true;
                break;
            case TransformKind::Match:
                break;
        }
    }
    return schedule;
}

std::string TransformProgram::dump() const {
    std::ostringstream out;
    out << "transform.sequence {\n";
    for (const auto& operation : operations_) out << "  " << operation.str() << '\n';
    out << '}';
    return out.str();
}

TransformProgram TransformProgram::fromSchedule(const Schedule& schedule) {
    TransformProgram program;
    program.add({TransformKind::Match, {}, "tensor.matmul"});
    program.add({TransformKind::Tile, {schedule.bm, schedule.bn, schedule.bk}, {}});
    if (schedule.order == LoopOrder::IKJ)
        program.add({TransformKind::Interchange, {0, 2, 1}, {}});
    program.add({TransformKind::RegisterTile, {schedule.mr, schedule.nr}, {}});
    program.add({TransformKind::Vectorize, {schedule.vector_width}, {}});
    program.add({TransformKind::Unroll, {schedule.unroll_k}, {}});
    if (schedule.pack_a || schedule.pack_b)
        program.add({TransformKind::Pack, {},
            std::string(schedule.pack_a ? "a" : "") + (schedule.pack_b ? "b" : "")});
    if (schedule.prefetch_distance > 0)
        program.add({TransformKind::Prefetch, {schedule.prefetch_distance}, {}});
    program.add({TransformKind::Parallelize, {schedule.threads}, {}});
    program.add({TransformKind::Tensorize, {},
        "avx2_f32_m" + std::to_string(schedule.mr) + "n" + std::to_string(schedule.nr)});
    return program;
}

TransformProgram TransformProgram::parse(const std::string& text) {
    TransformProgram program;
    std::stringstream lines(text);
    std::string line;
    const std::regex integer_pattern(R"(-?[0-9]+)");
    while (std::getline(lines, line)) {
        std::vector<int> integers;
        for (auto iterator = std::sregex_iterator(line.begin(), line.end(), integer_pattern);
             iterator != std::sregex_iterator(); ++iterator)
            integers.push_back(std::stoi(iterator->str()));
        if (line.find("match ") != std::string::npos)
            program.add({TransformKind::Match, {}, line.substr(line.find("match ") + 6)});
        else if (line.find("register_tile") != std::string::npos)
            program.add({TransformKind::RegisterTile, integers, {}});
        else if (line.find("tile ") != std::string::npos)
            program.add({TransformKind::Tile, integers, {}});
        else if (line.find("interchange") != std::string::npos)
            program.add({TransformKind::Interchange, integers, {}});
        else if (line.find("vectorize") != std::string::npos)
            program.add({TransformKind::Vectorize, integers, {}});
        else if (line.find("unroll") != std::string::npos)
            program.add({TransformKind::Unroll, integers, {}});
        else if (line.find("prefetch") != std::string::npos)
            program.add({TransformKind::Prefetch, integers, {}});
        else if (line.find("parallelize") != std::string::npos)
            program.add({TransformKind::Parallelize, integers, {}});
        else if (line.find("tensorize ") != std::string::npos)
            program.add({TransformKind::Tensorize, {}, line.substr(line.find("tensorize ") + 10)});
        else if (line.find("pack ") != std::string::npos)
            program.add({TransformKind::Pack, {}, line.substr(line.find("pack ") + 5)});
    }
    return program;
}

std::optional<TensorIntrinsic> TensorizationPass::match(
    const StructuredCompute& compute, const Schedule& schedule,
    const TargetInfo& target) const {
    if (compute.body.find("+=") == std::string::npos || schedule.vector_width <= 1)
        return std::nullopt;
    TensorIntrinsic intrinsic;
    intrinsic.name = "avx2_f32_m" + std::to_string(schedule.mr) +
                     "n" + std::to_string(schedule.nr);
    intrinsic.mr = schedule.mr;
    intrinsic.nr = schedule.nr;
    intrinsic.vector_width = schedule.vector_width;
    intrinsic.isa = target.isa;
    return intrinsic;
}

std::size_t QuantizationPropagationPass::run(TensorGraph& graph) const {
    std::size_t propagated = 0;
    for (auto& operation : graph.operations()) {
        if (operation.outputs.empty() || operation.inputs.empty()) continue;
        if (operation.kind != GraphOpKind::Add && operation.kind != GraphOpKind::Maximum &&
            operation.kind != GraphOpKind::MatMul) continue;
        const auto& input_type = graph.values().at(static_cast<std::size_t>(operation.inputs[0])).type;
        if (!input_type.quant_scale) continue;
        auto& output_type = graph.values().at(static_cast<std::size_t>(operation.outputs[0])).type;
        output_type.dtype = input_type.dtype;
        output_type.quant_scale = input_type.quant_scale;
        output_type.quant_zero_point = input_type.quant_zero_point;
        output_type.quant_axis = input_type.quant_axis;
        ++propagated;
    }
    return propagated;
}

bool FusionPlanner::legal(const TensorGraph& graph, int producer, int consumer) const {
    const auto& producer_op = graph.operations().at(static_cast<std::size_t>(producer));
    const auto& consumer_op = graph.operations().at(static_cast<std::size_t>(consumer));
    if (producer_op.outputs.empty()) return false;
    const auto& value = graph.values().at(static_cast<std::size_t>(producer_op.outputs[0]));
    if (value.users.size() != 1 || value.users.front() != consumer) return false;
    return consumer_op.kind == GraphOpKind::Add || consumer_op.kind == GraphOpKind::Gelu ||
           consumer_op.kind == GraphOpKind::Maximum;
}

FusionCost FusionPlanner::profitability(const TensorGraph& graph,
                                        const std::vector<int>& operations) const {
    double saved = 0.0;
    for (std::size_t index = 0; index + 1 < operations.size(); ++index) {
        const auto& operation = graph.operations().at(static_cast<std::size_t>(operations[index]));
        if (!operation.outputs.empty())
            saved += static_cast<double>(graph.values().at(static_cast<std::size_t>(operation.outputs[0])).type.staticBytes().value_or(0));
    }
    const double pressure = 2.0 + static_cast<double>(operations.size()) * 1.5;
    return {saved, 0.0, pressure, 1.0 + saved / (saved + 65536.0) - pressure * 0.01};
}

std::string Dispatch::dump(const TensorGraph& graph) const {
    std::ostringstream out;
    out << "dispatch @" << name << " [epilogue=" << epilogue << "] {\n";
    for (const int index : operations)
        out << "  " << graph.operations().at(static_cast<std::size_t>(index)).str(graph.values()) << '\n';
    out << "}\n  " << transforms.dump();
    if (intrinsic) out << "\n  intrinsic @" << intrinsic->name;
    out << "\n cost<saved_bytes=" << fusion_cost.saved_memory_bytes
        << ", register_pressure=" << fusion_cost.register_pressure
        << ", speedup=" << fusion_cost.estimated_speedup << ">";
    return out.str();
}

std::vector<Dispatch> FusionPlanner::run(const TensorGraph& graph) const {
    std::vector<Dispatch> dispatches;
    std::vector<bool> consumed(graph.operations().size(), false);
    for (std::size_t index = 0; index < graph.operations().size(); ++index) {
        if (consumed[index] || graph.operations()[index].kind != GraphOpKind::MatMul) continue;
        Dispatch dispatch;
        dispatch.name = "dispatch_" + std::to_string(dispatches.size());
        dispatch.operations.push_back(static_cast<int>(index));
        consumed[index] = true;
        std::size_t current = index;
        while (current + 1 < graph.operations().size() &&
               legal(graph, static_cast<int>(current), static_cast<int>(current + 1))) {
            ++current;
            dispatch.operations.push_back(static_cast<int>(current));
            consumed[current] = true;
        }
        for (const int operation_index : dispatch.operations) {
            const auto& operation = graph.operations().at(static_cast<std::size_t>(operation_index));
            if (operation.kind == GraphOpKind::Add) {
                const bool bias = std::any_of(operation.inputs.begin(), operation.inputs.end(),
                    [&](int value) {
                        const auto& input = graph.values().at(static_cast<std::size_t>(value));
                        return input.type.shape.size() == 1;
                    });
                const std::string add_kind = bias ? "bias" : "residual";
                dispatch.epilogue += dispatch.epilogue.empty() ? add_kind : "+" + add_kind;
            }
            if (operation.kind == GraphOpKind::Gelu) dispatch.epilogue += dispatch.epilogue.empty() ? "gelu" : "+gelu";
            for (const int input : operation.inputs) {
                const int producer = graph.values().at(static_cast<std::size_t>(input)).producer;
                if (std::find(dispatch.operations.begin(), dispatch.operations.end(), producer) == dispatch.operations.end() &&
                    std::find(dispatch.inputs.begin(), dispatch.inputs.end(), input) == dispatch.inputs.end())
                    dispatch.inputs.push_back(input);
            }
        }
        const auto& last = graph.operations().at(static_cast<std::size_t>(dispatch.operations.back()));
        if (!last.outputs.empty()) dispatch.outputs = last.outputs;
        dispatch.fusion_cost = profitability(graph, dispatch.operations);
        dispatches.push_back(std::move(dispatch));
    }
    return dispatches;
}

std::size_t GraphLayoutPlanner::run(TensorGraph& graph, std::vector<Dispatch>& dispatches) const {
    std::size_t removed_conversions = 0;
    for (std::size_t index = 0; index + 1 < dispatches.size(); ++index) {
        if (dispatches[index].outputs.empty()) continue;
        auto& value = graph.values().at(static_cast<std::size_t>(dispatches[index].outputs[0]));
        value.type.layout = {TensorLayoutKind::Blocked, 6, 16};
        ++removed_conversions;
    }
    return removed_conversions;
}

BufferizationResult GraphBufferizer::run(const TensorGraph& graph,
                                         const std::vector<Dispatch>& dispatches) const {
    BufferizationResult result;
    std::vector<GraphBufferPlan> temporaries;
    std::vector<bool> materialized(graph.values().size(), false);
    for (const auto& dispatch : dispatches) {
        for (const int input : dispatch.inputs) materialized[static_cast<std::size_t>(input)] = true;
        for (const int output : dispatch.outputs) materialized[static_cast<std::size_t>(output)] = true;
    }
    if (graph.returnValue() >= 0)
        materialized[static_cast<std::size_t>(graph.returnValue())] = true;
    for (std::size_t value_index = 0; value_index < graph.values().size(); ++value_index) {
        const auto& value = graph.values()[value_index];
        const std::size_t bytes = value.type.staticBytes().value_or(0);
        if (bytes == 0) continue;
        const bool produced_tensor = value.producer >= 0 &&
            graph.operations().at(static_cast<std::size_t>(value.producer)).kind != GraphOpKind::Input &&
            graph.operations().at(static_cast<std::size_t>(value.producer)).kind != GraphOpKind::Constant;
        if (produced_tensor && static_cast<int>(value_index) != graph.returnValue())
            result.naive_bytes += bytes;
        if (!materialized[value_index] && produced_tensor) continue;
        const bool external = value.constant || value.producer < 0 ||
            graph.operations().at(static_cast<std::size_t>(value.producer)).kind == GraphOpKind::Input ||
            static_cast<int>(value_index) == graph.returnValue();
        int first = static_cast<int>(dispatches.size());
        int last = -1;
        for (std::size_t dispatch_index = 0; dispatch_index < dispatches.size(); ++dispatch_index) {
            const auto& dispatch = dispatches[dispatch_index];
            if (std::find(dispatch.inputs.begin(), dispatch.inputs.end(), static_cast<int>(value_index)) != dispatch.inputs.end() ||
                std::find(dispatch.outputs.begin(), dispatch.outputs.end(), static_cast<int>(value_index)) != dispatch.outputs.end()) {
                first = std::min(first, static_cast<int>(dispatch_index));
                last = std::max(last, static_cast<int>(dispatch_index));
            }
        }
        GraphBufferPlan plan{value.name, bytes, 0, 64, std::max(0, first), std::max(0, last),
                             static_cast<int>(value_index), external, value.type.layout};
        result.buffers.push_back(plan);
        if (!external) {
            temporaries.push_back(plan);
        }
    }
    struct Slot { std::size_t offset; std::size_t bytes; int last; };
    std::vector<Slot> slots;
    for (auto& temporary : temporaries) {
        auto reusable = std::find_if(slots.begin(), slots.end(), [&](const Slot& slot) {
            return slot.last < temporary.first_dispatch && slot.bytes >= temporary.bytes;
        });
        if (reusable == slots.end()) {
            const std::size_t offset = align_to(result.workspace_bytes, temporary.alignment);
            slots.push_back({offset, temporary.bytes, temporary.last_dispatch});
            temporary.offset = offset;
            result.workspace_bytes = offset + temporary.bytes;
        } else {
            temporary.offset = reusable->offset;
            reusable->last = temporary.last_dispatch;
        }
        auto original = std::find_if(result.buffers.begin(), result.buffers.end(), [&](const GraphBufferPlan& plan) {
            return plan.value == temporary.value;
        });
        original->offset = temporary.offset;
    }
    return result;
}

std::string BufferizationResult::dump() const {
    std::ostringstream out;
    out << "bufferization naive=" << naive_bytes << " workspace=" << workspace_bytes << '\n';
    for (const auto& buffer : buffers)
        out << "  buffer @" << buffer.name << " bytes=" << buffer.bytes << " offset=" << buffer.offset
            << " lifetime=[" << buffer.first_dispatch << ',' << buffer.last_dispatch << "] layout="
            << buffer.layout.str() << (buffer.external ? " external" : " workspace") << '\n';
    return out.str();
}

std::string ExecutablePlan::dump() const {
    std::ostringstream out;
    out << "executable target=" << target << " hardware=\"" << hardware << "\" {\n"
        << graph.dump();
    out << "structured_compute {\n";
    for (const auto& compute : structured_computes) out << "  " << compute.dump() << '\n';
    out << "}\ndispatches {\n";
    for (std::size_t index = 0; index < dispatches.size(); ++index) {
        out << dispatches[index].dump(graph) << '\n';
        if (index < scheduled_loops.size())
            out << "  scheduled_loop @dispatch_" << index << " {\n"
                << scheduled_loops[index].dump() << "  }\n";
    }
    out << "}\nlayout_conversions_removed=" << layout_conversions_removed << '\n'
        << memory.dump() << "shape_guards {\n";
    for (const auto& guard : guards) out << "  if " << guard.expression << " -> " << guard.target << '\n';
    out << "}\nllvm_kernels=" << llvm_ir.size()
        << " compile_ms=" << llvm_compile_milliseconds << "\n";
    for (std::size_t index = 0; index < llvm_ir.size(); ++index)
        out << "llvm.kernel @dispatch_" << index << " {\n" << llvm_ir[index] << "\n}\n";
    out << "}\n";
    return out.str();
}

void ExecutablePlan::save(const std::filesystem::path& path) const {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write executable plan: " + path.string());
    output << dump();
}

TensorGraph build_transformer_mlp_graph(const MLPConfig& config, bool dynamic_batch) {
    TensorGraph graph;
    const Dimension rows = dynamic_batch ? Dimension::symbolic("B*S")
                                         : Dimension::fixed(config.batch * config.sequence);
    const int x = graph.addInput("x", {{rows, Dimension::fixed(config.hidden)}, DataType::F32, {}, std::nullopt, 0, -1});
    const int w1 = graph.addInput("w1", {{Dimension::fixed(config.hidden), Dimension::fixed(config.intermediate)}, DataType::F32, {}, std::nullopt, 0, -1}, true);
    const int b1 = graph.addInput("b1", {{Dimension::fixed(config.intermediate)}, DataType::F32, {}, std::nullopt, 0, -1}, true);
    const int w2 = graph.addInput("w2", {{Dimension::fixed(config.intermediate), Dimension::fixed(config.hidden)}, DataType::F32, {}, std::nullopt, 0, -1}, true);
    const int b2 = graph.addInput("b2", {{Dimension::fixed(config.hidden)}, DataType::F32, {}, std::nullopt, 0, -1}, true);
    const int mm1 = graph.addOperation(GraphOpKind::MatMul, "linear1", {x, w1});
    const int add1 = graph.addOperation(GraphOpKind::Add, "bias1", {mm1, b1});
    const int activation = graph.addOperation(GraphOpKind::Gelu, "gelu", {add1});
    const int mm2 = graph.addOperation(GraphOpKind::MatMul, "linear2", {activation, w2});
    const int add2 = graph.addOperation(GraphOpKind::Add, "bias2", {mm2, b2});
    const int residual = graph.addOperation(GraphOpKind::Add, "residual", {add2, x});
    graph.setReturn(residual);
    return graph;
}

TensorGraph build_mini_attention_graph(int sequence, int hidden, bool dynamic_sequence) {
    TensorGraph graph;
    const Dimension rows = dynamic_sequence ? Dimension::symbolic("S") : Dimension::fixed(sequence);
    const GraphTensorType activations{{rows, Dimension::fixed(hidden)}, DataType::F32, {}, std::nullopt, 0, -1};
    const GraphTensorType weights{{Dimension::fixed(hidden), Dimension::fixed(hidden)}, DataType::F32, {}, std::nullopt, 0, -1};
    const int x = graph.addInput("x", activations);
    const int wq = graph.addInput("wq", weights, true);
    const int wk = graph.addInput("wk", weights, true);
    const int wv = graph.addInput("wv", weights, true);
    const int q = graph.addOperation(GraphOpKind::MatMul, "query", {x, wq});
    const int k = graph.addOperation(GraphOpKind::MatMul, "key", {x, wk});
    const int v = graph.addOperation(GraphOpKind::MatMul, "value", {x, wv});
    const int kt = graph.addOperation(GraphOpKind::Transpose, "key_transpose", {k},
        {{Dimension::fixed(hidden), rows}, DataType::F32, {}, std::nullopt, 0, -1});
    const int scores = graph.addOperation(GraphOpKind::MatMul, "attention_scores", {q, kt});
    const int exponentials = graph.addOperation(GraphOpKind::Exp, "softmax_exp", {scores});
    const int denominator = graph.addOperation(GraphOpKind::Reduce, "softmax_sum", {exponentials});
    const int inverse = graph.addOperation(GraphOpKind::Rsqrt, "softmax_inverse", {denominator});
    const int probabilities = graph.addOperation(GraphOpKind::Multiply, "softmax_normalize",
        {exponentials, inverse});
    const int context = graph.addOperation(GraphOpKind::MatMul, "attention_value",
        {probabilities, v});
    graph.setReturn(context);
    return graph;
}

MLPData make_mlp_data(const MLPConfig& config, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    MLPData data;
    data.input.resize(static_cast<std::size_t>(config.batch * config.sequence * config.hidden));
    data.weight1.resize(static_cast<std::size_t>(config.hidden * config.intermediate));
    data.bias1.resize(static_cast<std::size_t>(config.intermediate));
    data.weight2.resize(static_cast<std::size_t>(config.intermediate * config.hidden));
    data.bias2.resize(static_cast<std::size_t>(config.hidden));
    for (auto* values : {&data.input, &data.weight1, &data.bias1, &data.weight2, &data.bias2})
        for (float& value : *values) value = distribution(generator);
    return data;
}

std::vector<float> reference_mlp(const MLPConfig& config, const MLPData& data) {
    const int rows = config.batch * config.sequence;
    std::vector<float> hidden(static_cast<std::size_t>(rows * config.intermediate));
    std::vector<float> output(static_cast<std::size_t>(rows * config.hidden));
    for (int i = 0; i < rows; ++i) for (int j = 0; j < config.intermediate; ++j) {
        float value = data.bias1[static_cast<std::size_t>(j)];
        for (int k = 0; k < config.hidden; ++k)
            value += data.input[static_cast<std::size_t>(i) * config.hidden + k] *
                     data.weight1[static_cast<std::size_t>(k) * config.intermediate + j];
        hidden[static_cast<std::size_t>(i) * config.intermediate + j] = gelu(value);
    }
    for (int i = 0; i < rows; ++i) for (int j = 0; j < config.hidden; ++j) {
        float value = data.bias2[static_cast<std::size_t>(j)] +
                      data.input[static_cast<std::size_t>(i) * config.hidden + j];
        for (int k = 0; k < config.intermediate; ++k)
            value += hidden[static_cast<std::size_t>(i) * config.intermediate + k] *
                     data.weight2[static_cast<std::size_t>(k) * config.hidden + j];
        output[static_cast<std::size_t>(i) * config.hidden + j] = value;
    }
    return output;
}

GraphBenchmarkResult execute_mlp(const ExecutablePlan& plan, const MLPConfig& config,
                                 const MLPData& data, int warmup, int repetitions) {
    if (plan.dispatches.size() != 2 || plan.scheduled_loops.size() != 2)
        throw std::invalid_argument("MLP plan requires two compiled dispatch loops");
    std::vector<float> hidden;
    std::vector<float> output;
    auto run = [&] {
        TensorData first{data.input, data.weight1, data.bias1, {}, {}};
        execute(plan.scheduled_loops[0], first, hidden);
        TensorData second{hidden, data.weight2, data.bias2, {}, data.input};
        execute(plan.scheduled_loops[1], second, output);
    };
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) run();
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        run();
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(timings.begin(), timings.end());
    return {timings[timings.size() / 2], max_abs_error(reference_mlp(config, data), output), output};
}

void MeasurementDatabase::add(ScheduleMeasurement measurement) {
    records_.push_back(std::move(measurement));
}
const std::vector<ScheduleMeasurement>& MeasurementDatabase::records() const { return records_; }

void MeasurementDatabase::saveCsv(const std::filesystem::path& path) const {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write measurement database: " + path.string());
    output << "m,n,k,time_ms,l1_miss,llc_miss,dtlb,register_pressure,schedule\n";
    for (const auto& record : records_) {
        output << record.problem.m << ',' << record.problem.n << ',' << record.problem.k << ','
               << record.milliseconds << ',' << record.simulation.l1_miss_rate() << ','
               << record.simulation.llc_miss_rate() << ',' << record.simulation.dtlb_misses << ','
               << record.simulation.register_pressure << ",\""
               << ScheduleDSL::print(record.schedule) << "\"\n";
    }
}

MeasurementDatabase MeasurementDatabase::loadCsv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read measurement database: " + path.string());
    MeasurementDatabase database;
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        const auto quote = line.find('"');
        if (quote == std::string::npos) continue;
        std::string prefix = line.substr(0, quote);
        if (!prefix.empty() && prefix.back() == ',') prefix.pop_back();
        std::stringstream fields(prefix);
        std::vector<std::string> values;
        std::string field;
        while (std::getline(fields, field, ',')) values.push_back(field);
        if (values.size() < 8) continue;
        ScheduleMeasurement record;
        record.problem = {std::stoi(values[0]), std::stoi(values[1]), std::stoi(values[2]), true, true};
        record.milliseconds = std::stod(values[3]);
        record.simulation.register_pressure = std::stod(values[7]);
        const auto last_quote = line.rfind('"');
        record.schedule = ScheduleDSL::parse(line.substr(quote + 1, last_quote - quote - 1));
        database.add(std::move(record));
    }
    return database;
}

namespace {

std::vector<double> learned_features(const Problem& problem, const Schedule& schedule,
                                     const SimulationResult& simulation) {
    const double operations = 2.0 * problem.m * problem.n * problem.k;
    const double tile_work = static_cast<double>(schedule.bm) * schedule.bn * schedule.bk;
    return {1.0,
            std::log1p(operations),
            std::log1p(tile_work),
            static_cast<double>(schedule.mr),
            static_cast<double>(schedule.nr),
            static_cast<double>(schedule.threads),
            static_cast<double>(schedule.vector_width),
            simulation.l1_miss_rate(),
            simulation.llc_miss_rate(),
            std::log1p(static_cast<double>(simulation.dtlb_misses)),
            simulation.register_pressure};
}

}  // namespace

void LearnedCostModel::fit(const MeasurementDatabase& database) {
    constexpr std::size_t feature_count = 11;
    weights_.assign(feature_count, 0.0);
    if (database.records().empty()) return;
    constexpr double learning_rate = 0.003;
    constexpr double regularization = 1.0e-4;
    for (int epoch = 0; epoch < 2500; ++epoch) {
        std::vector<double> gradient(feature_count, 0.0);
        for (const auto& record : database.records()) {
            const auto features = learned_features(record.problem, record.schedule, record.simulation);
            double prediction = 0.0;
            for (std::size_t index = 0; index < feature_count; ++index)
                prediction += weights_[index] * features[index];
            const double target = std::log(std::max(record.milliseconds, 1.0e-9));
            const double error = prediction - target;
            for (std::size_t index = 0; index < feature_count; ++index)
                gradient[index] += error * features[index];
        }
        const double scale = 1.0 / static_cast<double>(database.records().size());
        for (std::size_t index = 0; index < feature_count; ++index) {
            weights_[index] -= learning_rate *
                (gradient[index] * scale + regularization * weights_[index]);
        }
    }
}

double LearnedCostModel::predictMilliseconds(const Problem& problem, const Schedule& schedule,
                                             const SimulationResult& simulation) const {
    if (weights_.empty()) return std::numeric_limits<double>::infinity();
    const auto features = learned_features(problem, schedule, simulation);
    double prediction = 0.0;
    for (std::size_t index = 0; index < weights_.size(); ++index)
        prediction += weights_[index] * features[index];
    return std::exp(std::clamp(prediction, -20.0, 20.0));
}

double LearnedCostModel::hybridScore(const Problem& problem, const Schedule& schedule,
                                     const SimulationResult& simulation,
                                     double analytical_weight) const {
    const double learned = predictMilliseconds(problem, schedule, simulation);
    const double analytical = simulation.estimated_cycles / 3.5e6;
    if (!std::isfinite(learned)) return analytical;
    return analytical_weight * analytical + (1.0 - analytical_weight) * learned;
}
bool LearnedCostModel::trained() const { return !weights_.empty(); }

GraphBenchmarkResult ExecutableRuntime::runMLP(const ExecutablePlan& plan,
                                               const MLPConfig& config,
                                               const MLPData& data, int warmup,
                                               int repetitions) const {
    return execute_mlp(plan, config, data, warmup, repetitions);
}

TensorGraph StableHLOImporter::importFile(const std::filesystem::path& path) const {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open StableHLO file: " + path.string());
    return importText(std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));
}

TensorGraph StableHLOImporter::importText(const std::string& text) const {
    TensorGraph graph;
    std::unordered_map<std::string, int> values;
    std::stringstream lines(text);
    std::string line;
    const std::regex argument_pattern(R"((%[A-Za-z0-9_]+)\s*:\s*(tensor<[^>]+>))");
    const std::regex result_pattern(R"((%[A-Za-z0-9_]+)\s*=\s*stablehlo\.([A-Za-z0-9_]+)\s+([^:]+)(?::\s*(tensor<[^>]+>))?)");
    while (std::getline(lines, line)) {
        if (line.find("func.func") != std::string::npos) {
            for (auto iterator = std::sregex_iterator(line.begin(), line.end(), argument_pattern);
                 iterator != std::sregex_iterator(); ++iterator) {
                const std::string name = (*iterator)[1];
                if (!values.contains(name)) values[name] = graph.addInput(name.substr(1), parse_tensor_type((*iterator)[2]));
            }
            continue;
        }
        std::smatch match;
        if (std::regex_search(line, match, result_pattern)) {
            const std::string result = match[1];
            const std::string operation_name = match[2];
            const std::string operands_text = match[3];
            std::vector<int> operands;
            const std::regex operand_pattern(R"(%[A-Za-z0-9_]+)");
            for (auto iterator = std::sregex_iterator(operands_text.begin(), operands_text.end(), operand_pattern);
                 iterator != std::sregex_iterator(); ++iterator) {
                const auto found = values.find(iterator->str());
                if (found != values.end()) operands.push_back(found->second);
            }
            std::optional<GraphOpKind> kind;
            if (operation_name == "dot_general" || operation_name == "dot") kind = GraphOpKind::MatMul;
            else if (operation_name == "add") kind = GraphOpKind::Add;
            else if (operation_name == "multiply") kind = GraphOpKind::Multiply;
            else if (operation_name == "maximum") kind = GraphOpKind::Maximum;
            else if (operation_name == "broadcast_in_dim") kind = GraphOpKind::Broadcast;
            else if (operation_name == "reshape") kind = GraphOpKind::Reshape;
            else if (operation_name == "transpose") kind = GraphOpKind::Transpose;
            else if (operation_name == "reduce") kind = GraphOpKind::Reduce;
            else if (operation_name == "exponential") kind = GraphOpKind::Exp;
            else if (operation_name == "rsqrt") kind = GraphOpKind::Rsqrt;
            else if (operation_name == "convert") kind = GraphOpKind::Convert;
            else if (operation_name == "custom_call" && operands_text.find("Gelu") != std::string::npos)
                kind = GraphOpKind::Gelu;
            if (!kind)
                throw std::invalid_argument("unsupported StableHLO operation: " + operation_name);
            GraphTensorType type;
            if (match[4].matched) type = parse_tensor_type(match[4]);
            values[result] = graph.addOperation(*kind, result.substr(1), operands, std::move(type));
        } else if (line.find("return") != std::string::npos) {
            const std::regex return_pattern(R"(%[A-Za-z0-9_]+)");
            std::smatch returned;
            if (std::regex_search(line, returned, return_pattern) && values.contains(returned.str()))
                graph.setReturn(values[returned.str()]);
        }
    }
    if (graph.returnValue() < 0 && !graph.values().empty()) graph.setReturn(static_cast<int>(graph.values().size() - 1));
    ShapeInferencePass{}.run(graph);
    return GraphCanonicalizationPass{}.run(graph);
}

GraphCompiler::GraphCompiler(TargetInfo target) : target_(std::move(target)) {}

ExecutablePlan GraphCompiler::compile(TensorGraph graph, const GraphCompileOptions& options,
                                      const MLPConfig* mlp_config,
                                      const MLPData* mlp_data) const {
    graph = GraphCanonicalizationPass{}.run(graph);
    if (mlp_config) {
        const std::int64_t rows = static_cast<std::int64_t>(mlp_config->batch) * mlp_config->sequence;
        for (auto& value : graph.values()) {
            for (auto& dimension : value.type.shape) {
                if (dimension.kind != DimensionKind::Static &&
                    (dimension.symbol == "B*S" || dimension.symbol == "?")) {
                    dimension = Dimension::fixed(rows);
                }
            }
        }
    }
    const auto constraints = ShapeInferencePass{}.run(graph);
    ExecutablePlan plan;
    plan.target = options.target;
    plan.hardware = target_.str();
    plan.structured_computes = StructuredComputeLowering{}.run(graph);
    plan.dispatches = FusionPlanner{}.run(graph);
    plan.layout_conversions_removed = GraphLayoutPlanner{}.run(graph, plan.dispatches);
    plan.memory = GraphBufferizer{}.run(graph, plan.dispatches);
    for (const auto& constraint : constraints)
        if (constraint.runtime_guard) plan.guards.push_back({constraint.expression, "specialized_dispatch"});
    if (mlp_config && mlp_config->sequence > 0) {
        plan.guards.push_back({"S <= 64", "mlp_small"});
        plan.guards.push_back({"S <= 256", "mlp_medium"});
        plan.guards.push_back({"otherwise", "mlp_large"});
    }
    for (std::size_t index = 0; index < plan.dispatches.size(); ++index) {
        auto& dispatch = plan.dispatches[index];
        dispatch.kernel_problem = dispatch_problem(graph, dispatch);
        dispatch.schedule.threads = std::min(options.max_threads, target_.logical_cpus);
        dispatch.schedule.pin_threads = dispatch.schedule.threads > 1;
        if (options.autotune && mlp_config && mlp_data && dispatch.kernel_problem.m > 0) {
            TensorData kernel_data;
            if (index == 0) {
                kernel_data = {mlp_data->input, mlp_data->weight1, mlp_data->bias1, {}, {}};
            } else {
                std::vector<float> hidden(static_cast<std::size_t>(dispatch.kernel_problem.m * dispatch.kernel_problem.k));
                kernel_data = {std::move(hidden), mlp_data->weight2, mlp_data->bias2, {}, {}};
            }
            const auto tuning = autoschedule({dispatch.kernel_problem}, kernel_data, target_,
                options.max_threads, options.top_k, options.warmup, options.repetitions);
            dispatch.schedule = tuning.schedule;
            dispatch.tuning_search_space = tuning.search_space;
            dispatch.hardware_measurements = tuning.benchmarked;
            dispatch.tuning_cache_hit = tuning.cache_hit;
        }
        dispatch.transforms = TransformProgram::fromSchedule(dispatch.schedule);
        auto kernel_loop = apply_schedule(dispatch.kernel_problem, dispatch.schedule);
        verify_loop_ir(kernel_loop);
        (void)analyze_loop_ir(kernel_loop);
        auto scheduled_loop = kernel_loop;
        if (dispatch.epilogue.find("gelu") != std::string::npos) {
            LoopOperation operation;
            operation.kind = LoopOpKind::Gelu;
            operation.result = "acc";
            operation.source = "%acc";
            scheduled_loop.insertEpilogueBeforeStores(std::move(operation));
        }
        if (dispatch.epilogue.find("residual") != std::string::npos) {
            LoopOperation operation;
            operation.kind = LoopOpKind::AddResidual;
            operation.result = "acc";
            operation.source = "%residual";
            operation.destination = "%acc";
            scheduled_loop.insertEpilogueBeforeStores(std::move(operation));
        }
        verify_loop_ir(scheduled_loop);
        (void)analyze_loop_ir(scheduled_loop);
        plan.scheduled_loops.push_back(std::move(scheduled_loop));
        const auto compute = std::find_if(plan.structured_computes.begin(),
            plan.structured_computes.end(), [&](const StructuredCompute& candidate) {
                return candidate.name.find("linear") != std::string::npos ||
                       candidate.body.find("+=") != std::string::npos;
            });
        if (compute != plan.structured_computes.end())
            dispatch.intrinsic = TensorizationPass{}.match(*compute, dispatch.schedule, target_);
        if (mlp_data) {
            TensorData llvm_data;
            if (index == 0) {
                llvm_data = {mlp_data->input, mlp_data->weight1, mlp_data->bias1, {}, {}};
            } else {
                llvm_data.a.assign(static_cast<std::size_t>(dispatch.kernel_problem.m) *
                                   dispatch.kernel_problem.k, 0.0F);
                llvm_data.b = mlp_data->weight2;
                llvm_data.bias = mlp_data->bias2;
            }
            const auto jit = LLVMJITBackend{}.benchmark(kernel_loop, llvm_data, 0, 1);
            plan.llvm_compile_milliseconds += jit.compile_milliseconds;
            plan.llvm_ir.push_back(jit.llvm_ir);
        } else {
            Compiler compiler(target_, ".schedforge-graph-cache");
            plan.llvm_ir.push_back(compiler.compile(
                {dispatch.kernel_problem}, dispatch.schedule).llvm_ir);
        }
    }
    plan.graph = std::move(graph);
    return plan;
}

}  // namespace schedforge
