#include "schedforge/ir.h"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <stdexcept>

namespace schedforge {
namespace {

std::string indentation(unsigned indent) { return std::string(indent, ' '); }
std::atomic<std::uint64_t> next_value_id{0};

}  // namespace

ScalarType::ScalarType(TypeKind kind) : kind_(kind) {}
TypeKind ScalarType::kind() const { return kind_; }
std::string ScalarType::str() const { return kind_ == TypeKind::Index ? "index" : "f32"; }
std::shared_ptr<Type> ScalarType::index() { return std::make_shared<ScalarType>(TypeKind::Index); }
std::shared_ptr<Type> ScalarType::f32() { return std::make_shared<ScalarType>(TypeKind::Float32); }

ShapedType::ShapedType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type)
    : shape_(std::move(shape)), element_type_(std::move(element_type)) {}
const std::vector<std::int64_t>& ShapedType::shape() const { return shape_; }
const std::shared_ptr<Type>& ShapedType::elementType() const { return element_type_; }
std::string ShapedType::shapedStr(const std::string& prefix) const {
    std::ostringstream out;
    out << prefix << '<';
    for (const auto extent : shape_) out << (extent < 0 ? "?" : std::to_string(extent)) << 'x';
    out << element_type_->str() << '>';
    return out.str();
}

TensorType::TensorType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type)
    : ShapedType(std::move(shape), std::move(element_type)) {}
TypeKind TensorType::kind() const { return TypeKind::Tensor; }
std::string TensorType::str() const { return shapedStr("tensor"); }
std::shared_ptr<Type> TensorType::get(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type) {
    return std::make_shared<TensorType>(std::move(shape), std::move(element_type));
}

MemRefType::MemRefType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type)
    : ShapedType(std::move(shape), std::move(element_type)) {}
TypeKind MemRefType::kind() const { return TypeKind::MemRef; }
std::string MemRefType::str() const { return shapedStr("memref"); }
std::shared_ptr<Type> MemRefType::get(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type) {
    return std::make_shared<MemRefType>(std::move(shape), std::move(element_type));
}

VectorType::VectorType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type)
    : ShapedType(std::move(shape), std::move(element_type)) {}
TypeKind VectorType::kind() const { return TypeKind::Vector; }
std::string VectorType::str() const { return shapedStr("vector"); }
std::shared_ptr<Type> VectorType::get(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type) {
    return std::make_shared<VectorType>(std::move(shape), std::move(element_type));
}

Value::Value(std::shared_ptr<Type> type, Operation* defining_op, std::string name)
    : type_(std::move(type)), defining_op_(defining_op), name_(std::move(name)) {}
const std::shared_ptr<Type>& Value::type() const { return type_; }
Operation* Value::definingOp() const { return defining_op_; }
const std::string& Value::name() const { return name_; }
const std::vector<Operation*>& Value::users() const { return users_; }
void Value::addUser(Operation* operation) { users_.push_back(operation); }
void Value::replaceAllUsesWith(Value* replacement) {
    auto users = users_;
    for (Operation* user : users) user->replaceOperand(this, replacement);
    users_.clear();
}

Value* Block::addArgument(std::shared_ptr<Type> type, const std::string& name) {
    arguments_.push_back(std::make_unique<Value>(std::move(type), nullptr, name));
    return arguments_.back().get();
}
Operation* Block::append(std::unique_ptr<Operation> operation) {
    operations_.push_back(std::move(operation));
    return operations_.back().get();
}
std::vector<std::unique_ptr<Operation>>& Block::operations() { return operations_; }
const std::vector<std::unique_ptr<Operation>>& Block::operations() const { return operations_; }
const std::vector<std::unique_ptr<Value>>& Block::arguments() const { return arguments_; }
std::string Block::dump(unsigned indent) const {
    std::ostringstream out;
    for (const auto& operation : operations_) out << operation->dump(indent);
    return out.str();
}

Operation::Operation(std::string name, std::vector<Value*> operands,
                     std::vector<std::shared_ptr<Type>> result_types)
    : operands_(std::move(operands)), name_(std::move(name)) {
    for (Value* operand : operands_) operand->addUser(this);
    for (std::size_t index = 0; index < result_types.size(); ++index) {
        results_.push_back(std::make_unique<Value>(result_types[index], this,
                                                   "%v" + std::to_string(next_value_id.fetch_add(1))));
    }
}
const std::string& Operation::name() const { return name_; }
const std::vector<Value*>& Operation::operands() const { return operands_; }
Value* Operation::result(std::size_t index) const { return results_.at(index).get(); }
std::size_t Operation::resultCount() const { return results_.size(); }
void Operation::replaceOperand(Value* old_value, Value* new_value) {
    for (Value*& operand : operands_) if (operand == old_value) { operand = new_value; new_value->addUser(this); }
}
void Operation::setAttribute(std::string key, std::string value) { attributes_[std::move(key)] = std::move(value); }
std::string Operation::attribute(const std::string& key) const {
    const auto found = attributes_.find(key);
    return found == attributes_.end() ? "" : found->second;
}
std::string Operation::dump(unsigned indent) const {
    std::ostringstream out;
    out << indentation(indent);
    if (!results_.empty()) {
        for (std::size_t index = 0; index < results_.size(); ++index) {
            if (index) out << ", ";
            out << results_[index]->name();
        }
        out << " = ";
    }
    out << name_;
    if (!operands_.empty()) {
        out << ' ';
        for (std::size_t index = 0; index < operands_.size(); ++index) {
            if (index) out << ", ";
            out << operands_[index]->name();
        }
    }
    if (!attributes_.empty()) {
        out << " {";
        bool first = true;
        for (const auto& [key, value] : attributes_) {
            if (!first) out << ", ";
            out << key << '=' << value;
            first = false;
        }
        out << '}';
    }
    if (!results_.empty()) out << " : " << results_.front()->type()->str();
    out << '\n';
    return out.str();
}

ForOperation::ForOperation(const std::string& induction_name, std::int64_t lower,
                           std::int64_t upper, std::int64_t step)
    : Operation("loop.for", {}, {ScalarType::index()}), lower_(lower), upper_(upper), step_(step) {
    setAttribute("iv", induction_name);
}
Value* ForOperation::inductionVariable() const { return result(0); }
Block& ForOperation::body() { return body_; }
const Block& ForOperation::body() const { return body_; }
std::int64_t ForOperation::lower() const { return lower_; }
std::int64_t ForOperation::upper() const { return upper_; }
std::int64_t ForOperation::step() const { return step_; }
std::string ForOperation::dump(unsigned indent) const {
    std::ostringstream out;
    out << indentation(indent) << inductionVariable()->name() << " = loop.for " << lower_ << " to "
        << upper_ << " step " << step_ << " {\n" << body_.dump(indent + 2)
        << indentation(indent) << "}\n";
    return out.str();
}

Module::Module(std::string name) : name_(std::move(name)) {}
Block& Module::body() { return body_; }
const Block& Module::body() const { return body_; }
const std::string& Module::name() const { return name_; }
std::string Module::dump() const {
    return "module @" + name_ + " {\n" + body_.dump(2) + "}\n";
}

IRBuilder::IRBuilder(Block& block) : block_(block) {}
Value* IRBuilder::createInput(const std::string& name, std::shared_ptr<Type> type) {
    auto operation = std::make_unique<Operation>("tensor.input", std::vector<Value*>{},
                                                 std::vector<std::shared_ptr<Type>>{std::move(type)});
    operation->setAttribute("name", name);
    auto* raw = block_.append(std::move(operation));
    return raw->result(0);
}
Operation* IRBuilder::createMatMul(Value* lhs, Value* rhs, std::shared_ptr<Type> result_type) {
    return block_.append(std::make_unique<Operation>("tensor.matmul", std::vector<Value*>{lhs, rhs},
                                                     std::vector<std::shared_ptr<Type>>{std::move(result_type)}));
}
ForOperation* IRBuilder::createFor(const std::string& induction_name, std::int64_t lower,
                                   std::int64_t upper, std::int64_t step) {
    auto operation = std::make_unique<ForOperation>(induction_name, lower, upper, step);
    auto* raw = operation.get();
    block_.append(std::move(operation));
    return raw;
}
Operation* IRBuilder::createLoad(Value* buffer, std::vector<Value*> indices) {
    std::vector<Value*> operands{buffer};
    operands.insert(operands.end(), indices.begin(), indices.end());
    return block_.append(std::make_unique<Operation>("memref.load", std::move(operands),
                                                     std::vector<std::shared_ptr<Type>>{ScalarType::f32()}));
}
Operation* IRBuilder::createStore(Value* value, Value* buffer, std::vector<Value*> indices) {
    std::vector<Value*> operands{value, buffer};
    operands.insert(operands.end(), indices.begin(), indices.end());
    return block_.append(std::make_unique<Operation>("memref.store", std::move(operands),
                                                     std::vector<std::shared_ptr<Type>>{}));
}
Operation* IRBuilder::createBinary(const std::string& name, Value* lhs, Value* rhs) {
    return block_.append(std::make_unique<Operation>(name, std::vector<Value*>{lhs, rhs},
                                                     std::vector<std::shared_ptr<Type>>{lhs->type()}));
}
Operation* IRBuilder::createFMA(Value* lhs, Value* rhs, Value* accumulator) {
    return block_.append(std::make_unique<Operation>("arith.fma", std::vector<Value*>{lhs, rhs, accumulator},
                                                     std::vector<std::shared_ptr<Type>>{accumulator->type()}));
}
Operation* IRBuilder::createVectorLoad(Value* buffer, std::vector<Value*> indices,
                                       int width) {
    std::vector<Value*> operands{buffer};
    operands.insert(operands.end(), indices.begin(), indices.end());
    return block_.append(std::make_unique<Operation>("vector.load", std::move(operands),
        std::vector<std::shared_ptr<Type>>{VectorType::get({width})}));
}
Operation* IRBuilder::createVectorStore(Value* value, Value* buffer,
                                        std::vector<Value*> indices) {
    std::vector<Value*> operands{value, buffer};
    operands.insert(operands.end(), indices.begin(), indices.end());
    return block_.append(std::make_unique<Operation>("vector.store", std::move(operands),
        std::vector<std::shared_ptr<Type>>{}));
}
Operation* IRBuilder::createBroadcast(Value* value, int width) {
    return block_.append(std::make_unique<Operation>("vector.broadcast", std::vector<Value*>{value},
        std::vector<std::shared_ptr<Type>>{VectorType::get({width})}));
}
Operation* IRBuilder::createVectorFMA(Value* lhs, Value* rhs, Value* accumulator) {
    return block_.append(std::make_unique<Operation>("vector.fma", std::vector<Value*>{lhs, rhs, accumulator},
        std::vector<std::shared_ptr<Type>>{accumulator->type()}));
}
Operation* IRBuilder::createPack(Value* source, std::shared_ptr<Type> packed_type,
                                 const std::string& layout) {
    auto operation = std::make_unique<Operation>("tensor.pack", std::vector<Value*>{source},
        std::vector<std::shared_ptr<Type>>{std::move(packed_type)});
    operation->setAttribute("layout", layout);
    return block_.append(std::move(operation));
}
Operation* IRBuilder::createPrefetch(Value* buffer, std::vector<Value*> indices,
                                     int distance) {
    std::vector<Value*> operands{buffer};
    operands.insert(operands.end(), indices.begin(), indices.end());
    auto operation = std::make_unique<Operation>("memref.prefetch", std::move(operands),
        std::vector<std::shared_ptr<Type>>{});
    operation->setAttribute("distance", std::to_string(distance));
    return block_.append(std::move(operation));
}
Operation* IRBuilder::createReturn(Value* value) {
    return block_.append(std::make_unique<Operation>("func.return", std::vector<Value*>{value},
                                                     std::vector<std::shared_ptr<Type>>{}));
}

void OperationPassManager::run(Module& module) {
    for (const auto& pass : passes_) pass->run(module);
}

}  // namespace schedforge
