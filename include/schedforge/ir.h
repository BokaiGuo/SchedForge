#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace schedforge {

enum class TypeKind { Index, Float32, Tensor, MemRef, Vector };

class Type {
public:
    virtual ~Type() = default;
    virtual TypeKind kind() const = 0;
    virtual std::string str() const = 0;
};

class ScalarType final : public Type {
public:
    explicit ScalarType(TypeKind kind);
    TypeKind kind() const override;
    std::string str() const override;
    static std::shared_ptr<Type> index();
    static std::shared_ptr<Type> f32();
private:
    TypeKind kind_;
};

class ShapedType : public Type {
public:
    const std::vector<std::int64_t>& shape() const;
    const std::shared_ptr<Type>& elementType() const;
protected:
    ShapedType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type);
    std::string shapedStr(const std::string& prefix) const;
private:
    std::vector<std::int64_t> shape_;
    std::shared_ptr<Type> element_type_;
};

class TensorType final : public ShapedType {
public:
    TensorType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type);
    TypeKind kind() const override;
    std::string str() const override;
    static std::shared_ptr<Type> get(std::vector<std::int64_t> shape,
                                     std::shared_ptr<Type> element_type = ScalarType::f32());
};

class MemRefType final : public ShapedType {
public:
    MemRefType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type);
    TypeKind kind() const override;
    std::string str() const override;
    static std::shared_ptr<Type> get(std::vector<std::int64_t> shape,
                                     std::shared_ptr<Type> element_type = ScalarType::f32());
};

class VectorType final : public ShapedType {
public:
    VectorType(std::vector<std::int64_t> shape, std::shared_ptr<Type> element_type);
    TypeKind kind() const override;
    std::string str() const override;
    static std::shared_ptr<Type> get(std::vector<std::int64_t> shape,
                                     std::shared_ptr<Type> element_type = ScalarType::f32());
};

class Operation;

class Value {
public:
    Value(std::shared_ptr<Type> type, Operation* defining_op, std::string name);
    const std::shared_ptr<Type>& type() const;
    Operation* definingOp() const;
    const std::string& name() const;
    const std::vector<Operation*>& users() const;
    void addUser(Operation* operation);
    void replaceAllUsesWith(Value* replacement);
private:
    std::shared_ptr<Type> type_;
    Operation* defining_op_;
    std::string name_;
    std::vector<Operation*> users_;
};

class Block {
public:
    Value* addArgument(std::shared_ptr<Type> type, const std::string& name);
    Operation* append(std::unique_ptr<Operation> operation);
    std::vector<std::unique_ptr<Operation>>& operations();
    const std::vector<std::unique_ptr<Operation>>& operations() const;
    const std::vector<std::unique_ptr<Value>>& arguments() const;
    std::string dump(unsigned indent = 0) const;
private:
    std::vector<std::unique_ptr<Value>> arguments_;
    std::vector<std::unique_ptr<Operation>> operations_;
};

class Operation {
public:
    Operation(std::string name, std::vector<Value*> operands,
              std::vector<std::shared_ptr<Type>> result_types);
    virtual ~Operation() = default;
    const std::string& name() const;
    const std::vector<Value*>& operands() const;
    Value* result(std::size_t index) const;
    std::size_t resultCount() const;
    void replaceOperand(Value* old_value, Value* new_value);
    void setAttribute(std::string key, std::string value);
    std::string attribute(const std::string& key) const;
    virtual std::string dump(unsigned indent = 0) const;
protected:
    std::vector<Value*> operands_;
    std::vector<std::unique_ptr<Value>> results_;
    std::unordered_map<std::string, std::string> attributes_;
private:
    std::string name_;
};

class ForOperation final : public Operation {
public:
    ForOperation(const std::string& induction_name, std::int64_t lower,
                 std::int64_t upper, std::int64_t step);
    Value* inductionVariable() const;
    Block& body();
    const Block& body() const;
    std::int64_t lower() const;
    std::int64_t upper() const;
    std::int64_t step() const;
    std::string dump(unsigned indent = 0) const override;
private:
    std::int64_t lower_;
    std::int64_t upper_;
    std::int64_t step_;
    Block body_;
};

class Module {
public:
    explicit Module(std::string name);
    Block& body();
    const Block& body() const;
    const std::string& name() const;
    std::string dump() const;
private:
    std::string name_;
    Block body_;
};

class IRBuilder {
public:
    explicit IRBuilder(Block& block);
    Value* createInput(const std::string& name, std::shared_ptr<Type> type);
    Operation* createMatMul(Value* lhs, Value* rhs, std::shared_ptr<Type> result_type);
    ForOperation* createFor(const std::string& induction_name, std::int64_t lower,
                            std::int64_t upper, std::int64_t step);
    Operation* createLoad(Value* buffer, std::vector<Value*> indices);
    Operation* createStore(Value* value, Value* buffer, std::vector<Value*> indices);
    Operation* createBinary(const std::string& name, Value* lhs, Value* rhs);
    Operation* createFMA(Value* lhs, Value* rhs, Value* accumulator);
    Operation* createVectorLoad(Value* buffer, std::vector<Value*> indices,
                                int width);
    Operation* createVectorStore(Value* value, Value* buffer,
                                 std::vector<Value*> indices);
    Operation* createBroadcast(Value* value, int width);
    Operation* createVectorFMA(Value* lhs, Value* rhs, Value* accumulator);
    Operation* createPack(Value* source, std::shared_ptr<Type> packed_type,
                          const std::string& layout);
    Operation* createPrefetch(Value* buffer, std::vector<Value*> indices,
                              int distance);
    Operation* createReturn(Value* value);
private:
    Block& block_;
};

class OperationPass {
public:
    virtual ~OperationPass() = default;
    virtual void run(Module& module) = 0;
};

class OperationPassManager {
public:
    template <typename Pass, typename... Args>
    Pass& addPass(Args&&... args) {
        auto pass = std::make_unique<Pass>(std::forward<Args>(args)...);
        Pass& reference = *pass;
        passes_.push_back(std::move(pass));
        return reference;
    }
    void run(Module& module);
private:
    std::vector<std::unique_ptr<OperationPass>> passes_;
};

}  // namespace schedforge
