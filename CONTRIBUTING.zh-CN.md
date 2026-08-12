# 参与 SchedForge 开发

[English](CONTRIBUTING.md)

感谢你参与改进 SchedForge。贡献应保持项目最核心的架构约束：Tensor
语义、Schedule 决策、Simulator 和可执行代码生成必须彼此分离，同时通过同一份
Loop IR 形成闭环。

## 开发环境

```bash
sudo apt-get install -y build-essential cmake ninja-build \
  llvm-18-dev llvm-18-runtime llvm-18-tools clang-18

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Pull Request 要求

1. 重大架构或行为变更请先创建 Issue。
2. 保持改动聚焦，不要把无关清理和功能修改混在一起。
3. 在对应的 IR、Simulator、Backend 或 E2E 边界补充测试。
4. 实验报告必须保留负结果和 claim boundary。
5. 提交前运行 Release 测试；内存敏感改动还应运行 ASan/UBSan。
6. 修改公开使用方式时，需要同步更新中英文 README。

## 编码约定

- 使用 C++20，并遵循现有命名和所有权模式。
- 优先使用 RAII 和明确所有权，避免隐式生命周期耦合。
- Tensor IR、Loop IR、Schedule policy、Simulator 和 Runtime 应保持模块分离。
- 不得把某个 Schedule 硬编码成“对所有机器都最优”。
- 不得把模拟器指标表述成真实硬件计数器。
- 不提交构建目录、本地 Cache 或机器专属生成物；经过整理的实验 artifact 除外。

## 验证命令

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/schedforge-run --backend=native --M=65 --N=67 --K=63
./build/schedforge-run --backend=llvm --M=64 --N=64 --K=64
```
