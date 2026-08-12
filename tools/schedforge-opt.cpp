#include "schedforge/compiler.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        schedforge::Problem problem;
        std::string schedule_text = "order=ikj;outer=64,128,32;tile=32,64,32;micro=4,8;vector=8;unroll=4;threads=1;pack=b;prefetch=4;fuse=true";
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg.starts_with("--M=")) problem.m = std::stoi(arg.substr(4));
            else if (arg.starts_with("--N=")) problem.n = std::stoi(arg.substr(4));
            else if (arg.starts_with("--K=")) problem.k = std::stoi(arg.substr(4));
            else if (arg.starts_with("--schedule=")) schedule_text = arg.substr(11);
        }
        schedforge::Compiler compiler;
        const auto result = compiler.compile({problem}, schedforge::ScheduleDSL::parse(schedule_text));
        std::cout << "// Tensor IR\n" << result.tensor_module.dump()
                  << "\n// Loop IR\n" << result.loop.dump()
                  << "\n// LLVM IR\n" << result.llvm_ir;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
