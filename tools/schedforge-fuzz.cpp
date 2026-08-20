#include "schedforge/next_milestones.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::uint64_t seed = 1;
    int iterations = 1000;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument.starts_with("--seed=")) seed = std::stoull(argument.substr(7));
        else if (argument.starts_with("--iterations=")) iterations = std::stoi(argument.substr(13));
    }
    try {
        const auto summary = schedforge::run_schedforge_fuzz(seed, iterations);
        std::cout << summary.dump() << '\n';
        return summary.failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
