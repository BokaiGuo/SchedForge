#include "schedforge/compiler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/TargetParser/Host.h>

namespace schedforge {
namespace {

using MatMulFunction = void (*)(const float*, const float*, const float*, const float*, float*,
                                std::int64_t, std::int64_t, std::int64_t);

std::string checksum(const std::uint8_t* data, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ULL;
    }
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

std::string checksum(const std::string& content) {
    return checksum(reinterpret_cast<const std::uint8_t*>(content.data()), content.size());
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read AOT artifact: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_text(const std::filesystem::path& path, const std::string& content) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write AOT artifact: " + path.string());
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!output) throw std::runtime_error("failed writing AOT artifact: " + path.string());
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& content) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write AOT object: " + path.string());
    output.write(reinterpret_cast<const char*>(content.data()),
                 static_cast<std::streamsize>(content.size()));
    if (!output) throw std::runtime_error("failed writing AOT object: " + path.string());
}

std::filesystem::path find_linker() {
    const std::vector<std::string> names{"clang++-18", "clang++", "c++"};
    const char* path_value = std::getenv("PATH");
    const std::string path = path_value ? path_value : "";
    for (const auto& name : names) {
        std::stringstream paths(path);
        std::string directory;
        while (std::getline(paths, directory, ':')) {
            const auto candidate = std::filesystem::path(directory) / name;
            if (::access(candidate.c_str(), X_OK) == 0) return candidate;
        }
    }
    throw std::runtime_error("AOT linker driver not found (tried clang++-18, clang++, c++)");
}

double link_shared_object(const std::filesystem::path& object,
                          const std::filesystem::path& shared) {
    const auto linker = find_linker();
    const auto start = std::chrono::steady_clock::now();
    const pid_t child = ::fork();
    if (child < 0) throw std::runtime_error("fork failed while linking AOT artifact");
    if (child == 0) {
        ::execl(linker.c_str(), linker.c_str(), "-shared", object.c_str(), "-o",
                shared.c_str(), "-lm", static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    if (::waitpid(child, &status, 0) < 0)
        throw std::runtime_error("waitpid failed while linking AOT artifact");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("AOT shared-library link failed with status " +
                                 std::to_string(status));
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

std::map<std::string, std::string> parse_fields(const std::string& text) {
    std::map<std::string, std::string> fields;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos)
            throw std::runtime_error("invalid AOT manifest line: " + line);
        const auto [iterator, inserted] = fields.emplace(
            line.substr(0, separator), line.substr(separator + 1));
        (void)iterator;
        if (!inserted) throw std::runtime_error("duplicate AOT manifest field: " +
                                                line.substr(0, separator));
    }
    return fields;
}

const std::string& required(const std::map<std::string, std::string>& fields,
                            const std::string& key) {
    const auto found = fields.find(key);
    if (found == fields.end()) throw std::runtime_error("missing AOT manifest field: " + key);
    return found->second;
}

void validate_host(const AOTManifest& manifest) {
    const std::string host_triple = llvm::sys::getDefaultTargetTriple();
    const std::string host_cpu = llvm::sys::getHostCPUName().str();
    if (manifest.target_triple != host_triple)
        throw std::runtime_error("AOT target triple mismatch: artifact=" + manifest.target_triple +
                                 " host=" + host_triple);
    if (manifest.target_cpu != host_cpu)
        throw std::runtime_error("AOT target CPU mismatch: artifact=" + manifest.target_cpu +
                                 " host=" + host_cpu);
}

void validate_package_files(const std::filesystem::path& path,
                            const AOTManifest& manifest) {
    if (!std::filesystem::is_directory(path))
        throw std::runtime_error("AOT package is not a directory: " + path.string());
    const auto loop = read_text(path / "loop.ir");
    const auto object = read_text(path / "kernel.o");
    const auto shared = read_text(path / "kernel.so");
    if (checksum(loop) != manifest.loop_checksum)
        throw std::runtime_error("AOT loop checksum mismatch");
    if (checksum(object) != manifest.object_checksum)
        throw std::runtime_error("AOT object checksum mismatch");
    if (checksum(shared) != manifest.shared_object_checksum)
        throw std::runtime_error("AOT shared-object checksum mismatch");
}

class DynamicLibrary {
public:
    explicit DynamicLibrary(const std::filesystem::path& path) {
        handle_ = ::dlopen(std::filesystem::absolute(path).c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) throw std::runtime_error("dlopen failed: " + std::string(::dlerror()));
    }
    ~DynamicLibrary() { if (handle_) ::dlclose(handle_); }
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    MatMulFunction lookup(const std::string& symbol) const {
        ::dlerror();
        void* address = ::dlsym(handle_, symbol.c_str());
        if (const char* error = ::dlerror())
            throw std::runtime_error("dlsym failed: " + std::string(error));
        return reinterpret_cast<MatMulFunction>(address);
    }
    const char* lookupData(const std::string& symbol) const {
        ::dlerror();
        void* address = ::dlsym(handle_, symbol.c_str());
        if (const char* error = ::dlerror())
            throw std::runtime_error("dlsym failed: " + std::string(error));
        return static_cast<const char*>(address);
    }
private:
    void* handle_ = nullptr;
};

void validate_embedded_metadata(const DynamicLibrary& library,
                                const AOTManifest& manifest) {
    const auto fields = parse_fields(library.lookupData("schedforge_aot_metadata_v1"));
    const auto match = [&](const std::string& key, const std::string& expected) {
        if (required(fields, key) != expected)
            throw std::runtime_error("AOT embedded metadata mismatch: " + key);
    };
    match("format_version", std::to_string(manifest.format_version));
    match("abi", manifest.abi);
    match("symbol", manifest.symbol);
    match("target_triple", manifest.target_triple);
    match("target_cpu", manifest.target_cpu);
    match("m", std::to_string(manifest.problem.m));
    match("n", std::to_string(manifest.problem.n));
    match("k", std::to_string(manifest.problem.k));
    match("bias", std::to_string(manifest.problem.bias));
    match("relu", std::to_string(manifest.problem.relu));
    match("threads", std::to_string(manifest.threads));
}

AOTManifest validate_package_without_loading(const std::filesystem::path& path) {
    const auto manifest = AOTManifest::parse(read_text(path / "manifest.sfe"));
    validate_host(manifest);
    validate_package_files(path, manifest);
    return manifest;
}

void validate_data(const Problem& problem, const TensorData& data) {
    const auto a_size = static_cast<std::size_t>(problem.m) * problem.k;
    const auto b_size = static_cast<std::size_t>(problem.k) * problem.n;
    const auto output_size = static_cast<std::size_t>(problem.m) * problem.n;
    if (data.a.size() != a_size || data.b.size() != b_size)
        throw std::invalid_argument("AOT input shape does not match manifest");
    if (problem.bias && data.bias.size() != static_cast<std::size_t>(problem.n))
        throw std::invalid_argument("AOT bias shape does not match manifest");
    if (!data.residual.empty() && data.residual.size() != output_size)
        throw std::invalid_argument("AOT residual shape does not match manifest");
}

}  // namespace

std::string AOTManifest::dump() const {
    std::ostringstream output;
    output << "# SchedForge Executable\n"
           << "format_version=" << format_version << '\n'
           << "schedforge_version=" << schedforge_version << '\n'
           << "abi=" << abi << '\n'
           << "symbol=" << symbol << '\n'
           << "target_triple=" << target_triple << '\n'
           << "target_cpu=" << target_cpu << '\n'
           << "m=" << problem.m << '\n'
           << "n=" << problem.n << '\n'
           << "k=" << problem.k << '\n'
           << "bias=" << problem.bias << '\n'
           << "relu=" << problem.relu << '\n'
           << "threads=" << threads << '\n'
           << "loop_checksum=" << loop_checksum << '\n'
           << "object_checksum=" << object_checksum << '\n'
           << "shared_object_checksum=" << shared_object_checksum << '\n';
    return output.str();
}

AOTManifest AOTManifest::parse(const std::string& text) {
    const auto fields = parse_fields(text);
    AOTManifest manifest;
    manifest.format_version = std::stoi(required(fields, "format_version"));
    manifest.schedforge_version = required(fields, "schedforge_version");
    manifest.abi = required(fields, "abi");
    manifest.symbol = required(fields, "symbol");
    manifest.target_triple = required(fields, "target_triple");
    manifest.target_cpu = required(fields, "target_cpu");
    manifest.problem.m = std::stoi(required(fields, "m"));
    manifest.problem.n = std::stoi(required(fields, "n"));
    manifest.problem.k = std::stoi(required(fields, "k"));
    manifest.problem.bias = std::stoi(required(fields, "bias")) != 0;
    manifest.problem.relu = std::stoi(required(fields, "relu")) != 0;
    manifest.threads = std::stoi(required(fields, "threads"));
    manifest.loop_checksum = required(fields, "loop_checksum");
    manifest.object_checksum = required(fields, "object_checksum");
    manifest.shared_object_checksum = required(fields, "shared_object_checksum");
    if (manifest.format_version != 1)
        throw std::runtime_error("unsupported AOT format version: " +
                                 std::to_string(manifest.format_version));
    if (manifest.abi != "schedforge_matmul_v1" || manifest.symbol != manifest.abi)
        throw std::runtime_error("unsupported AOT ABI: " + manifest.abi);
    if (manifest.threads != 1)
        throw std::runtime_error("unsupported AOT thread dispatch in format v1");
    if (manifest.problem.m <= 0 || manifest.problem.n <= 0 || manifest.problem.k <= 0)
        throw std::runtime_error("invalid AOT problem shape");
    return manifest;
}

AOTPackageResult create_aot_package(const LoopIR& loop,
                                    const std::filesystem::path& path) {
    if (path.extension() != ".sfe")
        throw std::invalid_argument("AOT package path must have a .sfe suffix");
    const auto object = LLVMAOTBackend{}.compile(loop);
    const auto execution = analyze_loop_ir(loop);
    const auto parent = path.parent_path().empty() ? std::filesystem::current_path()
                                                   : path.parent_path();
    std::filesystem::create_directories(parent);
    const auto temporary = parent / (path.filename().string() + ".tmp-" +
                                     std::to_string(::getpid()));
    std::filesystem::remove_all(temporary);
    std::filesystem::create_directories(temporary);
    try {
        const auto object_path = temporary / "kernel.o";
        const auto shared_path = temporary / "kernel.so";
        const std::string loop_text = loop.dump();
        write_bytes(object_path, object.object_code);
        write_text(temporary / "loop.ir", loop_text);
        write_text(temporary / "kernel.ll", object.llvm_ir);
        write_text(temporary / "kernel.s", object.assembly);
        const double link_milliseconds = link_shared_object(object_path, shared_path);
        AOTManifest manifest;
        manifest.abi = object.symbol;
        manifest.symbol = object.symbol;
        manifest.target_triple = object.target_triple;
        manifest.target_cpu = object.target_cpu;
        manifest.problem = loop.problem;
        manifest.threads = execution.threads;
        manifest.loop_checksum = checksum(loop_text);
        manifest.object_checksum = checksum(read_text(object_path));
        manifest.shared_object_checksum = checksum(read_text(shared_path));
        write_text(temporary / "manifest.sfe", manifest.dump());
        std::filesystem::remove_all(path);
        std::filesystem::rename(temporary, path);
        return {manifest, object.compile_milliseconds, link_milliseconds, path};
    } catch (...) {
        std::filesystem::remove_all(temporary);
        throw;
    }
}

AOTManifest inspect_aot_package(const std::filesystem::path& path) {
    const auto manifest = validate_package_without_loading(path);
    DynamicLibrary library(path / "kernel.so");
    validate_embedded_metadata(library, manifest);
    return manifest;
}

AOTBenchmarkResult benchmark_aot_package(const std::filesystem::path& path,
                                         const TensorData& data,
                                         int warmup, int repetitions) {
    const auto manifest = validate_package_without_loading(path);
    validate_data(manifest.problem, data);
    const auto load_start = std::chrono::steady_clock::now();
    DynamicLibrary library(path / "kernel.so");
    validate_embedded_metadata(library, manifest);
    const auto function = library.lookup(manifest.symbol);
    const double load_milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    std::vector<float> output(static_cast<std::size_t>(manifest.problem.m) *
                              manifest.problem.n);
    const auto invoke = [&] {
        function(data.a.data(), data.b.data(),
                 manifest.problem.bias ? data.bias.data() : nullptr,
                 data.residual.empty() ? nullptr : data.residual.data(), output.data(),
                 manifest.problem.m, manifest.problem.n, manifest.problem.k);
    };
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) invoke();
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        invoke();
        timings.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double execution_milliseconds = timings[timings.size() / 2];
    const auto expected = reference(manifest.problem, data);
    const double operations = 2.0 * manifest.problem.m * manifest.problem.n * manifest.problem.k;
    return {load_milliseconds, execution_milliseconds,
            operations / (execution_milliseconds * 1.0e6),
            max_abs_error(expected, output)};
}

}  // namespace schedforge
