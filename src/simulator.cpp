#include "schedforge/schedforge.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace schedforge {
namespace {

class Cache {
public:
    Cache(std::size_t capacity, std::size_t line_size, std::size_t associativity)
        : line_size_(line_size), ways_(associativity),
          sets_(associativity && line_size ? capacity / line_size / associativity : 0),
          lines_(sets_ * ways_) {
        if (line_size_ == 0 || ways_ == 0 || sets_ == 0) {
            throw std::invalid_argument("cache capacity, line size, and associativity must define at least one set");
        }
    }

    bool access(std::uint64_t address) {
        ++clock_;
        const std::uint64_t line_address = address / line_size_;
        const std::size_t set = static_cast<std::size_t>(line_address % sets_);
        const std::uint64_t tag = line_address / sets_;
        Line* victim = &lines_[set * ways_];
        for (std::size_t way = 0; way < ways_; ++way) {
            Line& line = lines_[set * ways_ + way];
            if (line.valid && line.tag == tag) {
                line.last_used = clock_;
                return true;
            }
            if (!line.valid || line.last_used < victim->last_used) victim = &line;
        }
        victim->valid = true;
        victim->tag = tag;
        victim->last_used = clock_;
        return false;
    }

private:
    struct Line { std::uint64_t tag = 0; std::uint64_t last_used = 0; bool valid = false; };
    std::size_t line_size_;
    std::size_t ways_;
    std::size_t sets_;
    std::vector<Line> lines_;
    std::uint64_t clock_ = 0;
};

class Hierarchy {
public:
    explicit Hierarchy(const TargetInfo& target)
        : page_bytes_(target.page_bytes), l1_(target.l1_bytes, target.cache_line_bytes, 8),
          l2_(target.l2_bytes, target.cache_line_bytes, 8),
          l3_(target.l3_bytes, target.cache_line_bytes, 16) {}

    void access(std::uint64_t address, bool prefetch = false) {
        ++result.memory_accesses;
        const std::uint64_t page = address / page_bytes_;
        auto found_page = tlb_.find(page);
        if (found_page != tlb_.end()) {
            ++result.dtlb_hits;
            found_page->second = ++clock_;
        } else {
            ++result.dtlb_misses;
            if (tlb_.size() >= 64) {
                auto victim = std::min_element(tlb_.begin(), tlb_.end(),
                    [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
                tlb_.erase(victim);
            }
            tlb_[page] = ++clock_;
        }
        if (!prefetch && prefetched_.erase(address / 64) > 0) ++result.useful_prefetches;
        if (prefetch) { ++result.prefetched_lines; prefetched_[address / 64] = true; }
        if (l1_.access(address)) { ++result.l1.hits; return; }
        ++result.l1.misses;
        if (l2_.access(address)) { ++result.l2.hits; return; }
        ++result.l2.misses;
        if (l3_.access(address)) { ++result.l3.hits; return; }
        ++result.l3.misses;
        ++result.dram_accesses;
    }
    SimulationResult result;
private:
    std::size_t page_bytes_;
    Cache l1_;
    Cache l2_;
    Cache l3_;
    std::unordered_map<std::uint64_t, std::uint64_t> tlb_;
    std::unordered_map<std::uint64_t, bool> prefetched_;
    std::uint64_t clock_ = 0;
};

}  // namespace

double SimulationResult::l1_miss_rate() const {
    const auto total = l1.hits + l1.misses;
    return total ? static_cast<double>(l1.misses) / static_cast<double>(total) : 0.0;
}

double SimulationResult::llc_miss_rate() const {
    const auto total = l3.hits + l3.misses;
    return total ? static_cast<double>(l3.misses) / static_cast<double>(total) : 0.0;
}

SimulationResult simulate(const LoopIR& loop) {
    return simulate(loop, TargetInfo::detect());
}

SimulationResult simulate(const LoopIR& loop, const TargetInfo& target) {
    Hierarchy hierarchy(target);
    const int m = std::min(loop.problem.m, 64);
    const int n = std::min(loop.problem.n, 64);
    const int k_size = std::min(loop.problem.k, 64);
    constexpr std::uint64_t a_base = 0x100000000ULL;
    constexpr std::uint64_t b_base = 0x200000000ULL;
    constexpr std::uint64_t c_base = 0x300000000ULL;
    constexpr std::uint64_t d_base = 0x400000000ULL;
    constexpr std::uint64_t packed_a_base = 0x500000000ULL;
    constexpr std::uint64_t packed_b_base = 0x600000000ULL;
    auto a = [&](int i, int k) { return a_base + 4ULL * static_cast<std::uint64_t>(i * loop.problem.k + k); };
    auto b = [&](int k, int j) { return b_base + 4ULL * static_cast<std::uint64_t>(k * loop.problem.n + j); };
    auto c = [&](int i, int j) { return c_base + 4ULL * static_cast<std::uint64_t>(i * loop.problem.n + j); };

    if (loop.schedule.order == LoopOrder::IJK) {
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
            for (int k = 0; k < k_size; ++k) { hierarchy.access(a(i, k)); hierarchy.access(b(k, j)); }
            hierarchy.access(c(i, j));
        }
    } else {
        const int bm = loop.schedule.tiled ? loop.schedule.bm : m;
        const int bn = loop.schedule.tiled ? loop.schedule.bn : n;
        const int bk = loop.schedule.tiled ? loop.schedule.bk : k_size;
        for (int ii = 0; ii < m; ii += bm) for (int jj = 0; jj < n; jj += bn) {
            for (int kk = 0; kk < k_size; kk += bk)
                for (int i = ii; i < std::min(ii + bm, m); ++i)
                    for (int k = kk; k < std::min(kk + bk, k_size); ++k) {
                        hierarchy.access(loop.schedule.pack_a
                            ? packed_a_base + 4ULL * static_cast<std::uint64_t>((i - ii) * bk + (k - kk))
                            : a(i, k));
                        for (int j = jj; j < std::min(jj + bn, n); ++j) {
                            hierarchy.access(loop.schedule.pack_b
                                ? packed_b_base + 4ULL * static_cast<std::uint64_t>((k - kk) * bn + (j - jj))
                                : b(k, j));
                            hierarchy.access(c(i, j)); hierarchy.access(c(i, j));
                            if (loop.schedule.prefetch_distance > 0 && j + loop.schedule.prefetch_distance < std::min(jj + bn, n)) {
                                hierarchy.access(b(k, j + loop.schedule.prefetch_distance), true);
                            }
                        }
                    }
        }
    }
    if (loop.schedule.pack_a) {
        for (int i = 0; i < m; ++i) for (int k = 0; k < k_size; ++k) {
            hierarchy.access(a(i, k));
            hierarchy.access(packed_a_base + 4ULL * static_cast<std::uint64_t>(i * k_size + k));
        }
    }
    if (loop.schedule.pack_b) {
        for (int k = 0; k < k_size; ++k) for (int j = 0; j < n; ++j) {
            hierarchy.access(b(k, j));
            hierarchy.access(packed_b_base + 4ULL * static_cast<std::uint64_t>(k * n + j));
        }
    }
    if (!loop.schedule.fused && (loop.problem.bias || loop.problem.relu)) {
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
            hierarchy.access(c(i, j)); hierarchy.access(d_base + 4ULL * static_cast<std::uint64_t>(i * loop.problem.n + j));
            if (loop.problem.relu) { hierarchy.access(d_base + 4ULL * static_cast<std::uint64_t>(i * loop.problem.n + j)); hierarchy.access(c(i, j)); }
        }
    }
    auto result = hierarchy.result;
    const auto pressure = estimate_register_pressure(loop.schedule, target);
    result.register_pressure = static_cast<double>(pressure.total);
    result.estimated_bandwidth_bytes = static_cast<double>(result.dram_accesses * target.cache_line_bytes);
    const double sampled_memory_cycles = static_cast<double>(result.l1.hits) * target.l1_latency +
                                         static_cast<double>(result.l2.hits) * target.l2_latency +
                                         static_cast<double>(result.l3.hits) * target.l3_latency +
                                         static_cast<double>(result.dram_accesses) * target.dram_latency;
    const double scale = (static_cast<double>(loop.problem.m) / m) *
                         (static_cast<double>(loop.problem.n) / n) *
                         (static_cast<double>(loop.problem.k) / k_size);
    const double vector_lanes = static_cast<double>(std::max(1, loop.schedule.vector_width));
    const double thread_parallelism = static_cast<double>(std::max(1, loop.schedule.threads));
    const double fma_throughput = 2.0 * vector_lanes;
    const double compute_cycles = static_cast<double>(loop.problem.m) * loop.problem.n *
                                  loop.problem.k / fma_throughput;
    const double tlb_cycles = static_cast<double>(result.dtlb_misses) * 30.0 * scale;
    const double packing_bytes = (loop.schedule.pack_b
        ? 4.0 * static_cast<double>(loop.problem.k) * loop.problem.n : 0.0) +
        (loop.schedule.pack_a ? 4.0 * static_cast<double>(loop.problem.m) * loop.problem.k : 0.0);
    const double packing_cycles = packing_bytes / 32.0;
    const double spill_penalty = pressure.spills ? compute_cycles * 0.75 : 0.0;
    const double bandwidth_cycles = target.memory_bandwidth_gbps > 0.0
        ? result.estimated_bandwidth_bytes / target.memory_bandwidth_gbps
        : 0.0;
    result.estimated_cycles = sampled_memory_cycles * scale / thread_parallelism +
                              compute_cycles / thread_parallelism + tlb_cycles +
                              packing_cycles + spill_penalty + bandwidth_cycles;
    return result;
}

}  // namespace schedforge
