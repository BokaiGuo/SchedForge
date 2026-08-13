#include "schedforge/schedforge.h"
#include "schedforge/compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace schedforge {
namespace {

struct PackedMatrices {
    std::vector<float> a;
    std::vector<float> b;
    int mr = 1;
    int nr = 1;
};

class ThreadTeam {
public:
    explicit ThreadTeam(int thread_count) : thread_count_(thread_count) {
        workers_.reserve(static_cast<std::size_t>(thread_count_));
        for (int worker_index = 0; worker_index < thread_count_; ++worker_index) {
            workers_.emplace_back([this, worker_index] { worker_loop(worker_index); });
        }
    }

    ~ThreadTeam() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    void run(const std::function<void(int)>& job) {
        std::lock_guard dispatch_lock(dispatch_mutex_);
        {
            std::lock_guard lock(mutex_);
            job_ = job;
            completed_ = 0;
            ++generation_;
        }
        start_.notify_all();
        std::unique_lock lock(mutex_);
        finished_.wait(lock, [this] { return completed_ == thread_count_; });
        job_ = {};
    }

private:
    void worker_loop(int worker_index) {
        std::size_t observed_generation = 0;
        while (true) {
            std::function<void(int)> job;
            {
                std::unique_lock lock(mutex_);
                start_.wait(lock, [this, observed_generation] {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                job = job_;
            }
            job(worker_index);
            {
                std::lock_guard lock(mutex_);
                ++completed_;
                if (completed_ == thread_count_) finished_.notify_one();
            }
        }
    }

    int thread_count_;
    std::vector<std::thread> workers_;
    std::mutex dispatch_mutex_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable finished_;
    std::function<void(int)> job_;
    std::size_t generation_ = 0;
    int completed_ = 0;
    bool stopping_ = false;
};

ThreadTeam& thread_team(int thread_count) {
    static std::mutex teams_mutex;
    static std::map<int, std::unique_ptr<ThreadTeam>> teams;
    std::lock_guard lock(teams_mutex);
    auto& team = teams[thread_count];
    if (!team) team = std::make_unique<ThreadTeam>(thread_count);
    return *team;
}

#if defined(__linux__)
int read_topology_value(int cpu, const char* name) {
    std::ifstream input("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                        "/topology/" + name);
    int value = cpu;
    input >> value;
    return value;
}

const std::vector<int>& preferred_cpus() {
    static const std::vector<int> cpus = [] {
        cpu_set_t allowed;
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return std::vector<int>{};

        std::vector<int> primary;
        std::vector<int> siblings;
        std::set<std::pair<int, int>> seen_cores;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) continue;
            const int package = read_topology_value(cpu, "physical_package_id");
            const int core = read_topology_value(cpu, "core_id");
            if (seen_cores.insert({package, core}).second) primary.push_back(cpu);
            else siblings.push_back(cpu);
        }
        primary.insert(primary.end(), siblings.begin(), siblings.end());
        return primary;
    }();
    return cpus;
}

const cpu_set_t& allowed_cpu_set() {
    static const cpu_set_t allowed = [] {
        cpu_set_t result;
        CPU_ZERO(&result);
        const auto& cpus = preferred_cpus();
        if (cpus.empty()) {
            CPU_SET(0, &result);
        } else {
            for (const int cpu : cpus) CPU_SET(cpu, &result);
        }
        return result;
    }();
    return allowed;
}

int preferred_cpu(int worker_index) {
    const auto& cpus = preferred_cpus();
    return cpus.empty() ? worker_index : cpus[static_cast<std::size_t>(worker_index) % cpus.size()];
}
#endif

PackedMatrices pack_matrices(const Problem& problem, const Schedule& schedule,
                             const TensorData& data) {
    PackedMatrices packed;
    packed.mr = std::max(1, schedule.mr);
    packed.nr = std::max(1, schedule.nr);
    if (schedule.pack_a) {
        const int panels = (problem.m + packed.mr - 1) / packed.mr;
        packed.a.assign(static_cast<std::size_t>(panels) * problem.k * packed.mr, 0.0F);
        for (int i = 0; i < problem.m; ++i) for (int k = 0; k < problem.k; ++k) {
            const int panel = i / packed.mr;
            const int lane = i % packed.mr;
            packed.a[(static_cast<std::size_t>(panel) * problem.k + k) * packed.mr + lane] =
                data.a[static_cast<std::size_t>(i) * problem.k + k];
        }
    }
    if (schedule.pack_b) {
        const int panels = (problem.n + packed.nr - 1) / packed.nr;
        packed.b.assign(static_cast<std::size_t>(panels) * problem.k * packed.nr, 0.0F);
        for (int j = 0; j < problem.n; ++j) for (int k = 0; k < problem.k; ++k) {
            const int panel = j / packed.nr;
            const int lane = j % packed.nr;
            packed.b[(static_cast<std::size_t>(panel) * problem.k + k) * packed.nr + lane] =
                data.b[static_cast<std::size_t>(k) * problem.n + j];
        }
    }
    return packed;
}

void apply_epilogue_range(const Problem& problem, const TensorData& data,
                          std::vector<float>& output, int row_begin, int row_end,
                          int column_begin, int column_end) {
    for (int i = row_begin; i < row_end; ++i) {
        for (int j = column_begin; j < column_end; ++j) {
            float value = output[static_cast<std::size_t>(i) * problem.n + j];
            if (problem.bias) {
                value += data.bias[static_cast<std::size_t>(j)];
            }
            if (problem.relu) {
                value = std::max(0.0F, value);
            }
            output[static_cast<std::size_t>(i) * problem.n + j] = value;
        }
    }
}

void execute_ijk(const Problem& problem, const Schedule& schedule, const TensorData& data,
                 std::vector<float>& output, int row_begin, int row_end) {
    for (int i = row_begin; i < row_end; ++i) {
        for (int j = 0; j < problem.n; ++j) {
            float sum = 0.0F;
            for (int k = 0; k < problem.k; ++k) {
                sum += data.a[static_cast<std::size_t>(i) * problem.k + k] *
                       data.b[static_cast<std::size_t>(k) * problem.n + j];
            }
            if (schedule.fused) {
                if (problem.bias) sum += data.bias[static_cast<std::size_t>(j)];
                if (problem.relu) sum = std::max(0.0F, sum);
            }
            output[static_cast<std::size_t>(i) * problem.n + j] = sum;
        }
    }
    if (!schedule.fused) {
        apply_epilogue_range(problem, data, output, row_begin, row_end, 0, problem.n);
    }
}

void update_row(float* output, const float* b, float a, int count, int vector_width) {
    int j = 0;
#if defined(__AVX2__)
    if (vector_width >= 8) {
        const __m256 av = _mm256_set1_ps(a);
        for (; j + 8 <= count; j += 8) {
            const __m256 bv = _mm256_loadu_ps(b + j);
            __m256 cv = _mm256_loadu_ps(output + j);
#if defined(__FMA__)
            cv = _mm256_fmadd_ps(av, bv, cv);
#else
            cv = _mm256_add_ps(cv, _mm256_mul_ps(av, bv));
#endif
            _mm256_storeu_ps(output + j, cv);
        }
    }
#else
    (void)vector_width;
#endif
    for (; j < count; ++j) {
        output[j] += a * b[j];
    }
}

#if defined(__AVX2__)
template <int Rows>
void register_kernel_8(const Problem& problem, const Schedule& schedule,
                       const TensorData& data, std::vector<float>& output,
                       int row, int column) {
    __m256 accumulators[Rows];
    for (int lane = 0; lane < Rows; ++lane) {
        accumulators[lane] = _mm256_setzero_ps();
    }

    for (int k = 0; k < problem.k; ++k) {
        const __m256 b_values = _mm256_loadu_ps(
            data.b.data() + static_cast<std::size_t>(k) * problem.n + column);
        for (int lane = 0; lane < Rows; ++lane) {
            const __m256 a_value = _mm256_broadcast_ss(
                data.a.data() + static_cast<std::size_t>(row + lane) * problem.k + k);
#if defined(__FMA__)
            accumulators[lane] = _mm256_fmadd_ps(a_value, b_values, accumulators[lane]);
#else
            accumulators[lane] = _mm256_add_ps(
                accumulators[lane], _mm256_mul_ps(a_value, b_values));
#endif
        }
    }

    if (schedule.fused && problem.bias) {
        const __m256 bias = _mm256_loadu_ps(data.bias.data() + column);
        for (int lane = 0; lane < Rows; ++lane) {
            accumulators[lane] = _mm256_add_ps(accumulators[lane], bias);
        }
    }
    if (schedule.fused && problem.relu) {
        const __m256 zero = _mm256_setzero_ps();
        for (int lane = 0; lane < Rows; ++lane) {
            accumulators[lane] = _mm256_max_ps(accumulators[lane], zero);
        }
    }
    for (int lane = 0; lane < Rows; ++lane) {
        _mm256_storeu_ps(output.data() +
                             static_cast<std::size_t>(row + lane) * problem.n + column,
                         accumulators[lane]);
    }
}

template <int Rows>
void register_kernel_16(const Problem& problem, const Schedule& schedule,
                        const TensorData& data, std::vector<float>& output,
                        int row, int column) {
    __m256 low[Rows];
    __m256 high[Rows];
    for (int lane = 0; lane < Rows; ++lane) {
        low[lane] = _mm256_setzero_ps();
        high[lane] = _mm256_setzero_ps();
    }

    for (int k = 0; k < problem.k; ++k) {
        const float* b = data.b.data() + static_cast<std::size_t>(k) * problem.n + column;
        const __m256 b_low = _mm256_loadu_ps(b);
        const __m256 b_high = _mm256_loadu_ps(b + 8);
        for (int lane = 0; lane < Rows; ++lane) {
            const __m256 a_value = _mm256_broadcast_ss(
                data.a.data() + static_cast<std::size_t>(row + lane) * problem.k + k);
#if defined(__FMA__)
            low[lane] = _mm256_fmadd_ps(a_value, b_low, low[lane]);
            high[lane] = _mm256_fmadd_ps(a_value, b_high, high[lane]);
#else
            low[lane] = _mm256_add_ps(low[lane], _mm256_mul_ps(a_value, b_low));
            high[lane] = _mm256_add_ps(high[lane], _mm256_mul_ps(a_value, b_high));
#endif
        }
    }

    if (schedule.fused && problem.bias) {
        const __m256 bias_low = _mm256_loadu_ps(data.bias.data() + column);
        const __m256 bias_high = _mm256_loadu_ps(data.bias.data() + column + 8);
        for (int lane = 0; lane < Rows; ++lane) {
            low[lane] = _mm256_add_ps(low[lane], bias_low);
            high[lane] = _mm256_add_ps(high[lane], bias_high);
        }
    }
    if (schedule.fused && problem.relu) {
        const __m256 zero = _mm256_setzero_ps();
        for (int lane = 0; lane < Rows; ++lane) {
            low[lane] = _mm256_max_ps(low[lane], zero);
            high[lane] = _mm256_max_ps(high[lane], zero);
        }
    }
    for (int lane = 0; lane < Rows; ++lane) {
        float* destination = output.data() +
            static_cast<std::size_t>(row + lane) * problem.n + column;
        _mm256_storeu_ps(destination, low[lane]);
        _mm256_storeu_ps(destination + 8, high[lane]);
    }
}

void dispatch_register_kernel_8(const Problem& problem, const Schedule& schedule,
                                const TensorData& data, std::vector<float>& output,
                                int row, int rows, int column) {
    switch (rows) {
        case 8: register_kernel_8<8>(problem, schedule, data, output, row, column); break;
        case 7: register_kernel_8<7>(problem, schedule, data, output, row, column); break;
        case 6: register_kernel_8<6>(problem, schedule, data, output, row, column); break;
        case 5: register_kernel_8<5>(problem, schedule, data, output, row, column); break;
        case 4: register_kernel_8<4>(problem, schedule, data, output, row, column); break;
        case 3: register_kernel_8<3>(problem, schedule, data, output, row, column); break;
        case 2: register_kernel_8<2>(problem, schedule, data, output, row, column); break;
        default: register_kernel_8<1>(problem, schedule, data, output, row, column); break;
    }
}
#endif

void execute_register_blocked(const Problem& problem, const Schedule& schedule,
                              const TensorData& data, std::vector<float>& output,
                              int row_begin, int row_end) {
    const int bm = schedule.tiled ? std::max(1, schedule.bm) : row_end - row_begin;
    const int bn = schedule.tiled ? std::max(8, schedule.bn) : problem.n;
    const int mr = std::clamp(schedule.mr, 1, 8);

    for (int ii = row_begin; ii < row_end; ii += bm) {
        const int i_end = std::min(ii + bm, row_end);
        for (int jj = 0; jj < problem.n; jj += bn) {
            const int j_end = std::min(jj + bn, problem.n);
            for (int i0 = ii; i0 < i_end; i0 += mr) {
                const int rows = std::min(mr, i_end - i0);
                int j = jj;
#if defined(__AVX2__)
                if (rows <= 6) {
                    for (; j + 16 <= j_end; j += 16) {
                        switch (rows) {
                            case 6: register_kernel_16<6>(problem, schedule, data, output, i0, j); break;
                            case 5: register_kernel_16<5>(problem, schedule, data, output, i0, j); break;
                            case 4: register_kernel_16<4>(problem, schedule, data, output, i0, j); break;
                            case 3: register_kernel_16<3>(problem, schedule, data, output, i0, j); break;
                            case 2: register_kernel_16<2>(problem, schedule, data, output, i0, j); break;
                            default: register_kernel_16<1>(problem, schedule, data, output, i0, j); break;
                        }
                    }
                }
                for (; j + 8 <= j_end; j += 8) {
                    dispatch_register_kernel_8(problem, schedule, data, output,
                                               i0, rows, j);
                }
#endif
                for (; j < j_end; ++j) {
                    for (int lane = 0; lane < rows; ++lane) {
                        float value = 0.0F;
                        for (int k = 0; k < problem.k; ++k) {
                            value += data.a[static_cast<std::size_t>(i0 + lane) * problem.k + k] *
                                     data.b[static_cast<std::size_t>(k) * problem.n + j];
                        }
                        if (schedule.fused && problem.bias) value += data.bias[static_cast<std::size_t>(j)];
                        if (schedule.fused && problem.relu) value = std::max(0.0F, value);
                        output[static_cast<std::size_t>(i0 + lane) * problem.n + j] = value;
                    }
                }
            }
            if (!schedule.fused) {
                apply_epilogue_range(problem, data, output, ii, i_end, jj, j_end);
            }
        }
    }
}

void execute_ikj(const Problem& problem, const Schedule& schedule,
                 const TensorData& data, std::vector<float>& output,
                 int row_begin, int row_end) {
    const int bm = schedule.tiled ? schedule.bm : std::max(1, row_end - row_begin);
    const int bn = schedule.tiled ? schedule.bn : problem.n;
    const int bk = schedule.tiled ? schedule.bk : problem.k;
    const int mr = std::max(1, schedule.mr);

    for (int ii = row_begin; ii < row_end; ii += bm) {
        const int i_end = std::min(ii + bm, row_end);
        for (int jj = 0; jj < problem.n; jj += bn) {
            const int j_end = std::min(jj + bn, problem.n);
            for (int kk = 0; kk < problem.k; kk += bk) {
                const int k_end = std::min(kk + bk, problem.k);
                for (int i0 = ii; i0 < i_end; i0 += mr) {
                    const int micro_end = std::min(i0 + mr, i_end);
                    for (int k = kk; k < k_end; ++k) {
                        const float* b_ptr = data.b.data() + static_cast<std::size_t>(k) * problem.n + jj;
                        for (int i = i0; i < micro_end; ++i) {
                            float* c_ptr = output.data() + static_cast<std::size_t>(i) * problem.n + jj;
                            const float a_value = data.a[static_cast<std::size_t>(i) * problem.k + k];
                            update_row(c_ptr, b_ptr, a_value, j_end - jj, schedule.vector_width);
                        }
                    }
                }
            }
            if (schedule.fused) {
                apply_epilogue_range(problem, data, output, ii, i_end, jj, j_end);
            }
        }
    }
    if (!schedule.fused) {
        apply_epilogue_range(problem, data, output, row_begin, row_end, 0, problem.n);
    }
}

void execute_packed(const Problem& problem, const Schedule& schedule,
                    const TensorData& data, const PackedMatrices& packed,
                    std::vector<float>& output,
                    int row_begin, int row_end) {
    const int mc = std::max(schedule.bm, schedule.mc);
    const int nc = std::max(schedule.bn, schedule.nc);
    const int kc = std::max(schedule.bk, schedule.kc);
    const int bm = std::max(1, schedule.bm);
    const int bn = std::max(1, schedule.bn);
    const int bk = std::max(1, schedule.bk);
    const int mr = std::max(1, schedule.mr);
    const int nr = std::max(1, schedule.nr);
    for (int jc = 0; jc < problem.n; jc += nc) {
        const int n_size = std::min(nc, problem.n - jc);
        for (int pc = 0; pc < problem.k; pc += kc) {
            const int k_size = std::min(kc, problem.k - pc);
            for (int ic = row_begin; ic < row_end; ic += mc) {
                const int m_size = std::min(mc, row_end - ic);
                for (int jj = 0; jj < n_size; jj += bn) {
                    const int n_block = std::min(bn, n_size - jj);
                    for (int kk = 0; kk < k_size; kk += bk) {
                        const int k_block = std::min(bk, k_size - kk);
                        for (int ii = 0; ii < m_size; ii += bm) {
                            const int m_block = std::min(bm, m_size - ii);
                            for (int jr = 0; jr < n_block;) {
                                const int global_j = jc + jj + jr;
                                const int panel_remaining = schedule.pack_b
                                    ? packed.nr - global_j % packed.nr
                                    : nr;
                                const int n_micro = std::min(panel_remaining, n_block - jr);
                                for (int ir = 0; ir < m_block; ir += mr) {
                                    const int m_micro = std::min(mr, m_block - ir);
                                    const int unroll = std::max(1, schedule.unroll_k);
                                    for (int k0 = 0; k0 < k_block; k0 += unroll) {
                                        const int k_end = std::min(k0 + unroll, k_block);
                                        for (int k = k0; k < k_end; ++k) {
                                            const int global_k = pc + kk + k;
                                            const int b_panel = global_j / packed.nr;
                                            const int b_lane = global_j % packed.nr;
                                            const float* b_ptr = schedule.pack_b
                                                ? packed.b.data() + (static_cast<std::size_t>(b_panel) * problem.k + global_k) * packed.nr + b_lane
                                                : data.b.data() + static_cast<std::size_t>(global_k) * problem.n + global_j;
                                            if (schedule.prefetch_distance > 0 && k + schedule.prefetch_distance < k_block) {
                                                const int prefetch_k = global_k + schedule.prefetch_distance;
                                                const float* next_b = schedule.pack_b
                                                    ? packed.b.data() + (static_cast<std::size_t>(b_panel) * problem.k + prefetch_k) * packed.nr + b_lane
                                                    : data.b.data() + static_cast<std::size_t>(prefetch_k) * problem.n + global_j;
                                                __builtin_prefetch(next_b, 0, 2);
                                            }
                                            for (int i = 0; i < m_micro; ++i) {
                                                const int global_i = ic + ii + ir + i;
                                                const float a_value = schedule.pack_a
                                                    ? packed.a[(static_cast<std::size_t>(global_i / packed.mr) * problem.k + global_k) * packed.mr + global_i % packed.mr]
                                                    : data.a[static_cast<std::size_t>(global_i) * problem.k + global_k];
                                                float* c_ptr = output.data() +
                                                    static_cast<std::size_t>(global_i) * problem.n + global_j;
                                                update_row(c_ptr, b_ptr, a_value, n_micro, schedule.vector_width);
                                            }
                                        }
                                    }
                                }
                                jr += n_micro;
                            }
                        }
                    }
                }
            }
        }
        if (schedule.fused) {
            apply_epilogue_range(problem, data, output, row_begin, row_end, jc,
                                 std::min(jc + nc, problem.n));
        }
    }
    if (!schedule.fused) {
        apply_epilogue_range(problem, data, output, row_begin, row_end, 0, problem.n);
    }
}

}  // namespace

TensorData make_data(const Problem& problem, std::uint32_t seed) {
    if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) {
        throw std::invalid_argument("matrix dimensions must be positive");
    }
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    TensorData data;
    data.a.resize(static_cast<std::size_t>(problem.m) * problem.k);
    data.b.resize(static_cast<std::size_t>(problem.k) * problem.n);
    data.bias.resize(static_cast<std::size_t>(problem.n));
    data.output.resize(static_cast<std::size_t>(problem.m) * problem.n);
    for (float& value : data.a) value = distribution(generator);
    for (float& value : data.b) value = distribution(generator);
    for (float& value : data.bias) value = distribution(generator);
    return data;
}

std::vector<float> reference(const Problem& problem, const TensorData& data) {
    std::vector<float> output(static_cast<std::size_t>(problem.m) * problem.n, 0.0F);
    Schedule schedule;
    schedule.fused = true;
    execute_ijk(problem, schedule, data, output, 0, problem.m);
    return output;
}

void execute(const LoopIR& loop, const TensorData& data, std::vector<float>& output) {
    output.resize(static_cast<std::size_t>(loop.problem.m) * loop.problem.n);
    const PackedMatrices packed = pack_matrices(loop.problem, loop.schedule, data);
    const int thread_count = std::clamp(loop.schedule.threads, 1, loop.problem.m);
    auto worker = [&](int begin, int end, int cpu) {
#if defined(__linux__)
        if (loop.schedule.pin_threads) {
            cpu_set_t cpu_set;
            CPU_ZERO(&cpu_set);
            CPU_SET(preferred_cpu(cpu), &cpu_set);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
        } else {
            const cpu_set_t& cpu_set = allowed_cpu_set();
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
        }
#else
        (void)cpu;
#endif
        if (loop.schedule.order == LoopOrder::IJK) {
            execute_ijk(loop.problem, loop.schedule, data, output, begin, end);
        } else if (loop.schedule.pack_a || loop.schedule.pack_b) {
            std::fill(output.begin() + static_cast<std::size_t>(begin) * loop.problem.n,
                      output.begin() + static_cast<std::size_t>(end) * loop.problem.n, 0.0F);
            execute_packed(loop.problem, loop.schedule, data, packed, output, begin, end);
        } else if (loop.schedule.vector_width >= 8) {
            execute_register_blocked(loop.problem, loop.schedule, data, output, begin, end);
        } else {
            std::fill(output.begin() + static_cast<std::size_t>(begin) * loop.problem.n,
                      output.begin() + static_cast<std::size_t>(end) * loop.problem.n, 0.0F);
            execute_ikj(loop.problem, loop.schedule, data, output, begin, end);
        }
    };
    if (thread_count == 1) {
        worker(0, loop.problem.m, 0);
        return;
    }
    const int rows_per_thread = (loop.problem.m + thread_count - 1) / thread_count;
    thread_team(thread_count).run([&](int thread_index) {
        const int begin = thread_index * rows_per_thread;
        const int end = std::min(begin + rows_per_thread, loop.problem.m);
        if (begin < end) worker(begin, end, thread_index);
    });
}

double max_abs_error(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size()) return std::numeric_limits<double>::infinity();
    double error = 0.0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        error = std::max(error, std::abs(static_cast<double>(lhs[index] - rhs[index])));
    }
    return error;
}

BenchmarkResult benchmark(const LoopIR& loop, const TensorData& data,
                          int warmup, int repetitions) {
    const auto expected = reference(loop.problem, data);
    std::vector<float> output;
    for (int iteration = 0; iteration < std::max(0, warmup); ++iteration) {
        execute(loop, data, output);
    }
    std::vector<double> timings;
    for (int iteration = 0; iteration < std::max(1, repetitions); ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        execute(loop, data, output);
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double milliseconds = timings[timings.size() / 2];
    const double operations = 2.0 * loop.problem.m * loop.problem.n * loop.problem.k;
    return {milliseconds, operations / (milliseconds * 1.0e6), max_abs_error(expected, output)};
}

BenchmarkResult benchmark_bf16(const Problem& problem, const TensorData& data,
                               int repetitions) {
    const auto a_bf16 = convert_to_bf16(data.a);
    const auto b_bf16 = convert_to_bf16(data.b);
    const auto a = convert_from_bf16(a_bf16);
    const auto b = convert_from_bf16(b_bf16);
    const auto expected = reference(problem, data);
    std::vector<float> output(static_cast<std::size_t>(problem.m) * problem.n);
    std::vector<double> timings;
    for (int repetition = 0; repetition < std::max(1, repetitions); ++repetition) {
        std::fill(output.begin(), output.end(), 0.0F);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < problem.m; ++i) for (int j = 0; j < problem.n; ++j) {
            float accumulator = 0.0F;
            for (int k = 0; k < problem.k; ++k) {
                accumulator += a[static_cast<std::size_t>(i) * problem.k + k] *
                               b[static_cast<std::size_t>(k) * problem.n + j];
            }
            if (problem.bias) accumulator += data.bias[static_cast<std::size_t>(j)];
            output[static_cast<std::size_t>(i) * problem.n + j] =
                problem.relu ? std::max(0.0F, accumulator) : accumulator;
        }
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double milliseconds = timings[timings.size() / 2];
    const double operations = 2.0 * problem.m * problem.n * problem.k;
    return {milliseconds, operations / (milliseconds * 1.0e6), max_abs_error(expected, output)};
}

BenchmarkResult benchmark_int8(const Problem& problem, const TensorData& data,
                               int repetitions) {
    const auto a = quantize_int8(data.a);
    const auto b = quantize_int8(data.b);
    const auto expected = reference(problem, data);
    std::vector<float> output(static_cast<std::size_t>(problem.m) * problem.n);
    std::vector<double> timings;
    for (int repetition = 0; repetition < std::max(1, repetitions); ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < problem.m; ++i) for (int j = 0; j < problem.n; ++j) {
            std::int32_t accumulator = 0;
            for (int k = 0; k < problem.k; ++k) {
                accumulator += static_cast<std::int32_t>(a.values[static_cast<std::size_t>(i) * problem.k + k]) *
                               static_cast<std::int32_t>(b.values[static_cast<std::size_t>(k) * problem.n + j]);
            }
            float result = static_cast<float>(accumulator) * a.scale * b.scale;
            if (problem.bias) result += data.bias[static_cast<std::size_t>(j)];
            output[static_cast<std::size_t>(i) * problem.n + j] = problem.relu ? std::max(0.0F, result) : result;
        }
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(timings.begin(), timings.end());
    const double milliseconds = timings[timings.size() / 2];
    const double operations = 2.0 * problem.m * problem.n * problem.k;
    return {milliseconds, operations / (milliseconds * 1.0e6), max_abs_error(expected, output)};
}

}  // namespace schedforge
