#include "schedforge/schedforge.h"
#include "schedforge/compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
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
            CPU_SET(cpu, &cpu_set);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
        }
#else
        (void)cpu;
#endif
        std::fill(output.begin() + static_cast<std::size_t>(begin) * loop.problem.n,
                  output.begin() + static_cast<std::size_t>(end) * loop.problem.n, 0.0F);
        if (loop.schedule.order == LoopOrder::IJK) {
            execute_ijk(loop.problem, loop.schedule, data, output, begin, end);
        } else if (loop.schedule.pack_a || loop.schedule.pack_b) {
            execute_packed(loop.problem, loop.schedule, data, packed, output, begin, end);
        } else {
            execute_ikj(loop.problem, loop.schedule, data, output, begin, end);
        }
    };
    if (thread_count == 1) {
        worker(0, loop.problem.m, 0);
        return;
    }
    std::vector<std::thread> threads;
    const int rows_per_thread = (loop.problem.m + thread_count - 1) / thread_count;
    for (int thread = 0; thread < thread_count; ++thread) {
        const int begin = thread * rows_per_thread;
        const int end = std::min(begin + rows_per_thread, loop.problem.m);
        if (begin < end) threads.emplace_back(worker, begin, end, thread);
    }
    for (auto& thread : threads) thread.join();
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
