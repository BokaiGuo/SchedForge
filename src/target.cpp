#include "schedforge/schedforge.h"

#include <sstream>
#include <thread>

namespace schedforge {

TargetInfo TargetInfo::detect() {
    TargetInfo target;
    target.logical_cpus = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
#if !defined(__AVX2__)
    target.isa = ISA::Scalar;
    target.vector_width = 1;
    target.vector_registers = 8;
#endif
    return target;
}

std::string TargetInfo::str() const {
    std::ostringstream out;
    const char* isa_name = isa == ISA::AVX512 ? "AVX512" : isa == ISA::AVX2 ? "AVX2" :
                           isa == ISA::NEON ? "NEON" : "scalar";
    out << architecture << ' ' << isa_name
        << " vector=" << vector_width << " registers=" << vector_registers
        << " cpus=" << logical_cpus;
    return out.str();
}

RegisterPressure estimate_register_pressure(const Schedule& schedule,
                                             const TargetInfo& target) {
    RegisterPressure pressure;
    pressure.accumulators = schedule.mr *
        std::max(1, (schedule.nr + std::max(1, schedule.vector_width) - 1) /
                    std::max(1, schedule.vector_width));
    pressure.broadcasts = schedule.vector_width > 1 ? 1 : 0;
    pressure.temporaries = 2 + (schedule.prefetch_distance > 0 ? 1 : 0);
    pressure.total = pressure.accumulators + pressure.broadcasts + pressure.temporaries;
    pressure.spills = pressure.total > target.vector_registers - 1;
    return pressure;
}

}  // namespace schedforge
