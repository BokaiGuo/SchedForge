#include "schedforge/compiler.h"

#include <iostream>

int main() {
    const schedforge::HardwareCalibrator calibrator;
    const auto calibration = calibrator.run();
    const auto target = calibrator.apply(schedforge::TargetInfo::detect(), calibration);
    std::cout << "target: " << target.str() << '\n'
              << "memory_bandwidth_gbps: " << calibration.memory_bandwidth_gbps << '\n'
              << "fma_proxy_gflops: " << calibration.fma_gflops << '\n'
              << "stream_ns: " << calibration.stream_nanoseconds << '\n';
}
