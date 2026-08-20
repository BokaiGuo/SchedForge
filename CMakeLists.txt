cmake_minimum_required(VERSION 3.20)
project(SchedForge VERSION 0.16.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SCHEDFORGE_NATIVE_ARCH "Enable native CPU instructions" ON)
option(SCHEDFORGE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(SCHEDFORGE_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "SchedForge sanitizers require Clang or GCC")
    endif()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

find_package(LLVM REQUIRED CONFIG PATHS /usr/lib/llvm-18/lib/cmake/llvm /usr/lib/llvm-18/cmake)
if(NOT LLVM_VERSION_MAJOR EQUAL 18)
    message(FATAL_ERROR "SchedForge requires LLVM 18, found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

add_library(schedforge
    src/generic_ir.cpp
    src/ir.cpp
    src/target.cpp
    src/compiler.cpp
    src/graph_compiler.cpp
    src/moe_compiler.cpp
    src/attention_compiler.cpp
    src/attention_llvm_backend.cpp
    src/decoder_compiler.cpp
    src/llvm_backend.cpp
    src/aot_runtime.cpp
    src/next_milestones.cpp
    src/runtime.cpp
    src/simulator.cpp
    src/autoscheduler.cpp
)
target_include_directories(schedforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(schedforge SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(schedforge PRIVATE ${LLVM_DEFINITIONS})
target_link_libraries(schedforge PUBLIC LLVM)
target_link_libraries(schedforge PRIVATE ${CMAKE_DL_LIBS})
target_compile_options(schedforge PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
if(SCHEDFORGE_NATIVE_ARCH)
    target_compile_options(schedforge PRIVATE -march=native)
endif()

add_executable(schedforge-bench src/main.cpp)
target_link_libraries(schedforge-bench PRIVATE schedforge)

add_executable(schedforge-opt tools/schedforge-opt.cpp)
target_link_libraries(schedforge-opt PRIVATE schedforge)
add_executable(schedforge-sim tools/schedforge-sim.cpp)
target_link_libraries(schedforge-sim PRIVATE schedforge)
add_executable(schedforge-run tools/schedforge-run.cpp)
target_link_libraries(schedforge-run PRIVATE schedforge)
add_executable(schedforge-study tools/schedforge-study.cpp)
target_link_libraries(schedforge-study PRIVATE schedforge)
add_executable(schedforge-calibrate tools/schedforge-calibrate.cpp)
target_link_libraries(schedforge-calibrate PRIVATE schedforge)
add_executable(schedforge-search tools/schedforge-search.cpp)
target_link_libraries(schedforge-search PRIVATE schedforge)
add_executable(schedforge-resolution tools/schedforge-resolution.cpp)
target_link_libraries(schedforge-resolution PRIVATE schedforge)
add_executable(schedforge-compile tools/schedforge-compile.cpp)
target_link_libraries(schedforge-compile PRIVATE schedforge)
add_executable(schedforge-moe tools/schedforge-moe.cpp)
target_link_libraries(schedforge-moe PRIVATE schedforge)
add_executable(schedforge-attention tools/schedforge-attention.cpp)
target_link_libraries(schedforge-attention PRIVATE schedforge)
add_executable(schedforge-decoder tools/schedforge-decoder.cpp)
target_link_libraries(schedforge-decoder PRIVATE schedforge)
add_executable(schedforge-decoder-bench tools/schedforge-decoder-bench.cpp)
target_link_libraries(schedforge-decoder-bench PRIVATE schedforge)
add_executable(schedforge-codegen-study tools/schedforge-codegen-study.cpp)
target_link_libraries(schedforge-codegen-study PRIVATE schedforge)
add_executable(schedforge-aot tools/schedforge-aot.cpp)
target_link_libraries(schedforge-aot PRIVATE schedforge)
add_executable(schedforge-aot-study tools/schedforge-aot-study.cpp)
target_link_libraries(schedforge-aot-study PRIVATE schedforge)
add_executable(schedforge-next-study tools/schedforge-next-study.cpp)
target_link_libraries(schedforge-next-study PRIVATE schedforge)
add_executable(schedforge-fuzz tools/schedforge-fuzz.cpp)
target_link_libraries(schedforge-fuzz PRIVATE schedforge)

enable_testing()
add_executable(schedforge-tests tests/test_schedforge.cpp)
target_link_libraries(schedforge-tests PRIVATE schedforge)
target_compile_definitions(schedforge-tests PRIVATE
    SCHEDFORGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME schedforge-tests COMMAND schedforge-tests)
add_test(NAME schedforge-aot-compile COMMAND schedforge-aot compile
    --M=13 --N=17 --K=11 --output=${CMAKE_CURRENT_BINARY_DIR}/ctest-matmul.sfe)
set_tests_properties(schedforge-aot-compile PROPERTIES FIXTURES_SETUP schedforge-aot-artifact)
add_test(NAME schedforge-aot-run COMMAND schedforge-aot run
    --artifact=${CMAKE_CURRENT_BINARY_DIR}/ctest-matmul.sfe --warmup=0 --repetitions=1)
set_tests_properties(schedforge-aot-run PROPERTIES FIXTURES_REQUIRED schedforge-aot-artifact)
add_test(NAME schedforge-next-study COMMAND schedforge-next-study --quick)
add_test(NAME schedforge-fuzz COMMAND schedforge-fuzz --iterations=128 --seed=17)

install(TARGETS schedforge
    EXPORT SchedForgeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS
    schedforge-bench
    schedforge-opt
    schedforge-sim
    schedforge-run
    schedforge-study
    schedforge-calibrate
    schedforge-search
    schedforge-resolution
    schedforge-compile
    schedforge-moe
    schedforge-attention
    schedforge-decoder
    schedforge-decoder-bench
    schedforge-codegen-study
    schedforge-aot
    schedforge-aot-study
    schedforge-next-study
    schedforge-fuzz
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/schedforge DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT SchedForgeTargets
    FILE SchedForgeTargets.cmake
    NAMESPACE SchedForge::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SchedForge
)

configure_package_config_file(
    cmake/SchedForgeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SchedForgeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SchedForge
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SchedForgeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SchedForgeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SchedForgeConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SchedForge
)
