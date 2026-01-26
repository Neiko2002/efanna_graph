cmake_minimum_required(VERSION 3.19)

project(kgraph_benchmark)

set(KGRAPH_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
set(DATA_PATH "" CACHE PATH "Path to data sets")

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)

add_library(kgrapha STATIC
    ${KGRAPH_ROOT}/kgraph.cpp
    ${KGRAPH_ROOT}/metric.cpp)

target_compile_features(kgrapha PUBLIC cxx_std_17)

target_include_directories(kgrapha PUBLIC
    ${KGRAPH_ROOT}
    ${PROJECT_SOURCE_DIR}/include)

target_compile_options(kgrapha PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native>)

target_link_libraries(kgrapha PUBLIC
    OpenMP::OpenMP_CXX
    Boost::headers
    Boost::chrono
    Boost::system
    Boost::program_options
    Boost::timer)

target_compile_definitions(kgrapha PRIVATE
    KGRAPH_VERSION=1.5
    KGRAPH_BUILD_NUMBER=1
    KGRAPH_BUILD_ID=1)

add_executable(kgraph_build_and_test
    ${PROJECT_SOURCE_DIR}/src/kgraph_build_and_test.cpp)

target_link_libraries(kgraph_build_and_test PUBLIC kgrapha)

target_compile_definitions(kgraph_build_and_test PUBLIC
    "DATA_PATH=\"${DATA_PATH}\"")
