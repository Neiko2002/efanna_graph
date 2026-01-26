cmake_minimum_required(VERSION 3.19)

project(efanna_benchmark)

set(EFANNA_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Efanna library from the main project (excluding index_pq which requires FAISS)
add_library(efanna2e_lib STATIC
    ${EFANNA_ROOT}/src/index.cpp
    ${EFANNA_ROOT}/src/index_graph.cpp
    ${EFANNA_ROOT}/src/index_kdtree.cpp
    ${EFANNA_ROOT}/src/index_random.cpp)

target_compile_features(efanna2e_lib PUBLIC cxx_std_17)

target_include_directories(efanna2e_lib PUBLIC
    ${EFANNA_ROOT}/include
    ${PROJECT_SOURCE_DIR}/include)

target_compile_options(efanna2e_lib PUBLIC
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native>)

target_link_libraries(efanna2e_lib PUBLIC
    OpenMP::OpenMP_CXX)

# Benchmark executable
add_executable(efanna_build_and_test
    ${PROJECT_SOURCE_DIR}/src/efanna_build_and_test.cpp)

target_link_libraries(efanna_build_and_test PUBLIC efanna2e_lib)

target_compile_definitions(efanna_build_and_test PUBLIC
    "DATA_PATH=\"${DATA_PATH}\"")
