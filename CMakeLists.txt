cmake_minimum_required(VERSION 3.19)

project(efanna2e)

#OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(DATA_PATH "" CACHE PATH "Path to data sets")

add_subdirectory(src)
add_subdirectory(benchmark)
