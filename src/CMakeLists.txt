set(CPP_SOURCES index_graph.cpp index_kdtree.cpp index_random.cpp index.cpp)

if(NOT WIN32)
    list(APPEND CPP_SOURCES index_pq.cpp)
endif()

add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                  $<INSTALL_INTERFACE:include>)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

target_compile_options(${PROJECT_NAME} PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
                                              $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native>)

target_compile_definitions(${PROJECT_NAME} PUBLIC INFO)
