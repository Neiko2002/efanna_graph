add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent ${PROJECT_NAME})

add_executable(test_nndescent_refine test_nndescent_refine.cpp)
target_link_libraries(test_nndescent_refine ${PROJECT_NAME})

add_executable(test_kdtree_graph test_kdtree_graph.cpp)
target_link_libraries(test_kdtree_graph ${PROJECT_NAME})

if(NOT WIN32)
    # OpenBLAS
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})

    # external libraries like faiss
    include_directories(${PROJECT_SOURCE_DIR}/extern_libraries)
    link_directories(${PROJECT_SOURCE_DIR}/extern_libraries/faiss/lib)

    add_executable(test_faiss_graph test_faiss_graph.cpp)
    target_link_libraries(test_faiss_graph ${PROJECT_NAME} faiss ${BLAS_LIB})
endif()
