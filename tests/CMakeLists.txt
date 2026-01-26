add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent ${PROJECT_NAME})

add_executable(test_nndescent_refine test_nndescent_refine.cpp)
target_link_libraries(test_nndescent_refine ${PROJECT_NAME})

add_executable(test_kdtree_graph test_kdtree_graph.cpp)
target_link_libraries(test_kdtree_graph ${PROJECT_NAME})

add_executable(test_nndescent_sift1m test_nndescent_sift1m.cpp)
target_link_libraries(test_nndescent_sift1m ${PROJECT_NAME})

add_executable(test_nndescent_sift1m_evaluation test_nndescent_sift1m_evaluation.cpp)
target_link_libraries(test_nndescent_sift1m_evaluation ${PROJECT_NAME})

add_executable(test_efanna_exploration test_efanna_exploration.cpp)
target_link_libraries(test_efanna_exploration ${PROJECT_NAME})

add_executable(test_efanna_stats test_efanna_stats.cpp)
target_link_libraries(test_efanna_stats ${PROJECT_NAME})

add_executable(test_nndescent_glove test_nndescent_glove.cpp)
target_link_libraries(test_nndescent_glove ${PROJECT_NAME})

add_executable(test_nndescent_glove_evaluation test_nndescent_glove_evaluation.cpp)
target_link_libraries(test_nndescent_glove_evaluation ${PROJECT_NAME})

add_executable(test_efanna_graph test_efanna_graph.cpp)
target_link_libraries(test_efanna_graph ${PROJECT_NAME})

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
