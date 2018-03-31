cmake_minimum_required(VERSION 3.8)
project(wmsketch)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        src/countmin.cpp
        src/countsketch.cpp
        src/dataset.cpp
        src/hash.cpp
        src/heap.cpp
        src/logistic.cpp
        src/logistic_sketch.cpp
        src/paired_countmin.cpp
        src/topk.cpp
        src/util.cpp
        src/sgns.cpp)

add_library(wmsketch ${SOURCE_FILES})
target_include_directories(wmsketch PUBLIC include)

add_executable(wmsketch_classification
        src/experiments/cxxopts.hpp
        src/experiments/json.hpp
        src/experiments/classification.cpp)
target_link_libraries(wmsketch_classification wmsketch)

add_executable(wmsketch_pmi
        src/experiments/cxxopts.hpp
        src/experiments/json.hpp
        src/experiments/pmi.cpp)
target_link_libraries(wmsketch_pmi wmsketch)
