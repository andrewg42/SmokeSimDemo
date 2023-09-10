cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)

project(SmokeSim LANGUAGES CXX CUDA)

include_directories(./utils)

add_executable(main main.cu)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

find_package(OpenVDB REQUIRED)
message(STATUS "Found OpenVDB ${OpenVDB_VERSION} at ${OpenVDB_LIBRARIES}")
target_link_libraries(main PUBLIC OpenVDB::openvdb)
target_sources(main PUBLIC writevdb.cpp)

