cmake_minimum_required(VERSION 3.24)
project(GeoThermalParallel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Arquitectura Turing para GTX 1650 Ti Mobile
set(CMAKE_CUDA_ARCHITECTURES 75)

# Banderas del compilador C++
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
    )
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Banderas para NVCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Solo buscar OpenMP para CXX (evita que busque OpenMP_CUDA)
find_package(OpenMP REQUIRED COMPONENTS CXX)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(SOURCES
    src/main.cpp
    src/cpu_engine.cpp
    src/cuda_engine.cu
    src/exporter.cpp
)

add_executable(GeoThermalParallel ${SOURCES})

target_link_libraries(GeoThermalParallel PRIVATE 
    OpenMP::OpenMP_CXX
    ws2_32.lib
)

set_target_properties(GeoThermalParallel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
