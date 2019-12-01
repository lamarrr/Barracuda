cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_STANDARD 14)

project(Barracuda
VERSION 0.1.1
DESCRIPTION "A CUDA GPU based Deep Learning Inference Micro-Framework"
LANGUAGES CXX CUDA
)

message(STATUS ${CMAKE_PROJECT_NAME} " v" ${CMAKE_PROJECT_VERSION})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "Release build")
else()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
message(STATUS "Debug build")
else()
message(STATUS "No recognized build profile selected")
endif()
endif()


set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
message(STATUS "Found CUDA v" ${CUDA_VERSION_STRING})
else()
message(FATAL_ERROR "Unable to Find CUDA")
endif()

find_package(fmt REQUIRED)


if(${fmt_FOUND})
message(STATUS "Found fmt v" ${fmt_VERSION})
else()
message(FATAL_ERROR "fmt not found")
endif()

#find_library(fmt_LIB fmt)
#if(NOT fmt_LIB)
#message(FATAL_ERROR "libfmt not found")
#else()
#message(STATUS "libfmt found")
#endif()


find_library(gtest_LIB gtest)
if(NOT gtest_LIB)
message(FATAL_ERROR "libgtest not found")
else()
message(STATUS "libgtest found")
endif()



if(CUDA_curand_LIBRARY)
message(STATUS "Curand: " ${CUDA_curand_LIBRARY})
else()
message(FATAL_ERROR "libcurand not found")
endif()

find_package(WebP REQUIRED)

find_library(Vpx_LIB vpx)
if(Vpx_LIB)
message(STATUS "Found libvpx: " ${Vpx_LIB})
else()
message(FATAL_ERROR "libvpx not found")
endif()


if(WebP_FOUND)
message(STATUS "WebP: " ${WebP_INCLUDE_DIRS})
endif()


if(CUDA_CUBLAS_LIBRARIES)
message(STATUS "Found cublas: " ${CUDA_CUBLAS_LIBRARIES})
else()
message(FATAL_ERROR "cublas not found")
endif()

find_package(Qt 5.3.1 REQUIRED)
if(Qt_FOUND)
message(STATUS "Found Qt v" ${Qt_VERSION})
else()
message(FATAL_ERROR "Qt not found")
endif()



set(BCUDA_CXX_COMPILE_FLAGS "-Wall -Wextra -pedantic -Werror")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${BCUDA_CXX_COMPILE_FLAGS})


message(STATUS "Compiler: " ${CMAKE_CXX_COMPILER_ID} " v" ${CMAKE_CXX_COMPILER_VERSION})

include("${CMAKE_CURRENT_SOURCE_DIR}/barracuda/CMakeLists.txt")


include("${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
