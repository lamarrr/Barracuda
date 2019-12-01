
message(STATUS "Source dir: " ${CMAKE_CURRENT_SOURCE_DIR})

add_library(barracuda STATIC 
${CMAKE_CURRENT_LIST_DIR}/ops/conv.cu
${CMAKE_CURRENT_LIST_DIR}/rng/rng.cu
${CMAKE_CURRENT_LIST_DIR}/tensor.cu
)


add_executable(Barracuda.bin 
${CMAKE_CURRENT_LIST_DIR}/barracuda.cc
${CMAKE_CURRENT_LIST_DIR}/execute.cu
)

add_library(image_tensor_test STATIC
${CMAKE_CURRENT_LIST_DIR}/image.cu)

# change to PRIVATE as appropriate
target_include_directories(Barracuda.bin PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(barracuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(image_tensor_test PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" ${WebP_INCLUDE_DIRS})


target_link_libraries(barracuda fmt::fmt ${CUDA_curand_LIBRARY})
target_link_libraries(Barracuda.bin barracuda)
target_link_libraries(image_tensor_test barracuda fmt::fmt  webp webpdecoder)

