




add_executable(BarracudaTest.bin
${CMAKE_CURRENT_LIST_DIR}/test.cc
${CMAKE_CURRENT_LIST_DIR}/tensor.cu
${CMAKE_CURRENT_LIST_DIR}/kernels/activations.cu
${CMAKE_CURRENT_LIST_DIR}/kernels/add.cu
${CMAKE_CURRENT_LIST_DIR}/kernels/multiply.cu
)

# change to PUBLIC as appropriate
target_include_directories(BarracudaTest.bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(BarracudaTest.bin ${gtest_LIB} fmt::fmt barracuda)

