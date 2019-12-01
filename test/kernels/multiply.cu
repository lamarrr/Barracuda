
#include <gtest/gtest.h>

#include "barracuda/kernels/activations.cuh"
#include "barracuda/tensor.cuh"

namespace kerns = bcuda::gpu_kernels;

TEST(MultiplyKernel1d, Uniform) {
  auto tensor_x = bcuda::Tensor<double>({46, 2876}, 122.2);
  auto tensor_y = bcuda::Tensor<double>({46, 2876}, 0.5);

  kerns::ElementwiseMulInplace<<<1, 1>>>(tensor_x.Size(), tensor_x.Data(),
                                   tensor_y.Data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaError_t::cudaSuccess);

  for (auto i : tensor_x) {
    ASSERT_DOUBLE_EQ(i, 61.1);
  }
}
