
#include <gtest/gtest.h>

#include "barracuda/kernels/activations.cuh"
#include "barracuda/tensor.cuh"

namespace kerns = bcuda::gpu_kernels;

TEST(AddKernel1d, Uniform) {
  auto tensor_x = bcuda::Tensor<double>::Zeros({46, 2876});
  auto tensor_y = bcuda::Tensor<double>::Ones({46, 2876});

  kerns::AddInplace<<<1, 1>>>(tensor_x.Size(), tensor_x.Data(),
                              tensor_y.Data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaError_t::cudaSuccess);

  for (auto i : tensor_x) {
    ASSERT_FLOAT_EQ(i, 1.0f);
  }
}
