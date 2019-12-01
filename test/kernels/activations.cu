
#include <gtest/gtest.h>

#include "barracuda/kernels/activations.cuh"
#include "barracuda/tensor.cuh"

namespace kerns = bcuda::gpu_kernels;

TEST(ReLUKernel1d, ValueZero) {
  auto tensor = bcuda::Tensor<double>::Zeros({46, 2876});
  kerns::ReLUInplace<<<1, 1>>>(tensor.Size(), tensor.Data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaError_t::cudaSuccess);

  for (auto i : tensor) {
    ASSERT_FLOAT_EQ(i, 0.0f);
  }
}

TEST(ReLUKernel1d, ValueOne) {
  auto tensor = bcuda::Tensor<float>::Ones({48, 383898});
  kerns::ReLUInplace<<<1, 1>>>(tensor.Size(), tensor.Data());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaError_t::cudaSuccess);

  for (auto i : tensor) {
    ASSERT_FLOAT_EQ(i, 1.0f);
  }
}

TEST(ReLUKernel1d, NegativeValue) {
  auto tensor = bcuda::Tensor<double>({48, 383898});

  kerns::LinearFill<<<1, 1>>>(tensor.Size(), tensor.Data(), -1.0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaError_t::cudaSuccess);

  kerns::ReLUInplace<<<1, 1>>>(tensor.Size(), tensor.Data());

  ASSERT_EQ(cudaDeviceSynchronize(), cudaError_t::cudaSuccess);

  for (auto i : tensor) {
    ASSERT_FLOAT_EQ(i, 0.0f);
  }
}
