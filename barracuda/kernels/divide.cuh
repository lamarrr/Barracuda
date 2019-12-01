/**
 * @file divide.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef BARRACUDA_KERNELS_DIVIDE_CUH_
#define BARRACUDA_KERNELS_DIVIDE_CUH_

#include <cinttypes>
#include <cmath>

#include "barracuda/execution_policy.cuh"

namespace bcuda {
namespace gpu_kernels {

template <typename T, typename ExecutionPolicy =
                          ::bcuda::execution_policy::LinearGridStride>
__global__ inline void Divide(size_t n, T const *input_1, T const *input_2,
                              T *output) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    output[i] = input_1[i] / input_2[i];
  }
}

template <typename T, typename ExecutionPolicy =
                          ::bcuda::execution_policy::LinearGridStride>
__global__ inline void DivideInplace(size_t n, T *input_1, T const *input_2) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    input_1[i] = input_1[i] / input_2[i];
  }
}
}  // namespace gpu_kernels
}  // namespace bcuda

#endif  // BARRACUDA_KERNELS_DIVIDE_CUH_
