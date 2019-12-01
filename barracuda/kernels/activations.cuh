/**
 * @file activations.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BARRACUDA_KERNELS_ACTIVATIONS_CUH_
#define BARRACUDA_KERNELS_ACTIVATIONS_CUH_

#include <cinttypes>
#include <type_traits>

#include "barracuda/execution_policy.cuh"

namespace bcuda {
namespace gpu_kernels {

namespace hlpr_ {

// TODO(lamarrr): rounding et al
template <typename T>
__device__ inline T Sigmoid(T x) {
  static_assert(true, "Sigmoid undefined for type");
  return {};
}

template <>
__device__ inline float Sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
template <>
__device__ inline double Sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}
// __device__ inline int8_t Sigmoid(int8_t x) { return exp(x) / (1.0 + exp(x));
// }

template <typename T>
__device__ inline T ReLU(T x, T threshold = static_cast<T>(0)) {
  return x > threshold ? x : threshold;
}

template <typename T>
__device__ inline T ReLU6(T x, T threshold = static_cast<T>(0),
                          T max_point = static_cast<T>(6)) {
  T y = ReLU(x, threshold);
  return y > max_point ? max_point : y;
}

// only correct for floating points
template <typename T>
__device__ inline T Swish(T x, T b) {
  static_assert(true, "Swish Not yet implemented for type");
  return {};
}

template <>
__device__ inline float Swish(float x, float b) {
  return x * Sigmoid(x * b);
}

template <>
__device__ inline double Swish(double x, double b) {
  return x * Sigmoid(x * b);
}

// only correct for floating points
template <typename T>
__device__ inline T HardSwish(T x, T b, T threshold, T max_point) {
  static_assert(true, "HardSwish not yet implemented for type");
  return {};
}

template <>
__device__ inline float HardSwish(float x, float b, float threshold,
                                  float max_point) {
  auto y = x * ReLU6(x + b, threshold, max_point);
  return y / 6.0f;
}

template <>
__device__ inline double HardSwish(double x, double b, double threshold,
                                   double max_point) {
  auto y = x * ReLU6(x + b);
  return y / 6.0;
}

};  // namespace hlpr_

template <typename T, typename ExecutionPolicy =
                          ::bcuda::execution_policy::LinearGridStride>
__global__ inline void ReLU(size_t n, T const *input, T *output) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    output[i] = hlpr_::ReLU(input[i]);
  }
}

template <typename T, typename ExecutionPolicy =
                          ::bcuda::execution_policy::LinearGridStride>
__global__ inline void ReLUInplace(size_t n, T *input) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    input[i] = hlpr_::ReLU(input[i]);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void ReLU6(size_t n, float const *input, float *output) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    output[i] = hlpr_::ReLU6(input[i]);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void ReLU6(size_t n, int8_t const *input, int8_t *output) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    output[i] = hlpr_::ReLU6(input[i]);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void ReLU6Inplace(size_t n, float *input) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    input[i] = hlpr_::ReLU6(input[i]);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void ReLU6Inplace(size_t n, int8_t *input) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    input[i] = hlpr_::ReLU6(input[i]);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void Swish(size_t n, float const *input, float *output,
                             float beta) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    output[i] = input[i] * hlpr_::Sigmoid(input[i] * beta);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void SwishInplace(size_t n, float *input, float beta) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    input[i] = input[i] * hlpr_::Sigmoid(input[i] * beta);
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void HardSwish(size_t n, float const *input, float *output,
                                 float beta = 3.0f) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    output[i] = (input[i] * hlpr_::ReLU6(input[i] + beta)) / 6.0f;
  }
}

template <
    typename ExecutionPolicy = ::bcuda::execution_policy::LinearGridStride>
__global__ inline void HardSwishInplace(size_t n, float *input,
                                        float beta = 3.0f) {
  ::bcuda::KernelContext context{gridDim, blockDim, blockIdx, threadIdx};

  size_t index = ExecutionPolicy::ThreadIndex(context);
  size_t stride = ExecutionPolicy::Stride(context);

  for (size_t i = index; i < n; i += stride) {
    input[i] = (input[i] * hlpr_::ReLU6(input[i] + beta)) / 6.0f;
  }
}

}  // namespace gpu_kernels
}  // namespace bcuda

#endif  // BARRACUDA_KERNELS_ACTIVATIONS_CUH_
