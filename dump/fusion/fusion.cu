#include <functional>

#include "barracuda/fusion/fusion.cuh"

__device__ inline float X(float g) noexcept { return g * 4; }

__device__ inline float XX(float g) { return g * 4; }

__device__ inline float XXX(float g) { return g * 4; }

__device__ inline float XXXX(float g) { return g * 4; }

__device__ inline float XXXXX(float g) { return g * 4; }

__global__ void KernelFused(float* data, size_t n,
                            std::function<__device__ float(float)>* func) {
  for (int i = 0; i < n; i++) data[i] = func->operator()(data[i]);
}

__global__ void KernelUnfused(float* data, size_t n) {
  for (int i = 0; i < n; i++) data[i] = X(XX(XXX(XXXX(XXXXX(data[i])))));
}

void LaunchKernelFused(float* data, size_t n) {
  using func_t = std::function<__device__ float(float)>;  // NOLINT
  func_t* funcptr = nullptr;
  cudaMallocManaged(&funcptr, sizeof(func_t));
  *funcptr = [] __device__(float u) { return u * 2; };  // NOLINT
  KernelFused<<<1, 1>>>(data, n, funcptr);
  cudaFree(funcptr);
}

void LaunchKernelUnfused(float* data, size_t n) {
  KernelUnfused<<<1, 1>>>(data, n);
}
