/**
 * @file memory.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef BARRACUDA_UTILS_MEMORY_CUH_
#define BARRACUDA_UTILS_MEMORY_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <memory>

#include "barracuda/utils/predication.cuh"

namespace bcuda {

template <typename T>
inline T *CudaAllocate(size_t n) noexcept {
  T *mem;
  BCUDA_ENSURE_OK_STR(cudaMallocManaged(&mem, n * sizeof(T)),
                      "Memory Allocation Failed");
  return mem;
}

template <typename T>
inline void CudaDellocate(T *mem) noexcept {
  BCUDA_ENSURE_OK_STR(cudaFree(mem), "Memory Deallocation Failed");
}

template <typename T>
class CudaAllocator {
 public:
  using value_type = T;
  T *allocate(size_t n) const noexcept { return CudaAllocate<T>(n); }
  void deallocate(T *mem) const noexcept { return CudaDellocate(mem); }
};

class CudaDeallocator {
 public:
  CudaDeallocator() {}
  template <typename T>
  void operator()(T *mem) const noexcept {
    if (mem != nullptr) cudaFree(mem);
  }
};

template <typename T>
using CudaArray = std::unique_ptr<T[], CudaDeallocator>;

template <typename H>
constexpr inline H GetDimSize(H h) noexcept {
  static_assert(std::is_integral<H>::value, "Use Integral Type For Dimension");
  return h;
}

template <typename H, typename... T>
constexpr inline H GetDimSize(H h, T... t) noexcept {
  static_assert(std::is_integral<H>::value, "Use Integral Type For Dimension");
  return h * GetDimSize(t...);
}

template <typename T, typename HeadT, typename... OthersT>
CudaArray<T> MakeCudaArray(HeadT head, OthersT... dims) {
  // static_assert(sizeof...(dims) == 1U, "Only 2D Arrays Supported For Now");
  return CudaArray<T>{CudaAllocate<T>(GetDimSize(head, dims...))};
}

}  // namespace bcuda
#endif  // BARRACUDA_CUDA_UTILS_MEMORY_CUH_
