/**
 * @file execution_policy.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BARRACUDA_EXECUTION_POLICY_CUH_
#define BARRACUDA_EXECUTION_POLICY_CUH_

#include <cuda_profiler_api.h>

#include <cinttypes>

#include "barracuda/utils/predication.cuh"

namespace bcuda {

/**
 * @brief GPU kernel execution context
 */
__device__ struct KernelContext {
  // grid dimensions; number of blocks for each grid dimension: x,y,z
  dim3 gridDim;

  // block dimensions; number of threads for each block dimension: x,y,z
  dim3 blockDim;

  // block index within the grid
  dim3 blockIdx;

  // thread index within the block
  dim3 threadIdx;
};

struct KernelConfig {
  constexpr KernelConfig(uint32_t grid_x = 1, uint32_t grid_y = 1,
                         uint32_t grid_z = 1, uint32_t block_x = 1,
                         uint32_t block_y = 1, uint32_t block_z = 1)
      : gridDim_{grid_x, grid_y, grid_z},
        blockDim_{block_x, block_y, block_z} {};

  constexpr KernelConfig(dim3 const &grid_dim, dim3 const &block_dim)
      : gridDim_{grid_dim}, blockDim_{block_dim} {}

  constexpr dim3 const &Block() const noexcept { return blockDim_; }
  constexpr dim3 const &Grid() const noexcept { return gridDim_; }

  dim3 &Block() { return blockDim_; }
  dim3 &Grid() { return gridDim_; }

 private:
  dim3 gridDim_;
  dim3 blockDim_;
};

namespace execution_policy {
class LinearGridStride {
 public:
  __device__ inline static size_t ThreadIndex(
      KernelContext const &context) noexcept {
    return context.blockDim.x * context.blockIdx.x + context.threadIdx.x;
  }

  // Grid Stride
  __device__ inline static size_t Stride(
      KernelContext const &context) noexcept {
    return context.gridDim.x * context.blockDim.x;
  }

  inline static KernelConfig ConstructKernelConfig(
      cudaDeviceProp const &device_prop, size_t element_count) noexcept {
    // available number of blocks
    auto max_blocks = device_prop.maxGridSize[0];

    // available number of threads per block
    auto max_threads_per_block = device_prop.maxThreadsPerBlock;

    KernelConfig config{};

    auto grid_x = ((element_count + max_threads_per_block) - 1UL) /
                  static_cast<size_t>(max_threads_per_block);
    config.Grid().x = grid_x;
    config.Block().x = max_threads_per_block;

    BCUDA_ENSURE_TRUE_STR(
        grid_x <= static_cast<size_t>(max_blocks),
        "Element Size too large for Linear Grid Stride Execution");

    return config;
  }

};  // namespace linear_grid_stride

};  // namespace execution_policy
};  // namespace bcuda

#endif  // BARRACUDA_CUDA_UTILS_EXECUTION_POLICY_CUH_
