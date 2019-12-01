/**
 * @file barracuda.cc
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <fmt/format.h>

#include "barracuda/utils/predication.cuh"

namespace bcuda {
extern int Execute(cudaDeviceProp const &);
}
// namespace cu

int main() {
  int count{};
  BCUDA_CHECK_OK_STR(cudaGetDeviceCount(&count),
                     "Unable to Get CUDA supported GPU device count");
  fmt::print("Device Count: {}\n", count);
  BCUDA_CHECK_TRUE_STR(count > 0, "No CUDA supported GPU device found");
  int device_id = 0;
  fmt::print("[#] Selecting device: GPU:{}\n", device_id);
  BCUDA_ENSURE_OK_STR(cudaSetDevice(device_id), "Error setting device");

  cudaDeviceProp device_props{};
  BCUDA_CHECK_OK_STR(cudaGetDeviceProperties(&device_props, device_id),
                     "Error getting device properties");

  fmt::print(
      "\tDevice Name: {}\n\tClock Rate (kHz): {}\n\tMaximum Threads Per Block: "
      "{}\n\tMultiprocessor Count: {}\n\tMaximum Threads Per Multiprocessor: "
      "{}\n",
      device_props.name, device_props.clockRate,
      device_props.maxThreadsPerBlock, device_props.multiProcessorCount,
      device_props.maxThreadsPerMultiProcessor);

  fmt::print("\tMaximum Grid Size:\t\tX = {}\tY = {}\tZ = {}\n",
             device_props.maxGridSize[0], device_props.maxGridSize[1],
             device_props.maxGridSize[2]);

  fmt::print("\tMaximum Threads Dimension:\tX = {}\tY = {}\tZ = {}\n",
             device_props.maxThreadsDim[0], device_props.maxThreadsDim[1],
             device_props.maxThreadsDim[2]);
  fmt::print("\tTotal Global Mem: {}\n", device_props.totalGlobalMem);

  fmt::print("[#] Running GPU code\n");
  return bcuda::Execute(device_props);
}