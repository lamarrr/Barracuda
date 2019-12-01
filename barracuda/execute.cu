#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cudart_platform.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>

#include "barracuda/execution_policy.cuh"
#include "barracuda/kernels/kernels.cuh"
#include "barracuda/tensor.cuh"
#include "barracuda/utils/memory.cuh"
#include "barracuda/utils/predication.cuh"

namespace bcuda {
int Execute(cudaDeviceProp const &device_props) {
  std::cout << "[#] Starting Profiler" << std::endl;
  BCUDA_CHECK_OK(cudaProfilerStart());

  constexpr size_t N = static_cast<size_t>(1U) << 20;

  Tensor<float> x{{N}};
  Tensor<float> y{{N}};

  Tensor<float> tens = Tensor<float>::RandomUniformUnison({N}, 10.0f, 25.0f);

  std::iota(x.begin(), x.end(), -100.5f);
  std::iota(y.begin(), y.end(), -101.5f);

  std::cout << "First: " << x.Data()[0] << "\tLast: " << x.Data()[N - 1]
            << std::endl;

  using ExecutionPolicy = bcuda::execution_policy::LinearGridStride;
  auto config = ExecutionPolicy::ConstructKernelConfig(device_props, N);

  Tensor<int> xx = Tensor<int>::Ones({N});
  Tensor<int> yy = Tensor<int>::Ones({N});

  gpu_kernels::AddInplace<<<config.Grid(), config.Block()>>>(N, xx.Data(),
                                                             yy.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  // for (auto i : xx) {
  // BCUDA_ENSURE_TRUE_STR(i == 2, "Linear Add Kernel invalid");
  //}
  // std::cout << "[*]"
  //          "Linear Add kernel Valid\n";

  gpu_kernels::SubtractInplace<<<config.Grid(), config.Block()>>>(N, x.Data(),
                                                                  y.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  gpu_kernels::ElementwiseMulInplace<<<config.Grid(), config.Block()>>>(N, x.Data(),
                                                                  y.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  gpu_kernels::DivideInplace<<<config.Grid(), config.Block()>>>(N, x.Data(),
                                                                y.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  gpu_kernels::ReLUInplace<<<config.Grid(), config.Block()>>>(N, x.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  gpu_kernels::ReLU6Inplace<<<config.Grid(), config.Block()>>>(N, x.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  Tensor<float> tensor = Tensor<float>::Ones({1, 1, 1, N, 2});

  gpu_kernels::SwishInplace<<<config.Grid(), config.Block()>>>(
      tensor.Size(), tensor.Data(), 0.1);
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  gpu_kernels::HardSwishInplace<<<config.Grid(), config.Block()>>>(
      tensor.Size(), tensor.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  gpu_kernels::ExpInplace<<<config.Grid(), config.Block()>>>(tensor.Size(),
                                                             tensor.Data());
  BCUDA_CHECK_OK(cudaDeviceSynchronize());

  std::cout << (x.Data()[N - 1] + x.Data()[N - 1]) << std::endl;
  std::cout << "[#] Stopping Profiler" << std::endl;

  BCUDA_CHECK_OK(cudaProfilerStop());

  return EXIT_SUCCESS;
}
}  // namespace bcuda
