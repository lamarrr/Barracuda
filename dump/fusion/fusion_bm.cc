#include <benchmark/benchmark.h>  // NOLINT
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>

#include "barracuda/fusion/fusion.h"
#include "barracuda/utils/predication.cuh"

constexpr auto kSize = 184746;

// static void Benchmark_KernelFusion(benchmark::State& state) {  // NOLINT
void Benchmark_KernelFusion() {
  
  // for (auto _ : state) {
  for (int i = 0; i < 10000; i++) {
    float* data = nullptr;
    BCUDA_ENSURE_OK(cudaMallocManaged(&data, kSize));
    LaunchKernelFused(data, kSize);
    BCUDA_ENSURE_OK(cudaDeviceSynchronize());
    BCUDA_ENSURE_OK(cudaFree(data));
    std::cout << "Fused Iter.: " << i << std::endl;
  }
}

// static void Benchmark_Normal(benchmark::State& state) {  // NOLINT
void Benchmark_Normal() {
  // for (auto _ : state) {
  for (int i = 0; i < 10000; i++) {
    float* data = nullptr;
    BCUDA_ENSURE_OK(cudaMallocManaged(&data, kSize));
    LaunchKernelUnfused(data, kSize);
    BCUDA_ENSURE_OK(cudaDeviceSynchronize());
    BCUDA_ENSURE_OK(cudaFree(data));
    std::cout << "Unfused Iter.: " << i << std::endl;
  }
}
/*
BENCHMARK(Benchmark_Normal)
    ->Iterations(5000)
    ->ThreadPerCpu()
    ->Repetitions(10);

BENCHMARK(Benchmark_KernelFusion)
    ->Iterations(5000)
    ->ThreadPerCpu()
    ->Repetitions(10);

BENCHMARK_MAIN();
*/

int main() {
  Benchmark_Normal();
  Benchmark_KernelFusion();
}