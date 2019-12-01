#include <cuda.h>
#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <type_traits>

#include "barracuda/fusion/fusion.h"

template <typename HeadFuncSig, typename... TailFuncSig>
struct UnaryFunctionFuse {
  using HeadType = HeadFuncSig*;

  UnaryFunctionFuse<TailFuncSig...> tail;

  __device__ constexpr UnaryFunctionFuse(HeadType h, TailFuncSig... t)
      : head_{h}, tail{t...} {}

  template <typename Param>
  __forceinline__ __device__ auto operator()(Param const& param) const
      noexcept {
    return head_(tail(param));
  }
  __device__ constexpr HeadType Head() { return head_; }

 private:
  HeadType head_;
};

template <typename HeadFuncSig>
struct UnaryFunctionFuse<HeadFuncSig> {
  using HeadType = HeadFuncSig*;

  __device__ constexpr explicit UnaryFunctionFuse(HeadType h) : head_{h} {}

  __device__ constexpr HeadType Head() { return head_; }

  template <typename Param>
  __forceinline__ __device__ auto operator()(Param const& param) const
      noexcept {
    return head_(param);
  }

 private:
  HeadType head_;
};

__device__ inline float X(float g) noexcept;

__device__ inline float XX(float g);

__device__ inline float XXX(float g);

__device__ inline float XXXX(float g);

__device__ inline float XXXXX(float g);

using FuncT = UnaryFunctionFuse<decltype(X), decltype(XX), decltype(XXX),
                                decltype(XXXX), decltype(XXXXX)>;

__device__ FuncT fusion{X, XX, XXX, XXXX, XXXXX};

__global__ void KernelFused(float* data, size_t n,
                            std::function<__device__ float(float)>* func);
__global__ void KernelUnfused(float* data, size_t n, FuncT func);
