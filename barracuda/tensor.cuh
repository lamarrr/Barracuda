/**
 * @file tensor.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BARRACUDA_TENSOR_CUH_
#define BARRACUDA_TENSOR_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cinttypes>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "barracuda/execution_policy.cuh"
#include "barracuda/kernels/kernels.cuh"
#include "barracuda/rng/rng.cuh"
#include "barracuda/utils/memory.cuh"
#include "barracuda/utils/predication.cuh"

// TODO(lamarrr): move this file to a source file and link to high level ops
namespace bcuda {
// FUTURE
// using fp16 = __half2;

template <typename T, bool is_floating = false>
struct DefaultDistributionHelper_ {
  using distribution_type = std::uniform_int_distribution<T>;
};

template <typename T>
struct DefaultDistributionHelper_<T, true> {
  using distribution_type = std::uniform_real_distribution<T>;
};

template <typename T>
struct DefaultDistribution_ {
  using distribution_type = typename DefaultDistributionHelper_<
      T, std::is_floating_point<T>::value>::distribution_type;
};

class Index {
 public:
  using container_type = std::vector<size_t>;

  Index(container_type const &container) : container_{container} {}  // NOLINT
  Index(std::initializer_list<size_t> const &container)
      : container_{container} {}  // NOLINT
  Index(Index const &) = default;
  Index(Index &&) = default;
  Index &operator=(Index const &) = default;
  Index &operator=(Index &&) = default;
  ~Index() noexcept = default;

  size_t NumDimensions() const noexcept { return container_.size(); }

  size_t operator[](size_t index) const noexcept { return container_[index]; }
  size_t &operator[](size_t index) noexcept { return container_[index]; }

  size_t At(size_t index) const { return container_.at(index); }
  size_t &At(size_t index) { return container_.at(index); }

  container_type &Container() { return container_; }

  container_type const &Container() const { return container_; }

  explicit operator std::string() const {
    std::string s = "Index: [ ";
    std::for_each(
        container_.begin(), container_.end(),
        [&s](size_t dim) -> void { s.append(std::to_string(dim) + " "); });
    s += "]";
    return std::move(s);
  }

 private:
  container_type container_;
};

class Shape {
 public:
  using container_type = std::vector<size_t>;

  Shape(std::initializer_list<size_t> const &dims) : container_{dims} {}
  Shape() : container_{} {}
  Shape(Shape const &shape) = default;
  Shape(Shape &&shape) = default;
  Shape &operator=(Shape const &shape) = default;
  Shape &operator=(Shape &&shape) = default;

  Shape(container_type const &dims) : container_{dims} {}  // NOLINT

  size_t NumDimensions() const noexcept { return container_.size(); }

  bool operator==(Shape const &other) const noexcept {
    return (other.TensorSize() == TensorSize()) &&
           (NumDimensions() == other.NumDimensions()) &&
           std::equal(other.container_.begin(), other.container_.end(),
                      container_.begin());
  }
  size_t At(size_t index) const { return container_.at(index); }

  size_t TensorSize() const noexcept {
    return std::accumulate(container_.begin(), container_.end(), 1,
                           [](size_t a, size_t b) { return a * b; });
  }

  size_t operator[](size_t index) const noexcept { return container_[index]; }

  size_t const *Data() const noexcept { return container_.data(); }

  container_type const &Container() const noexcept { return container_; }

  explicit operator std::string() const {
    std::string s = "Shape: [ ";
    std::for_each(
        container_.begin(), container_.end(),
        [&s](size_t dim) -> void { s.append(std::to_string(dim) + " "); });
    s += "]";
    return std::move(s);
  }

  bool IsMatchingIndex(Index const &index) const {
    return (index.NumDimensions() == NumDimensions()) && IsInRange_(index);
  }

 private:
  container_type container_;

  bool IsInRange_(Index const &index) const {
    auto ends = std::mismatch(
        index.Container().begin(), index.Container().end(), Container().begin(),
        [&](size_t x, size_t y) -> bool { return x < y; });
    return (ends.first == index.Container().end()) &&
           (ends.second == Container().end());
  }
};

/**
 * @brief Computes the position of an element of a tensor from its shape
 * NOTE: Assumes you know what you're doing and already performed bound checking
 *
 * @param shape
 * @param indices
 * @return size_t
 */
inline size_t LinearPosition(Shape const &shape, Index const &index) noexcept {
  auto index_accum = 0UL;
  auto size = shape.TensorSize();

  for (size_t i = 0; i < index.NumDimensions(); i++) {
    size /= shape[i];
    index_accum += size * index[i];
  }

  return index_accum;
}

// Dense GPUTensor
template <typename ElementTp>
class Tensor {
 public:
  using ElementType = ElementTp;
  using Iterator = ElementType *;
  using ReverseIterator = std::reverse_iterator<Iterator>;

  using ConstIterator = ElementType const *;
  using ReverseConstIterator = std::reverse_iterator<ConstIterator>;

  using Pointer = Iterator;

  using ShapeType = ::bcuda::Shape;
  using IndexType = Index;

  Tensor() = delete;
  explicit Tensor(ShapeType const &shape) : shape_{shape} {
    size_t size = shape_.TensorSize();
    data_ = MakeCudaArray<ElementType>(size);
  }

  explicit Tensor(ShapeType const &shape, ElementType value) : shape_{shape} {
    size_t size = shape_.TensorSize();
    data_ = MakeCudaArray<ElementType>(size);
    gpu_kernels::LinearFill<<<1, 1>>>(Size(), Data(), value);
    BCUDA_ENSURE_OK_STR(cudaDeviceSynchronize(), "Unable to Fill Tensor");
  }

  Tensor(Tensor const &src) : shape_{src.Shape()} {
    AllocateMemory_();
    BCUDA_ENSURE_OK_STR(cudaMemcpy(data_.get(), src.data_.get(), src.Size(),
                                   cudaMemcpyDeviceToDevice),
                        "Error Copying Tensor Data from Device to Device");
  }
  Tensor(Tensor &&) = default;
  Tensor &operator=(Tensor const &src) {
    Tensor tmp{src};
    std::swap(tmp, *this);
    return *this;
  }

  Tensor &operator=(Tensor &&) = default;
  ~Tensor() noexcept = default;

  static Tensor Zeros(ShapeType const &shape) noexcept {
    Tensor tensor = Tensor(shape);
    BCUDA_ENSURE_OK_STR(
        cudaMemset(tensor.data_.get(), 0, tensor.Size() * sizeof(ElementType)),
        "Tensor Memset Failed");
    BCUDA_ENSURE_OK_STR(cudaDeviceSynchronize(),
                        "Device Synchronize Failed After Memset");
    return std::move(tensor);
  }

  // change name? Uniform?Shape?Dist?
  template <typename EngineT = std::default_random_engine>
  static Tensor RandomUniformUnison(ShapeType const &shape, ElementType begin,
                                    ElementType end,
                                    size_t seed = 1UL) noexcept {
    using dist_t =
        typename DefaultDistribution_<ElementType>::distribution_type;

    dist_t dist{begin, end};
    auto engine = EngineT{};
    engine.seed(seed);
    ElementType val = dist(engine);
    Tensor tensor = Tensor(shape);

    gpu_kernels::LinearFill<<<1, 1>>>(tensor.Size(), tensor.Data(), val);
    BCUDA_ENSURE_OK_STR(cudaDeviceSynchronize(), "Unable to Fill Tensor");

    return std::move(tensor);
  }

  /**
   * @brief:  Generate random numbers between 0.0 and 1.0 along a uniform
   * distribution
   *
   * @param shape: Output tensor shape
   * @param seed: seed for random number generator
   * @return Tensor
   */
  static Tensor RandomUniform(ShapeType const &shape,
                              size_t seed = 0) noexcept {
    pseudo_rng::Uniform rng;

    Tensor tensor = Tensor(shape);

    rng.GenerateSequence(tensor.Data(), tensor.Size());

    return std::move(tensor);
  }

  /**
   * @brief Create a tensor filled with ones of specified shape
   *
   * @param shape: Shape of output tensor
   * @return Tensor
   */
  static Tensor Ones(ShapeType const &shape) noexcept {
    Tensor tensor = Tensor(shape);
    // TODO(lamarrr): set execution policy
    gpu_kernels::LinearFill<<<1, 1>>>(tensor.Size(), tensor.Data(),
                                      static_cast<ElementType>(1));
    BCUDA_ENSURE_OK_STR(cudaDeviceSynchronize(), "Unable to Fill Tensor");

    return std::move(tensor);
  }

  ShapeType const &Shape() const noexcept { return shape_; }

  /**
   * @brief  gets total number of elements in the tensor
   *
   * @return size_t
   */
  size_t Size() const noexcept { return shape_.TensorSize(); }

  ElementType *Data() noexcept { return data_.get(); }
  ElementType const *Data() const noexcept { return data_.get(); }

  ConstIterator begin() const noexcept { return Data(); }
  ConstIterator end() const noexcept { return Data() + Size(); }
  Iterator begin() noexcept { return Data(); }
  Iterator end() noexcept { return Data() + Size(); }

  ElementType &operator[](size_t index) noexcept { return data_[index]; }
  ElementType operator[](size_t index) const noexcept { return data_[index]; }

  ElementType &At(size_t index) {
    if (index >= Size()) throw std::out_of_range{"Tensor index out of range"};
    return data_[index];
  }
  ElementType At(size_t index) const {
    if (index >= Size()) throw std::out_of_range{"Tensor index out of range"};
    return data_[index];
  }

  ElementType &operator[](IndexType index) noexcept {
    // BCUDA_ENSURE_TRUE_STR(index.NumDimensions() != shape.NumDimensions(),
    //                     "Tensor index size must be equal to Tensor shape
    //                     size");
  }

  ElementType operator[](IndexType index) const noexcept {}

  ElementType &At(IndexType index) {
    if (index >= Size()) throw std::out_of_range{"Tensor index out of range"};
    return data_[index];
  }

  ElementType At(IndexType index) const {
    if (index >= Size()) throw std::out_of_range{"Tensor index out of range"};
    return data_[index];
  }

#if false
  explicit operator std::string(){}
#endif

 private:
  CudaArray<ElementType> data_;
  ::bcuda::Shape shape_;

  void AllocateMemory_() {
    data_ = MakeCudaArray<ElementType>(shape_.TensorSize());
  }
};

}  // namespace bcuda
#endif  // BARRACUDA_TENSOR_CUH_
