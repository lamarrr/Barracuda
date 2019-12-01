/**
 * @file conv.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BARRACUDA_OPS_CONV_CUH_
#define BARRACUDA_OPS_CONV_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <memory>
#include <type_traits>
#include <utility>

#include "barracuda/tensor.cuh"
#include "barracuda/utils/memory.cuh"

namespace bcuda {
enum class PaddingType { Valid, Same };

struct Conv2dParams {
  constexpr Conv2dParams(int filter_height, int filter_width, int stride_height,
                         int stride_width,
                         // int padding_height,
                         //  int padding_width,
                         int dilation_rate_height = 1,
                         int dilation_rate_width = 1)
      : filter_height_{filter_height},
        filter_width_{filter_width},
        stride_height_{stride_height},
        stride_width_{stride_width},
        /*
        padding_height_{padding_height}, padding_width_{padding_width},
        */
        dilation_rate_height_{dilation_rate_height},
        dilation_rate_width_{dilation_rate_width} {};
  /*
    constexpr Conv2dParams(int filter_height, int filter_width, int
    stride_height, int stride_width, PaddingType padding_height_type,
                           PaddingType padding_width_type,
                           int dilation_rate_height = 1,
                           int dilation_rate_width = 1)
        : filter_height_{filter_height}, filter_width_{filter_width},
          stride_height_{stride_height}, stride_width_{stride_width},

          padding_height_{0}, padding_width_{0},
          dilation_rate_height_{dilation_rate_height}, dilation_rate_width_{
                                                           dilation_rate_width}
    {
      // perform calculation
      padding_height_ = -1;
      padding_width_ = -1;
    };
  */

  constexpr int filter_height() const noexcept { return filter_height_; }
  constexpr int filter_width() const noexcept { return filter_width_; }
  constexpr int stride_height() const noexcept { return stride_height_; }
  constexpr int stride_width() const noexcept { return stride_width_; }
  /*
  constexpr int padding_height() const noexcept { return padding_height_; };
  constexpr int padding_width() const noexcept { return padding_width_; };
  */
  constexpr int dilation_rate_height() const noexcept {
    return dilation_rate_height_;
  }
  constexpr int dilation_rate_width() const noexcept {
    return dilation_rate_width_;
  }

 private:
  int filter_height_;
  int filter_width_;
  int stride_height_;
  int stride_width_;
  /*
  int padding_width_;
  int padding_height_;
  */
  int dilation_rate_height_;
  int dilation_rate_width_;
};

// HW
// VALID only padding, SAME to be supported later on
constexpr inline int DilatedFilterDimension(int filter_dimension_size,
                                            int dilation_rate) noexcept {
  return (filter_dimension_size - 1) * dilation_rate + 1;
}

constexpr inline int DilatedMemoryIndex(int i, int j,
                                        Conv2dParams const &params) noexcept {
  return (params.dilation_rate_height() * j) *
             DilatedFilterDimension(params.filter_width(),
                                    params.dilation_rate_width()) +
         (i * params.dilation_rate_width());
}

/*
// CT_TEST
constexpr void fn() {
  constexpr Conv2dParams params = Conv2dParams{5, 5, 3, 3, 3, 2};
  constexpr int index = DilatedMemoryIndex(1, 1, params);
  constexpr int xf = DilatedFilterDimension(params.filter_width(),
                                            params.dilation_rate_width());
}
*/

template <typename T>
Tensor<T> Dilate2dFilter(Conv2dParams const &params, T const *const filter,
                         int *new_filter_height, int *new_filter_width) {
  auto new_height = DilatedFilterDimension(params.filter_height(),
                                           params.dilation_rate_height());
  auto new_width = DilatedFilterDimension(params.filter_width(),
                                          params.dilation_rate_width());
  Tensor<T> dilated_filter = Tensor<T>::Zeros({new_height, new_width});
  BCUDA_ENSURE_FALSE(dilated_filter.Data() == nullptr);

  // TODO(lamarrr): turn to parallel copy
  for (int j = 0; j < params.filter_height(); j++) {
    for (int i = 0; i < params.filter_width(); i++) {
      dilated_filter.Data()[DilatedMemoryIndex(i, j, params)] =
          filter[j * params.filter_width() + i];
    }
  }

  *new_filter_width = new_width;
  *new_filter_height = new_height;
  return std::move(dilated_filter);
}

}  // namespace bcuda

enum class ActivationType {
  None,
  ReLU,
  ReLU6,
  Sigmoid,
  HardSigmoid,
  Swish,
  HardSwish,
};

struct Conv2DParams {
  size_t x_stride;
  size_t y_stride;
  size_t filter_width;
  size_t filter_height;
  size_t dilation_rate;
  ActivationType activation_type;

  Conv2DParams(size_t x_stride, size_t y_stride, size_t filter_width,
               size_t filter_height, size_t dilation_rate = 1)
      : x_stride{x_stride},
        y_stride{y_stride},
        filter_width{filter_width},
        filter_height{filter_height},
        dilation_rate{dilation_rate} {}
};

// HW format
__global__ inline void Conv2D(size_t ndims, size_t *dimensions,
                              Conv2DParams *params, float *in, float *out) {
  for (size_t j = 0; j < params->filter_height; j += params->y_stride) {
    for (size_t i = 0; i < params->filter_width; i += params->x_stride) {
    }
  }

  //
  //
}

#endif  // BARRACUDA_OPS_CONV_CUH_
