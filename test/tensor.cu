/**
 * @file tensor.cu
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-28
 *
 * @copyright Copyright (c) 2019
 *
 */

#include <gtest/gtest.h>

#include <algorithm>

#include "barracuda/kernels/activations.cuh"
#include "barracuda/tensor.cuh"

namespace kerns = bcuda::gpu_kernels;

TEST(Tensor, Size) {
  auto tensor_x = bcuda::Tensor<double>::RandomUniform({460, 200});

  ASSERT_EQ(tensor_x.Size(), 460 * 200);
}

TEST(Tensor, Random) {
  auto tensor_x = bcuda::Tensor<double>::RandomUniform({46, 2});

  for (auto i : tensor_x) {
    std::cout << i << ", ";
  }

  std::cout << std::endl;
}

TEST(TensorUtils, LinearPositon) {
  bcuda::Shape shape{6, 7};
  bcuda::Index index{4, 3};

  ASSERT_EQ(bcuda::LinearPosition(shape, index), 4 * 7 + 3);

  shape = {5};
  index = {4};

  ASSERT_EQ(bcuda::LinearPosition(shape, index), 4);

  shape = {6, 7, 2};
  index = {4, 3, 1};

  ASSERT_EQ(bcuda::LinearPosition(shape, index), 4 * 7 * 2 + 3 * 2 + 1);
}

TEST(TensorShape, Subscripting) {
  bcuda::Shape shape{6, 75, 7, 17, 42, 2, 1, 234};

  ASSERT_TRUE(shape.IsMatchingIndex({5, 74, 6, 16, 41, 1, 0, 233}));
  ASSERT_FALSE(shape.IsMatchingIndex({5, 74, 6, 17, 41, 1, 0, 233}));
  ASSERT_FALSE(shape.IsMatchingIndex({6, 74, 6, 16, 41, 1, 0, 233}));
  ASSERT_FALSE(shape.IsMatchingIndex({5, 74, 6, 17, 41, 1, 0, 234}));
}
