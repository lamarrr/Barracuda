/**
 * @file rng.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-28
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BARRACUDA_RNG_RNG_CUH_
#define BARRACUDA_RNG_RNG_CUH_

#include <curand.h>

#include "barracuda/utils/predication.cuh"
namespace bcuda {
namespace pseudo_rng {

class Uniform {
 public:
  Uniform();

  template <typename T>
  void GenerateSequence(T *fp_data, size_t n) {
    static_assert(std::is_floating_point<T>::value,
                  "Random Uniform pseudo-generator only exists for floating "
                  "point values");
    return GenerateSequence_(fp_data, n);
  }

  ~Uniform();

  void Seed(size_t seed);

 private:
  curandGenerator_t generator_;

  void GenerateSequence_(float *data, size_t n);
  void GenerateSequence_(double *data, size_t n);
};
};  // namespace pseudo_rng
}  // namespace bcuda
#endif  // BARRACUDA_RNG_RNG_CUH_
