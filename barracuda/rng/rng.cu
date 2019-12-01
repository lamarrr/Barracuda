/**
 * @file rng.cu
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-28
 *
 * @copyright Copyright (c) 2019
 *
 */

#include "barracuda/rng/rng.cuh"

namespace bcuda {
namespace pseudo_rng {

Uniform::Uniform() : generator_{nullptr} {
  BCUDA_ENSURE_TRUE_STR(
      curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT) ==
          curandStatus_t::CURAND_STATUS_SUCCESS,
      "Unable to create Pseudo Random Number Generator");
}

Uniform::~Uniform() noexcept {
  BCUDA_ENSURE_TRUE_STR(curandDestroyGenerator(generator_) ==
                            curandStatus_t::CURAND_STATUS_SUCCESS,
                        "Error destroying pseudo-random number generator");
  generator_ = nullptr;
}

void Uniform::Seed(size_t seed) {
  BCUDA_ENSURE_TRUE_STR(curandSetPseudoRandomGeneratorSeed(generator_, seed) ==
                            curandStatus_t::CURAND_STATUS_SUCCESS,
                        "Unable to set pseudo-random number generator seed");
}

void Uniform::GenerateSequence_(float *data, size_t n) {
  BCUDA_ENSURE_TRUE_STR(curandGenerateUniform(generator_, data, n) ==
                            curandStatus_t::CURAND_STATUS_SUCCESS,
                        "Error generating Pseudo-random sequence for tensor");
}

void Uniform::GenerateSequence_(double *data, size_t n) {
  BCUDA_ENSURE_TRUE_STR(curandGenerateUniformDouble(generator_, data, n) ==
                            curandStatus_t::CURAND_STATUS_SUCCESS,
                        "Error generating Pseudo-random sequence for tensor");
}

}  // namespace  pseudo_rng

}  // namespace bcuda
