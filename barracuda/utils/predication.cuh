/**
 * @file predication.cuh
 * @author Basit Ayantunde (rlamarrr@gmail.com)
 * @brief
 * @version 0.1.0
 * @date 2019-11-26
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BARRACUDA_UTILS_PREDICATION_CUH_
#define BARRACUDA_UTILS_PREDICATION_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define BCUDA_ENSURE_OK(statement)                                            \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            cudaError_t>::value,                                              \
        "BCUDA_ENSURE_OK must be called on statements evaluating "            \
        "to a cudaError_t type");                                             \
    if (baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ !=        \
        cudaError_t::cudaSuccess) {                                           \
      std::cout                                                               \
          << "[" << __FILE__ << ":" << __LINE__                               \
          << "] Error at Statement: " << #statement << "\nError Code: "       \
          << static_cast<int>(                                                \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << "\nError: "                                                      \
          << cudaGetErrorString(                                              \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << std::endl;                                                       \
      std::exit(-1);                                                          \
    }                                                                         \
  }

#define BCUDA_ENSURE_TRUE(statement)                                          \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            bool>::value,                                                     \
        "BCUDA_ENSURE_TRUE must be called on statements evaluating "          \
        "to a bool type");                                                    \
    if (!baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) {       \
      std::cout << "[" << __FILE__ << ":" << __LINE__                         \
                << "] Error at Statement: " << #statement                     \
                << "\nEvaluated To False" << std::endl;                       \
      std::exit(-1);                                                          \
    }                                                                         \
  }

#define BCUDA_ENSURE_TRUE_STR(statement, error_msg)                           \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            bool>::value,                                                     \
        "BCUDA_ENSURE_TRUE must be called on statements evaluating "          \
        "to a bool type");                                                    \
    if (!baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) {       \
      std::cout << "[" << __FILE__ << ":" << __LINE__                         \
                << "] Error at Statement: " << #statement                     \
                << "\nEvaluated To False"                                     \
                << "\nMessage: " << error_msg << std::endl;                   \
      std::exit(-1);                                                          \
    }                                                                         \
  }

#define BCUDA_ENSURE_FALSE(statement)                                         \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            bool>::value,                                                     \
        "BCUDA_ENSURE_FALSE must be called on statements evaluating "         \
        "to a bool type");                                                    \
    if (baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) {        \
      std::cout << "[" << __FILE__ << ":" << __LINE__                         \
                << "] Error at Statement: " << #statement                     \
                << "\nEvaluated To True" << std::endl;                        \
      std::exit(-1);                                                          \
    }                                                                         \
  }

#define BCUDA_ENSURE_OK_STR(statement, error_msg)                             \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            cudaError_t>::value,                                              \
        "BCUDA_ENSURE_OK_STR must be called on statements "                   \
        "evaluating to a cudaError_t type");                                  \
    if (baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ !=        \
        cudaError_t::cudaSuccess) {                                           \
      std::cout                                                               \
          << "[" << __FILE__ << ":" << __LINE__                               \
          << "] Error at Statement: " << #statement << "\nError Code: "       \
          << static_cast<int>(                                                \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << "\nMessage: " << error_msg << std::endl;                         \
      std::exit(-1);                                                          \
    }                                                                         \
  }

#define BCUDA_CHECK_OK(statement)                                             \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            cudaError_t>::value,                                              \
        "BCUDA_CHECK_OK must be called on statements evaluating to "          \
        "a cudaError_t type");                                                \
    if (baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ !=        \
        cudaError::cudaSuccess) {                                             \
      std::cout                                                               \
          << "Error at Statement: " << #statement << "\nError Code: "         \
          << static_cast<int>(                                                \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << "\nError: "                                                      \
          << cudaGetErrorString(                                              \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << std::endl;                                                       \
      return -1;                                                              \
    }                                                                         \
  }

#define BCUDA_CHECK_OK_STR(statement, error_msg)                              \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            cudaError_t>::value,                                              \
        "BCUDA_CHECK_OK must be called on statements evaluating to "          \
        "a cudaError_t type");                                                \
    if (baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ !=        \
        cudaError::cudaSuccess) {                                             \
      std::cout                                                               \
          << "Error at Statement: " << #statement << "\nError Code: "         \
          << static_cast<int>(                                                \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << "\nError: "                                                      \
          << cudaGetErrorString(                                              \
                 baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) \
          << "\nMessage: " << error_msg << std::endl;                         \
      return -1;                                                              \
    }                                                                         \
  }

#define BCUDA_CHECK_TRUE_STR(statement, error_msg)                            \
  {                                                                           \
    auto baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_ =        \
        (statement);                                                          \
    static_assert(                                                            \
        std::is_same<                                                         \
            decltype(                                                         \
                baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_), \
            bool>::value,                                                     \
        "BCUDA_ENSURE_TRUE must be called on statements evaluating "          \
        "to a bool type");                                                    \
    if (!baRraCcuDaERRorCode_PLAceHoLder_WhichYouHopefullyDontAlias_) {       \
      std::cout << "[" << __FILE__ << ":" << __LINE__                         \
                << "] Error at Statement: " << #statement                     \
                << "\nEvaluated To False"                                     \
                << "\nMessage: " << error_msg << std::endl;                   \
      return -1;                                                              \
    }                                                                         \
  }

#endif  // BARRACUDA_CUDA_UTILS_PREDICATION_CUH_
