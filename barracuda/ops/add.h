#include "barracuda/tensor.cuh"

namespace bcuda {
namespace ops {
template <typename TensorT>
inline TensorT Add(TensorT const& input_1, TensorT const& input_2);

template <typename TensorT>
inline TensorT ElementwiseMul(TensorT const& input_1, TensorT const& input_2);
}  // namespace ops

}  // namespace bcuda
