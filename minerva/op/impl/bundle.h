#pragma once
#include "basic.h"
#include "op/impl/cuda.h"
#include "impl.h"
#include <glog/logging.h>
#include <iostream>
#include <map>

namespace minerva {

template<typename... Args>
void NO_IMPL(Args&&...) {
  CHECK(false) << "no implementation";
}

INSTALL_COMPUTE_FN(ArithmeticClosure, basic::Arithmetic, NO_IMPL, cuda::Arithmetic);
INSTALL_COMPUTE_FN(ArithmeticConstClosure, basic::ArithmeticConst, NO_IMPL, cuda::ArithmeticConst);
INSTALL_COMPUTE_FN(ElewiseClosure, basic::Elewise, NO_IMPL, cuda::Elewise);
INSTALL_COMPUTE_FN(MatMultClosure, basic::MatMult, NO_IMPL, cuda::MatMult);
INSTALL_COMPUTE_FN(TransposeClosure, basic::Transpose, NO_IMPL, cuda::Transpose);
INSTALL_COMPUTE_FN(ReductionClosure, basic::Reduction, NO_IMPL, cuda::Reduction);
INSTALL_COMPUTE_FN(NormArithmeticClosure, basic::NormArithmetic, NO_IMPL, cuda::NormArithmetic);
INSTALL_COMPUTE_FN(MaxIndexClosure, basic::MaxIndex, NO_IMPL, cuda::MaxIndex);
INSTALL_COMPUTE_FN(ReshapeClosure, basic::Reshape, NO_IMPL, cuda::Reshape);
INSTALL_COMPUTE_FN(ConvForwardClosure, NO_IMPL, NO_IMPL, cuda::ConvForward);
INSTALL_COMPUTE_FN(ConvBackwardDataClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardData);
INSTALL_COMPUTE_FN(ConvBackwardFilterClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardFilter);
INSTALL_COMPUTE_FN(ConvBackwardBiasClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardBias);
INSTALL_COMPUTE_FN(SoftmaxForwardClosure, NO_IMPL, NO_IMPL, cuda::SoftmaxForward);
INSTALL_COMPUTE_FN(SoftmaxBackwardClosure, NO_IMPL, NO_IMPL, cuda::SoftmaxBackward);
INSTALL_COMPUTE_FN(ActivationForwardClosure, NO_IMPL, NO_IMPL, cuda::ActivationForward);
INSTALL_COMPUTE_FN(ActivationBackwardClosure, NO_IMPL, NO_IMPL, cuda::ActivationBackward);
INSTALL_COMPUTE_FN(PoolingForwardClosure, NO_IMPL, NO_IMPL, cuda::PoolingForward);
INSTALL_COMPUTE_FN(PoolingBackwardClosure, NO_IMPL, NO_IMPL, cuda::PoolingBackward);

INSTALL_DATAGEN_FN(RandnClosure, basic::Randn, NO_IMPL, NO_IMPL);
INSTALL_DATAGEN_FN(FillClosure, basic::Fill, NO_IMPL, NO_IMPL);

}
