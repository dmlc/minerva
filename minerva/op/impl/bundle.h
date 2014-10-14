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
INSTALL_COMPUTE_FN(ConvForwardClosure, NO_IMPL, NO_IMPL, cuda::ConvForward);

INSTALL_DATAGEN_FN(RandnClosure, basic::Randn, NO_IMPL, NO_IMPL);
INSTALL_DATAGEN_FN(FillClosure, basic::Fill, NO_IMPL, NO_IMPL);

}
