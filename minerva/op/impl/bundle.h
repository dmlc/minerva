#pragma once
#include "basic.h"
#include "impl.h"
#include <glog/logging.h>
#include <iostream>
#include <map>

namespace minerva {

template<typename... Args>
void NO_IMPL(Args&&...) {
  CHECK(false) << "no implementation";
}

INSTALL_COMPUTE_FN(ArithmeticClosure, basic::Arithmetic, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(ArithmeticConstClosure, basic::ArithmeticConst, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(ElewiseClosure, basic::Elewise, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(MatMultClosure, basic::MatMult, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(TransposeClosure, basic::Transpose, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(ReductionClosure, basic::Reduction, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(AssembleClosure, basic::Assemble, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(SplitClosure, basic::Split, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(NormArithmeticClosure, basic::NormArithmetic, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(MaxIndexClosure, basic::MaxIndex, NO_IMPL, NO_IMPL);

INSTALL_DATAGEN_FN(RandnClosure, basic::Randn, NO_IMPL, NO_IMPL);
INSTALL_DATAGEN_FN(FillClosure, basic::Fill, NO_IMPL, NO_IMPL);

}
