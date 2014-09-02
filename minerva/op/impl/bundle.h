#pragma once
#include <iostream>
#include <map>
#include "basic.h"
#include "impl.h"

namespace minerva {

template<class A1, class A2>
void NO_IMPL(A1&, A2&) {
  std::cout << "No implementation" << std::endl;
}

template<class A1, class A2, class A3>
void NO_IMPL(A1&, A2&, A3&) {
  std::cout << "No implementation" << std::endl;
}

template<class A1, class A2, class A3, class A4>
void NO_IMPL(A1&, A2&, A3&, const A4&) {
  std::cout << "No implementation" << std::endl;
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
