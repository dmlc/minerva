#pragma once
#include "op/impl/basic.h"
#include "op/impl/cuda.h"
#include "op/impl/impl.h"
#include <dmlc/logging.h>

namespace minerva {

template<typename... Args>
void NO_IMPL(Args&&...) {
  LOG(FATAL) << "no implementation";
}

template<typename I, typename O, typename C, typename... Args>
void NO_IMPL(const I & i, const O & o, C & c, Args&&...) {
  LOG(FATAL) << "no implementation for " << typeid(C).name();
}

INSTALL_COMPUTE_FN(ArithmeticClosure, basic::Arithmetic, NO_IMPL, cuda::Arithmetic);
INSTALL_COMPUTE_FN(ArithmeticConstClosure, basic::ArithmeticConst, NO_IMPL, cuda::ArithmeticConst);
INSTALL_COMPUTE_FN(MatMultClosure, basic::MatMult, NO_IMPL, cuda::MatMult);
INSTALL_COMPUTE_FN(TransposeClosure, basic::Transpose, NO_IMPL, cuda::Transpose);
INSTALL_COMPUTE_FN(ReductionClosure, basic::Reduction, NO_IMPL, cuda::Reduction);
INSTALL_COMPUTE_FN(NormArithmeticClosure, basic::NormArithmetic, NO_IMPL, cuda::NormArithmetic);
INSTALL_COMPUTE_FN(MaxIndexClosure, basic::MaxIndex, NO_IMPL, cuda::MaxIndex);
INSTALL_COMPUTE_FN(ReshapeClosure, basic::Reshape, NO_IMPL, cuda::Reshape);
INSTALL_COMPUTE_FN(ElewiseClosure, basic::Elewise, NO_IMPL, cuda::Elewise);
INSTALL_COMPUTE_FN(SigmoidForwardClosure, basic::SigmoidForward, NO_IMPL, cuda::SigmoidForward);
INSTALL_COMPUTE_FN(SigmoidBackwardClosure, NO_IMPL, NO_IMPL, cuda::SigmoidBackward);
INSTALL_COMPUTE_FN(ReluForwardClosure, basic::ReluForward, NO_IMPL, cuda::ReluForward);
INSTALL_COMPUTE_FN(ThresholdNormClosure, basic::ThresholdNorm, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(ReluBackwardClosure, NO_IMPL, NO_IMPL, cuda::ReluBackward);
INSTALL_COMPUTE_FN(TanhForwardClosure, basic::TanhForward, NO_IMPL, cuda::TanhForward);
INSTALL_COMPUTE_FN(TanhBackwardClosure, NO_IMPL, NO_IMPL, cuda::TanhBackward);
INSTALL_COMPUTE_FN(ConvForwardClosure, NO_IMPL, NO_IMPL, cuda::ConvForward);
INSTALL_COMPUTE_FN(ConvBackwardDataClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardData);
INSTALL_COMPUTE_FN(ConvBackwardFilterClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardFilter);
INSTALL_COMPUTE_FN(ConvBackwardBiasClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardBias);
INSTALL_COMPUTE_FN(ConvForwardFindAlgorithmClosure, NO_IMPL, NO_IMPL, cuda::ConvForwardFindAlgorithm);
INSTALL_COMPUTE_FN(ConvBackwardFilterFindAlgorithmClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardFilterFindAlgorithm);
INSTALL_COMPUTE_FN(ConvBackwardDataFindAlgorithmClosure, NO_IMPL, NO_IMPL, cuda::ConvBackwardDataFindAlgorithm);
INSTALL_COMPUTE_FN(SoftmaxForwardClosure, basic::SoftmaxForward, NO_IMPL, cuda::SoftmaxForward);
INSTALL_COMPUTE_FN(SoftmaxBackwardClosure, NO_IMPL, NO_IMPL, cuda::SoftmaxBackward);
INSTALL_COMPUTE_FN(ActivationForwardClosure, basic::ActivationForward, NO_IMPL, cuda::ActivationForward);
INSTALL_COMPUTE_FN(ActivationBackwardClosure, NO_IMPL, NO_IMPL, cuda::ActivationBackward);
INSTALL_COMPUTE_FN(PoolingForwardClosure, NO_IMPL, NO_IMPL, cuda::PoolingForward);
INSTALL_COMPUTE_FN(PoolingBackwardClosure, NO_IMPL, NO_IMPL, cuda::PoolingBackward);
INSTALL_COMPUTE_FN(SyncWithPSClosure, basic::SyncWithPS, NO_IMPL, cuda::SyncWithPS);

INSTALL_DATAGEN_FN(ArrayLoaderClosure, basic::ArrayLoader, NO_IMPL, cuda::ArrayLoader);
INSTALL_DATAGEN_FN(RandnClosure, basic::Randn, NO_IMPL, cuda::Randn);
INSTALL_DATAGEN_FN(RandBernoulliClosure, basic::RandBernoulli, NO_IMPL, cuda::RandBernoulli);
INSTALL_DATAGEN_FN(FillClosure, basic::Fill, NO_IMPL, cuda::Fill);
INSTALL_COMPUTE_FN(LRNForwardClosure, NO_IMPL, NO_IMPL, cuda::LRNForward);
INSTALL_COMPUTE_FN(LRNBackwardClosure, NO_IMPL, NO_IMPL, cuda::LRNBackward);
INSTALL_COMPUTE_FN(ConcatClosure, NO_IMPL, NO_IMPL, cuda::Concat);
INSTALL_COMPUTE_FN(SliceClosure, NO_IMPL, NO_IMPL, cuda::Slice);
INSTALL_COMPUTE_FN(IndexClosure, basic::Index, NO_IMPL, NO_IMPL);
INSTALL_COMPUTE_FN(SelectClosure, NO_IMPL, NO_IMPL, cuda::Select);
}  // namespace minerva
