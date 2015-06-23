#pragma once
#include "op/context.h"
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
#ifdef HAS_CUDA
namespace cuda {

void Arithmetic(const DataList&, const DataList&, ArithmeticClosure&, const Context&);
void MatMult(const DataList&, const DataList&, MatMultClosure&, const Context&);
void ArithmeticConst(const DataList&, const DataList&, ArithmeticConstClosure&, const Context&);
void Transpose(const DataList&, const DataList&, TransposeClosure&, const Context&);
void NormArithmetic(const DataList&, const DataList&, NormArithmeticClosure&, const Context &);
void Reduction(const DataList&, const DataList&, ReductionClosure&, const Context&);
void ReductionWithReshape(const DataList&, const DataList&, ReductionWithReshapeClosure&, const Context&);
void MaxIndex(const DataList&, const DataList&, MaxIndexClosure&, const Context&);
void Reshape(const DataList&, const DataList&, ReshapeClosure&, const Context&);
void Elewise(const DataList&, const DataList&, ElewiseClosure&, const Context&);
void SigmoidForward(const DataList&, const DataList&, SigmoidForwardClosure&, const Context&);
void SigmoidBackward(const DataList&, const DataList&, SigmoidBackwardClosure&, const Context&);
void ReluForward(const DataList&, const DataList&, ReluForwardClosure&, const Context&);
void ReluBackward(const DataList&, const DataList&, ReluBackwardClosure&, const Context&);
void TanhForward(const DataList&, const DataList&, TanhForwardClosure&, const Context&);
void TanhBackward(const DataList&, const DataList&, TanhBackwardClosure&, const Context&);
void ConvForward(const DataList&, const DataList&, ConvForwardClosure&, const Context&);
void ConvBackwardData(const DataList&, const DataList&, ConvBackwardDataClosure&, const Context&);
void ConvBackwardFilter(const DataList&, const DataList&, ConvBackwardFilterClosure&, const Context&);
void ConvBackwardBias(const DataList&, const DataList&, ConvBackwardBiasClosure&, const Context&);
void SoftmaxForward(const DataList&, const DataList&, SoftmaxForwardClosure&, const Context&);
void SoftmaxBackward(const DataList&, const DataList&, SoftmaxBackwardClosure&, const Context&);
void ActivationForward(const DataList&, const DataList&, ActivationForwardClosure&, const Context&);
void ActivationBackward(const DataList&, const DataList&, ActivationBackwardClosure&, const Context&);
void PoolingForward(const DataList&, const DataList&, PoolingForwardClosure&, const Context&);
void PoolingBackward(const DataList&, const DataList&, PoolingBackwardClosure&, const Context&);
void SyncWithPS(const DataList& inputs, const DataList& outputs, SyncWithPSClosure& closure, const Context&);

void ArrayLoader(const DataList&, ArrayLoaderClosure& closure, const Context&);
void Randn(const DataList&, RandnClosure&, const Context&);
void RandBernoulli(const DataList&, RandBernoulliClosure&, const Context&);
void Fill(const DataList&, FillClosure&, const Context&);

void LRNForward(const DataList&, const DataList&, LRNForwardClosure&, const Context&);
void LRNBackward(const DataList&, const DataList&, LRNBackwardClosure&, const Context&);
void Concat(const DataList&, const DataList&, ConcatClosure&, const Context&);
void Slice(const DataList&, const DataList&, SliceClosure&, const Context&);
void Index(const DataList&, const DataList&, IndexClosure&, const Context&);

void Select(DataList const&, DataList const&, SelectClosure&, Context const&);

}
#endif
}
