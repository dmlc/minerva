#pragma once
#include "op/context.h"
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
#ifdef HAS_CUDA
namespace cuda {

void Arithmetic(const DataList&, const DataList&, ArithmeticClosure&, const CudaRuntimeContext&);
void MatMult(const DataList&, const DataList&, MatMultClosure&, const CudaRuntimeContext&);
void ArithmeticConst(const DataList&, const DataList&, ArithmeticConstClosure&, const CudaRuntimeContext&);
void Transpose(const DataList&, const DataList&, TransposeClosure&, const CudaRuntimeContext&);
void NormArithmetic(const DataList&, const DataList&, NormArithmeticClosure&, const CudaRuntimeContext &);
void Reduction(const DataList&, const DataList&, ReductionClosure&, const CudaRuntimeContext&);
void MaxIndex(const DataList&, const DataList&, MaxIndexClosure&, const CudaRuntimeContext&);
void Reshape(const DataList&, const DataList&, ReshapeClosure&, const CudaRuntimeContext&);
void Elewise(const DataList&, const DataList&, ElewiseClosure&, const CudaRuntimeContext&);
void SigmoidForward(const DataList&, const DataList&, SigmoidForwardClosure&, const CudaRuntimeContext&);
void SigmoidBackward(const DataList&, const DataList&, SigmoidBackwardClosure&, const CudaRuntimeContext&);
void ReluForward(const DataList&, const DataList&, ReluForwardClosure&, const CudaRuntimeContext&);
void ReluBackward(const DataList&, const DataList&, ReluBackwardClosure&, const CudaRuntimeContext&);
void TanhForward(const DataList&, const DataList&, TanhForwardClosure&, const CudaRuntimeContext&);
void TanhBackward(const DataList&, const DataList&, TanhBackwardClosure&, const CudaRuntimeContext&);
void ConvForward(const DataList&, const DataList&, ConvForwardClosure&, const CudaRuntimeContext&);
void ConvBackwardData(const DataList&, const DataList&, ConvBackwardDataClosure&, const CudaRuntimeContext&);
void ConvBackwardFilter(const DataList&, const DataList&, ConvBackwardFilterClosure&, const CudaRuntimeContext&);
void ConvBackwardBias(const DataList&, const DataList&, ConvBackwardBiasClosure&, const CudaRuntimeContext&);
void SoftmaxForward(const DataList&, const DataList&, SoftmaxForwardClosure&, const CudaRuntimeContext&);
void SoftmaxBackward(const DataList&, const DataList&, SoftmaxBackwardClosure&, const CudaRuntimeContext&);
void ActivationForward(const DataList&, const DataList&, ActivationForwardClosure&, const CudaRuntimeContext&);
void ActivationBackward(const DataList&, const DataList&, ActivationBackwardClosure&, const CudaRuntimeContext&);
void PoolingForward(const DataList&, const DataList&, PoolingForwardClosure&, const CudaRuntimeContext&);
void PoolingBackward(const DataList&, const DataList&, PoolingBackwardClosure&, const CudaRuntimeContext&);
void SyncWithPS(const DataList& inputs, const DataList& outputs, SyncWithPSClosure& closure, const CudaRuntimeContext&);

void ArrayLoader(const DataList&, ArrayLoaderClosure& closure, const CudaRuntimeContext&);
void Randn(const DataList&, RandnClosure&, const CudaRuntimeContext&);
void RandBernoulli(const DataList&, RandBernoulliClosure&, const CudaRuntimeContext&);
void Fill(const DataList&, FillClosure&, const CudaRuntimeContext&);

void LRNForward(const DataList&, const DataList&, LRNForwardClosure&, const CudaRuntimeContext&);
void LRNBackward(const DataList&, const DataList&, LRNBackwardClosure&, const CudaRuntimeContext&);
void Concat(const DataList&, const DataList&, ConcatClosure&, const CudaRuntimeContext&);
void Slice(const DataList&, const DataList&, SliceClosure&, const CudaRuntimeContext&);

}
#endif
}
