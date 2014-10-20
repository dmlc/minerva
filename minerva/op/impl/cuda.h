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
void Elewise(const DataList&, const DataList&, ElewiseClosure&, const CudaRuntimeContext&);
void ConvForward(const DataList&, const DataList&, ConvForwardClosure&, const CudaRuntimeContext&);
void ConvBackwardData(const DataList&, const DataList&, ConvBackwardDataClosure&, const CudaRuntimeContext&);
void ConvBackwardFilter(const DataList&, const DataList&, ConvBackwardFilterClosure&, const CudaRuntimeContext&);
void ConvBackwardBias(const DataList&, const DataList&, ConvBackwardBiasClosure&, const CudaRuntimeContext&);
void SoftmaxForward(const DataList&, const DataList&, SoftmaxForwardClosure&, const CudaRuntimeContext&);
void SoftmaxBackward(const DataList&, const DataList&, SoftmaxBackwardClosure&, const CudaRuntimeContext&);
void ActivationForward(const DataList&, const DataList&, ActivationForwardClosure&, const CudaRuntimeContext&);
void ActivationBackward(const DataList&, const DataList&, ActivationBackwardClosure&, const CudaRuntimeContext&);

}
#endif
}
