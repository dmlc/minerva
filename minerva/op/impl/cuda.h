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

}
#endif
}
