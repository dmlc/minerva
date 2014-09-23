#pragma once
#include "op/context.h"
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
namespace cuda {

void Arithmetic(DataList&, DataList&, ArithmeticClosure&, const CudaRuntimeContext&);
void MatMult(DataList&, DataList&, MatMultClosure&, const CudaRuntimeContext&);
void ArithmeticConst(DataList&, DataList&, ArithmeticConstClosure&, const CudaRuntimeContext&);
void Transpose(DataList&, DataList&, TransposeClosure&, const CudaRuntimeContext&);
void NormArithmetic(DataList&, DataList&, NormArithmeticClosure&, const CudaRuntimeContext &);
void Reduction(DataList&, DataList&, ReductionClosure&, const CudaRuntimeContext&);
void MaxIndex(DataList&, DataList&, MaxIndexClosure&, const CudaRuntimeContext&);
void Elewise(DataList&, DataList&, ElewiseClosure&, const CudaRuntimeContext&);

}
}
