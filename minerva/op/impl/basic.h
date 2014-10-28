#pragma once
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
namespace basic {

extern void Arithmetic(const DataList&, const DataList&, ArithmeticClosure&);
extern void ArithmeticConst(const DataList&, const DataList&, ArithmeticConstClosure&);
extern void Elewise(const DataList&, const DataList&, ElewiseClosure&);
extern void MatMult(const DataList&, const DataList&, MatMultClosure&);
extern void Transpose(const DataList&, const DataList&, TransposeClosure&);
extern void Reduction(const DataList&, const DataList&, ReductionClosure&);
extern void NormArithmetic(const DataList&, const DataList&, NormArithmeticClosure&);
extern void MaxIndex(const DataList&, const DataList&, MaxIndexClosure&);
extern void Reshape(const DataList&, const DataList&, ReshapeClosure&);

extern void Randn(const DataList&, RandnClosure& );
extern void Fill(const DataList&, FillClosure& );

}  // end of namespace basic
}  // end of namespace minerva
