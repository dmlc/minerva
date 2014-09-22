#pragma once
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
namespace basic {

extern void Arithmetic(DataList&, DataList&, ArithmeticClosure& );
extern void ArithmeticConst(DataList&, DataList&, ArithmeticConstClosure& );
extern void Elewise(DataList&, DataList&, ElewiseClosure& );
extern void MatMult(DataList&, DataList&, MatMultClosure& );
extern void Transpose(DataList&, DataList&, TransposeClosure& );
extern void Reduction(DataList&, DataList&, ReductionClosure& );
extern void NormArithmetic(DataList&, DataList&, NormArithmeticClosure&);
extern void MaxIndex(DataList&, DataList&, MaxIndexClosure&);

extern void Randn(DataList&, RandnClosure& );
extern void Fill(DataList&, FillClosure& );

} // end of namespace basic
} // end of namespace minerva
