#pragma once

#include "op/closure.h"
#include "op/physical.h"

namespace minerva {
namespace basic {

extern void Arithmetic(DataList&, DataList&, ArithmeticClosure& );
extern void ArithmeticConst(DataList&, DataList&, ArithmeticConstClosure& );
extern void Elewise(DataList&, DataList&, ElewiseClosure& );
extern void MatMult(DataList&, DataList&, MatMultClosure& );
extern void Transpose(DataList&, DataList&, TransposeClosure& );
extern void Reduction(DataList&, DataList&, ReductionClosure& );

extern void Randn(DataList&, RandnClosure& );
extern void Fill(DataList&, FillClosure& );

extern void Assemble(NVector<DataShard>&, float*, const Scale&);

extern void NCopy(
    float* src, const Scale& srcsize, const Scale& srcstart,
    float* dst, const Scale& dstsize, const Scale& dststart,
    const Scale& copysize);

} // end of namespace basic
} // end of namespace minerva
