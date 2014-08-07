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

extern void Randn(DataList&, RandnClosure& );
extern void Fill(DataList&, FillClosure& );

extern void Assemble(DataList&, DataList&, AssembleClosure&);
extern void Split(DataList&, DataList&, SplitClosure&);

extern void NCopy(
    float* src, const Scale& srcsize, const Scale& srcstart,
    float* dst, const Scale& dstsize, const Scale& dststart,
    const Scale& copysize);

} // end of namespace basic
} // end of namespace minerva
