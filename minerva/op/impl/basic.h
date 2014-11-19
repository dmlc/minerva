#pragma once
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
namespace basic {

void Arithmetic(const DataList&, const DataList&, ArithmeticClosure&);
void ArithmeticConst(const DataList&, const DataList&, ArithmeticConstClosure&);
void Elewise(const DataList&, const DataList&, ElewiseClosure&);
void MatMult(const DataList&, const DataList&, MatMultClosure&);
void Transpose(const DataList&, const DataList&, TransposeClosure&);
void Reduction(const DataList&, const DataList&, ReductionClosure&);
void NormArithmetic(const DataList&, const DataList&, NormArithmeticClosure&);
void MaxIndex(const DataList&, const DataList&, MaxIndexClosure&);
void Reshape(const DataList&, const DataList&, ReshapeClosure&);

void ArrayLoader(const DataList&, ArrayLoaderClosure&);
void Randn(const DataList&, RandnClosure&);
void RandBernoulli(const DataList&, RandBernoulliClosure&);
void Fill(const DataList&, FillClosure&);

}  // end of namespace basic
}  // end of namespace minerva
