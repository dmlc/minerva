#pragma once
#include "op/context.h"
#include "op/physical_fn.h"
#include "op/closure.h"

namespace minerva {
namespace cuda {

void Arithmetic(DataList&, DataList&, ArithmeticClosure&, const CudaRuntimeContext&);
void MatMult(DataList&, DataList&, MatMultClosure&, const CudaRuntimeContext&);

}
}
