#include "op/impl/cuda.h"
#include "op/impl/cuda/cuda_perform.h"
#include "op/context.h"
#include <glog/logging.h>

namespace minerva {
namespace cuda {

void Arithmetic(DataList& inputs, DataList& outputs, ArithmeticClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 2) << "Arithmetic takes 2 inputs";
  CHECK_EQ(outputs.size(), 1) << "Arithmetic takes 1 output";
  float* left = inputs[0].GetGpuData();
  float* right = inputs[1].GetGpuData();
  float* res = outputs[0].GetGpuData();
  size_t size = outputs[0].Size().Prod();
  switch (closure.type) {
    case ADD:
      CudaPerformArithmeticAdd(res, left, right, size, context.stream);
      break;
    case SUB:
      CudaPerformArithmeticSub(res, left, right, size, context.stream);
      break;
    case MULT:
      CudaPerformArithmeticMult(res, left, right, size, context.stream);
      break;
    case DIV:
      CudaPerformArithmeticDiv(res, left, right, size, context.stream);
      break;
  }
}

}
}
