#include "op/impl/cuda.h"
#include "op/impl/cuda/cuda_perform.h"
#include "op/context.h"
#include <glog/logging.h>

namespace minerva {
namespace cuda {

void Arithmetic(DataList& inputs, DataList& outputs, ArithmeticClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 2) << "Arithmetic takes 2 inputs";
  CHECK_EQ(outputs.size(), 1) << "Arithmetic takes 1 output";
  float* left = inputs[0].data();
  float* right = inputs[1].data();
  float* res = outputs[0].data();
  size_t size = outputs[0].size().Prod();
#ifdef HAS_CUDA
  switch (closure.type) {
    case ArithmeticType::kAdd:
      CudaPerformArithmeticAdd(res, left, right, size, context.stream);
      break;
    case ArithmeticType::kSub:
      CudaPerformArithmeticSub(res, left, right, size, context.stream);
      break;
    case ArithmeticType::kMult:
      CudaPerformArithmeticMult(res, left, right, size, context.stream);
      break;
    case ArithmeticType::kDiv:
      CudaPerformArithmeticDiv(res, left, right, size, context.stream);
      break;
  }
#endif
}

}
}
