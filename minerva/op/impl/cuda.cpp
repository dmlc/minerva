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
#ifdef HAS_CUDA
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
#endif
}

void MatMult(DataList& inputs, DataList& outputs, MatMultClosure& closure, const CudaRuntimeContext & context) {
  CHECK_EQ(inputs.size(), 2) << "(matmult) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(matmult) #outputs is wrong!";
  float* left_data = inputs[0].GetGpuData();
  float* right_data = inputs[1].GetGpuData();
  float* res_data = outputs[0].GetGpuData();
  int m = outputs[0].Size()[0];
  int n = outputs[0].Size()[1];
  int o = inputs[0].Size()[1];
  // ATTENTION: the data is column major !!
  CudaPerformMatMult(left_data, right_data, res_data, m, n, o, context.handle, context.zero, context.one);
}

}
}
