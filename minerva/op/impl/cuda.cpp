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
      int m = inputs[0].Size()[0];
      int n = inputs[0].Size()[1];
#ifdef HAS_CUDA
      switch (closure.type) {
      case ADD:
        //CudaPerformArithmeticAdd(res, left, right, size, context.stream);
	abort();
        break;
      case SUB:
        CudaPerformSub(left, right, res, m, n, context.handle);
        break;
      case MULT:
	abort();
        //CudaPerformArithmeticMult(res, left, right, size, context.stream);
        break;
      case DIV:
	abort();
        //CudaPerformArithmeticDiv(res, left, right, size, context.stream);
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
      int m = inputs[0].Size()[0];
      int k = inputs[0].Size()[1];
      int n = outputs[0].Size()[1];
      // ATTENTION: the data is column major !!
      CudaPerformMatMult(left_data, right_data, res_data, m, n, k, context.handle);
    }

    void ArithmeticConst(DataList& inputs, DataList& outputs,
      ArithmeticConstClosure& closure, const CudaRuntimeContext& context) {
      CHECK_EQ(inputs.size(), 1) << "(arithmetic const) #inputs is wrong!";
      CHECK_EQ(outputs.size(), 1) << "(arithmetic const) #outputs is wrong!";
      float val = closure.val;
      float* in_data = inputs[0].GetCpuData();
      float* res_data = outputs[0].GetCpuData();
      int m = inputs[0].Size()[0];
      int n = inputs[0].Size()[1];
      switch (closure.type) {
      case ADD:
        abort();
        break;
      case SUB:
        abort();
        break;
      case MULT:
        CudaPerformScale(in_data, res_data, m, n, val, context.handle);
        break;
      case DIV:
        abort();
        break;
      }
    }

    void Transpose(DataList& inputs, DataList& outputs,
      TransposeClosure& closure, const CudaRuntimeContext& context) {
      float* in_data = inputs[0].GetCpuData();
      float* res_data = outputs[0].GetCpuData();
      int m = inputs[0].Size()[0];
      int n = inputs[0].Size()[1];
      CudaPerformTranspose(in_data, res_data, m, n, context.handle);
    }


  }
}
