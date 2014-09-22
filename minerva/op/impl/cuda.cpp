#include "op/impl/cuda.h"
#include "op/impl/cuda/cuda_perform.h"
#include "op/context.h"
#include <glog/logging.h>
#include <cuda_runtime.h>

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
        CudaPerformAdd(left, right, res, m, n, context.handle);
        break;
      case SUB:
        CudaPerformSub(left, right, res, m, n, context.handle);
        break;
      case MULT:
        CudaPerformDotMult(left, right, res, size, context.stream);
        break;
      case DIV:
	abort();
        CudaPerformDotDiv(res, left, right, size, context.stream);
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
      float* in_data = inputs[0].GetGpuData();
      float* res_data = outputs[0].GetGpuData();
      int m = inputs[0].Size()[0];
      int n = inputs[0].Size()[1];
      size_t size = inputs[0].Size().Prod();
      switch (closure.type) {
      case ADD:
        abort();
        break;
      case SUB:
        if (closure.side == 0)  // const on left
        {
          CudaPerformLeftConstSub(in_data, res_data, val, size, context.stream);
        }
        else
        {
          CHECK(false) << "we support const on left only";
        }
        break;
      case MULT:
        CudaPerformScale(in_data, res_data, m, n, val, context.handle);
        break;
      case DIV:
        if (closure.side == 0) {// const on left
          CudaPerformLeftConstDiv(in_data, res_data, val, size, context.stream);
        }
        else {// const on right
          CudaPerformScale(in_data, res_data, m, n, 1 / val, context.handle);
        }
        break;
      }
    }

    void Transpose(DataList& inputs, DataList& outputs,
      TransposeClosure& closure, const CudaRuntimeContext& context) {
      float* in_data = inputs[0].GetGpuData();
      float* res_data = outputs[0].GetGpuData();
      int m = inputs[0].Size()[0];
      int n = inputs[0].Size()[1];
      CudaPerformTranspose(in_data, res_data, m, n, context.handle);
    }

    void NormArithmetic(DataList& inputs, DataList& outputs, NormArithmeticClosure& closure,
      const CudaRuntimeContext & context) {
      CHECK_EQ(inputs.size(), 2) << "NormArithmetic kernel wrong #input";
      CHECK_EQ(outputs.size(), 1) << "NormArithmetic kernel wrong #output";
      // Normalizee is the chunk with full size, normalizer is the chunk with reduced dimensions
      auto normalizee_size = inputs[0].Size();
      auto normalizer_size = inputs[1].Size();

      CHECK_EQ(normalizee_size, outputs[0].Size()) << "NormArithmetic kernel output size mismatch";
      for (size_t i = 0; i < normalizee_size.NumDims(); ++i) {
        if (normalizer_size[i] != 1 && normalizer_size[i] != normalizee_size[i]) {
          CHECK(false) << "NormArithmetic kernel size mismatch";
        }
      }
      auto normalizee_range = ScaleRange::MakeRangeFromOrigin(normalizee_size);
      auto normalizer_range = ScaleRange::MakeRangeFromOrigin(normalizer_size);
      auto normalizee_data = inputs[0].GetGpuData();
      auto normalizer_data = inputs[1].GetGpuData();
      auto res_data = outputs[0].GetGpuData();

      // TODO: support other types of norm op
      CHECK(normalizee_size.NumDims() == 2) << "currently support 2D normalizee matrix only, got "
        << normalizee_size.NumDims();
      CHECK(normalizer_size.NumDims() == 2) << "currently support 2D normalizer matrix only, got "
        << normalizer_size.NumDims();

      int m = normalizee_size[0];
      int n = normalizee_size[1];
      if (normalizer_size[0] == 1)
      {
        CHECK_EQ(normalizee_size[1], normalizer_size[1]) << "we can only do norm on one dimmension";
        CudaPerformNormOnCol(normalizee_data, normalizer_data, res_data, m, n, closure.type, context.stream);
      }
      else if (normalizer_size[1] == 1)
      {
        CHECK_EQ(normalizee_size[0], normalizer_size[0]) << "we can only do norm on one dimmension";
        CudaPerformNormOnRow(normalizee_data, normalizer_data, res_data, m, n, closure.type, context.stream);
      }
      else
      {
        CHECK(false) << "both two dimensions of normalizer are not 1";
      }
    }

    void Reduction(DataList& inputs, DataList& outputs,
      ReductionClosure& closure, const CudaRuntimeContext& context)
    {
      //CHECK_EQ(inputs.size(), 1) << "(reduction) #inputs is wrong!";
      //CHECK_EQ(outputs.size(), 1) << "(reduction) #outputs is wrong!";
      //float* in_data = inputs[0].GetGpuData();
      //float* res_data = outputs[0].GetGpuData();
      //auto in_max = inputs[0].Size();
      //auto res_max = outputs[0].Size();
      //auto accumulator = Scale::Origin(in_max.NumDims());
      //do {
      //  auto cur = accumulator;
      //  float tmp = in_data[in_range.Flatten(cur)];
      //  while (cur.IncrDimensions(in_max, closure.dims_to_reduce)) {
      //    float tmp2 = in_data[in_range.Flatten(cur)];
      //    // TODO Moving switch out of loop to optimize
      //    switch (closure.type) {
      //    case SUM:
      //      tmp += tmp2;
      //      break;
      //    case MAX:
      //      if (tmp < tmp2) {
      //        tmp = tmp2;
      //      }
      //      break;
      //    }
      //  }
      //  res_data[res_range.Flatten(accumulator)] = tmp;
      //} while (accumulator.IncrWithDimensionsFixed(res_max, closure.dims_to_reduce));
    }

    void MaxIndex(DataList&, DataList&, MaxIndexClosure&, const CudaRuntimeContext&)
    {}

    void Elewise(DataList& inputs, DataList& outputs, 
      ElewiseClosure& closure, const CudaRuntimeContext& context)
    {
      CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
      CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
      float* in_data = inputs[0].GetGpuData();
      float* res_data = outputs[0].GetGpuData();
      int length = outputs[0].Size().Prod();
      CudaPerformEleWise(in_data, res_data, length, closure.type, context.stream);
    }

  }
}
