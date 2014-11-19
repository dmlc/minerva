#include "op/impl/cuda.h"
#include "op/impl/cuda/cuda_perform.h"
#include "op/context.h"
#include "op/closure.h"
#include "common/cuda_utils.h"
#include <glog/logging.h>
#include <chrono>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

namespace minerva {
#ifdef HAS_CUDA
namespace cuda {

void Arithmetic(const DataList& inputs, const DataList& outputs, ArithmeticClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 2) << "Arithmetic takes 2 inputs";
  CHECK_EQ(outputs.size(), 1) << "Arithmetic takes 1 output";
  float* left = inputs[0].data();
  float* right = inputs[1].data();
  float* res = outputs[0].data();
  size_t size = outputs[0].size().Prod();
  switch (closure.type) {
    case ArithmeticType::kAdd:
      CudaPerformAdd(left, right, res, size, context.cublas_handle);
      break;
    case ArithmeticType::kSub:
      CudaPerformSub(left, right, res, size, context.cublas_handle);
      break;
    case ArithmeticType::kMult:
      CudaPerformDotMult(left, right, res, size, context.stream);
      break;
    case ArithmeticType::kDiv:
      CudaPerformDotDiv(left, right, res, size, context.stream);
      break;
  }
}

void MatMult(const DataList& inputs, const DataList& outputs, MatMultClosure& closure, const CudaRuntimeContext & context) {
  CHECK_EQ(inputs.size(), 2) << "(matmult) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(matmult) #outputs is wrong!";
  float* left_data = inputs[0].data();
  float* right_data = inputs[1].data();
  float* res_data = outputs[0].data();
  int m = inputs[0].size()[0];
  int k = inputs[0].size()[1];
  int n = outputs[0].size()[1];
  CudaPerformMatMult(left_data, right_data, res_data, m, n, k, context.cublas_handle);
}

void ArithmeticConst(const DataList& inputs, const DataList& outputs,
  ArithmeticConstClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "(arithmetic const) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(arithmetic const) #outputs is wrong!";
  float val = closure.val;
  float* in_data = inputs[0].data();
  float* res_data = outputs[0].data();
  size_t size = inputs[0].size().Prod();
  switch (closure.type) {
    case ArithmeticType::kAdd:
      CudaPerformConstAdd(in_data, res_data, val, size, context.stream);
      break;
    case ArithmeticType::kSub:
      if (closure.side == 0) {  // const on left
        CudaPerformLeftConstSub(in_data, res_data, val, size, context.stream);
      } else {
        CudaPerformConstAdd(in_data, res_data, -val, size, context.stream);
      }
      break;
    case ArithmeticType::kMult:
      CudaPerformScale(in_data, res_data, size, val, context.cublas_handle);
      break;
    case ArithmeticType::kDiv:
      if (closure.side == 0) {  // const on left
        CudaPerformLeftConstDiv(in_data, res_data, val, size, context.stream);
      } else {  // const on right
        CudaPerformScale(in_data, res_data, size, 1 / val, context.cublas_handle);
      }
      break;
  }
}

void Transpose(const DataList& inputs, const DataList& outputs,
  TransposeClosure& closure, const CudaRuntimeContext& context) {
  float* in_data = inputs[0].data();
  float* res_data = outputs[0].data();
  int m = inputs[0].size()[0];
  int n = inputs[0].size()[1];
  CudaPerformTranspose(in_data, res_data, m, n, context.cublas_handle);
}

void NormArithmetic(const DataList& inputs, const DataList& outputs, NormArithmeticClosure& closure,
  const CudaRuntimeContext & context) {
  CHECK_EQ(inputs.size(), 2) << "NormArithmetic kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "NormArithmetic kernel wrong #output";
  // Normalizee is the chunk with full size, normalizer is the chunk with reduced dimensions
  auto normalizee_size = inputs[0].size();
  auto normalizer_size = inputs[1].size();
  auto normalizee_data = inputs[0].data();
  auto normalizer_data = inputs[1].data();
  auto res_data = outputs[0].data();
  // TODO: support other types of norm op
  CHECK_EQ(normalizee_size.NumDims(), 2) << "currently support 2D normalizee matrix only";
  CHECK_EQ(closure.dims_to_replicate.NumDims(), 1) << "currently do norm on one dimension only";
  int m = normalizee_size[0];
  int n = normalizee_size[1];
  if (closure.dims_to_replicate[0] == 0) {
    switch(closure.type) {
      case ArithmeticType::kAdd:
        CudaPerformNormAddOnCol(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
      case ArithmeticType::kSub:
        CudaPerformNormSubOnCol(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
      case ArithmeticType::kMult:
        CudaPerformNormMultOnCol(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
      case ArithmeticType::kDiv:
        CudaPerformNormDivOnCol(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
    }
  } else {
    switch(closure.type) {
      case ArithmeticType::kAdd:
        CudaPerformNormAddOnRow(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
      case ArithmeticType::kSub:
        CudaPerformNormSubOnRow(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
      case ArithmeticType::kMult:
        CudaPerformNormMultOnRow(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
      case ArithmeticType::kDiv:
        CudaPerformNormDivOnRow(normalizee_data, normalizer_data, res_data, m, n, context.stream);
        break;
    }
  }
}

void Reduction(const DataList& inputs, const DataList& outputs,
  ReductionClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "Reduction kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "Reduction kernel wrong #output";
  auto in_size = inputs[0].size();
  auto out_size = outputs[0].size();
  auto in_data = inputs[0].data();
  auto out_data = outputs[0].data();
  // TODO: support other types of reduction op
  CHECK_EQ(in_size.NumDims(), 2) << "currently support 2D reduction matrix only";
  CHECK_EQ(closure.dims_to_reduce.NumDims(), 1) << "currently do reduction on one dimension only";
  int m = in_size[0];
  int n = in_size[1];
  if (closure.dims_to_reduce[0] == 0) {
    switch (closure.type) {
      case ReductionType::kSum:
        CudaPerformReductionSumOnCol(in_data, out_data, m, n, context.stream);
        break;
      case ReductionType::kMax:
        CudaPerformReductionMaxOnCol(in_data, out_data, m, n, context.stream);
        break;
    }
  } else {
    switch (closure.type) {
      case ReductionType::kSum:
        CudaPerformReductionSumOnRow(in_data, out_data, m, n, context.stream);
        break;
      case ReductionType::kMax:
        CudaPerformReductionMaxOnRow(in_data, out_data, m, n, context.stream);
        break;
    }
  }
}

void MaxIndex(const DataList& inputs, const DataList& outputs,
  MaxIndexClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "MaxIndex kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "MaxIndex kernel wrong #output";
  auto in_size = inputs[0].size();
  auto out_size = outputs[0].size();
  auto in_data = inputs[0].data();
  auto out_data = outputs[0].data();
  // TODO: support other types of max index op
  CHECK_EQ(in_size.NumDims(), 2) << "currently support 2D MaxIndex matrix only";
  int m = in_size[0];
  int n = in_size[1];
  if (closure.dim == 0) {
    CudaPerformMaxIndexOnCol(in_data, out_data, m, n, context.stream);
  } else {
    CudaPerformMaxIndexOnRow(in_data, out_data, m, n, context.stream);
  }
}

void Reshape(const DataList& inputs, const DataList& outputs, ReshapeClosure&, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  CudaPerformReshape(inputs[0].data(), outputs[0].data(), inputs[0].size().Prod() * sizeof(float), context.stream);
}


void Elewise(const DataList& inputs, const DataList& outputs, ElewiseClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
  float* in_data = inputs[0].data();
  float* res_data = outputs[0].data();
  int length = outputs[0].size().Prod();
  switch (closure.type) {
    case ElewiseType::kExp:
      CudaPerformElewiseExp(in_data, res_data, length, context.stream);
      break;
    case ElewiseType::kLn:
      CudaPerformElewiseLn(in_data, res_data, length, context.stream);
      break;
    case ElewiseType::kSigmoid:
      CudaPerformElewiseSigmoid(in_data, res_data, length, context.stream);
      break;
    case ElewiseType::kNegative:
      CudaPerformElewiseNegative(in_data, res_data, length, context.stream);
      break;
  }
}

void ConvForward(const DataList& inputs, const DataList& outputs, ConvForwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 3) << "(conv forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& filter = inputs[1];
  auto& bias = inputs[2];
  auto& top = outputs[0];
  int num_images = bottom.size()[3];
  int bottom_num_channels = bottom.size()[2];
  int top_num_channels = top.size()[2];
  int bottom_height = bottom.size()[1];
  int bottom_width = bottom.size()[0];
  int filter_height = filter.size()[1];
  int filter_width = filter.size()[0];
  CudaPerformConvForward(bottom.data(), filter.data(), bias.data(), top.data(), num_images, bottom_num_channels, top_num_channels, bottom_height, bottom_width, closure.pad_height, closure.pad_width, closure.stride_vertical, closure.stride_horizontal, filter_height, filter_width, context.stream, context.cudnn_handle);
}

void ConvBackwardData(const DataList& inputs, const DataList& outputs, ConvBackwardDataClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 2) << "(conv backward data) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv backward data) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& filter = inputs[1];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size()[3];
  int bottom_num_channels = bottom_diff.size()[2];
  int top_num_channels = top_diff.size()[2];
  int top_height = top_diff.size()[1];
  int top_width = top_diff.size()[0];
  int filter_height = filter.size()[1];
  int filter_width = filter.size()[0];
  CudaPerformConvBackwardData(top_diff.data(), filter.data(), bottom_diff.data(), num_images, bottom_num_channels, top_num_channels, top_height, top_width, closure.pad_height, closure.pad_width, closure.stride_vertical, closure.stride_horizontal, filter_height, filter_width, context.stream, context.cudnn_handle);
}

void ConvBackwardFilter(const DataList& inputs, const DataList& outputs, ConvBackwardFilterClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 2) << "(conv backward filter) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv backward filter) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& bottom = inputs[1];
  auto& filter_diff = outputs[0];
  int num_images = top_diff.size()[3];
  int bottom_num_channels = bottom.size()[2];
  int top_num_channels = top_diff.size()[2];
  int bottom_height = bottom.size()[1];
  int bottom_width = bottom.size()[0];
  int filter_height = filter_diff.size()[1];
  int filter_width = filter_diff.size()[0];
  CudaPerformConvBackwardFilter(bottom.data(), top_diff.data(), filter_diff.data(), num_images, bottom_num_channels, top_num_channels, bottom_height, bottom_width, closure.pad_height, closure.pad_width, closure.stride_vertical, closure.stride_horizontal, filter_height, filter_width, context.stream, context.cudnn_handle);
}

void ConvBackwardBias(const DataList& inputs, const DataList& outputs, ConvBackwardBiasClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "(conv backward bias) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv backward bias) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& bias_diff = outputs[0];
  int num_images = top_diff.size()[3];
  int top_num_channels = top_diff.size()[2];
  int top_height = top_diff.size()[1];
  int top_width = top_diff.size()[0];
  CudaPerformConvBackwardBias(top_diff.data(), bias_diff.data(), num_images, top_num_channels, top_height, top_width, context.stream, context.cudnn_handle);
}

void SoftmaxForward(const DataList& inputs, const DataList& outputs, SoftmaxForwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "(softmax forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(softmax forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  int num_images = bottom.size()[3];
  int num_channels = bottom.size()[2];
  int height = bottom.size()[1];
  int width = bottom.size()[0];
  switch (closure.algorithm) {
    case SoftmaxAlgorithm::kInstance:
      CudaPerformInstanceSoftmaxForward(bottom.data(), top.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case SoftmaxAlgorithm::kChannel:
      CudaPerformChannelSoftmaxForward(bottom.data(), top.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "softmax algorithm not supported";
  }
}

void SoftmaxBackward(const DataList& inputs, const DataList& outputs, SoftmaxBackwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 2) << "(softmax backward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(softmax backward) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size()[3];
  int num_channels = top_diff.size()[2];
  int height = top_diff.size()[1];
  int width = top_diff.size()[0];
  switch (closure.algorithm) {
    case SoftmaxAlgorithm::kInstance:
      CudaPerformInstanceSoftmaxBackward(top_diff.data(), top.data(), bottom_diff.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case SoftmaxAlgorithm::kChannel:
      CudaPerformChannelSoftmaxBackward(top_diff.data(), top.data(), bottom_diff.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "softmax algorithm not supported";
  }
}

void ActivationForward(const DataList& inputs, const DataList& outputs, ActivationForwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "(activation forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(activation forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  int num_images = bottom.size()[3];
  int num_channels = bottom.size()[2];
  int height = bottom.size()[1];
  int width = bottom.size()[0];
  switch (closure.algorithm) {
    case ActivationAlgorithm::kSigmoid:
      CudaPerformSigmoidForward(bottom.data(), top.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kRelu:
      CudaPerformReluForward(bottom.data(), top.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kTanh:
      CudaPerformTanhForward(bottom.data(), top.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "activation algorithm not supported";
  }
}

void ActivationBackward(const DataList& inputs, const DataList& outputs, ActivationBackwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 3) << "(activation backward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(activation backward) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size()[3];
  int num_channels = top_diff.size()[2];
  int height = top_diff.size()[1];
  int width = top_diff.size()[0];
  switch (closure.algorithm) {
    case ActivationAlgorithm::kSigmoid:
      CudaPerformSigmoidBackward(bottom.data(), top.data(), top_diff.data(), bottom_diff.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kRelu:
      CudaPerformReluBackward(bottom.data(), top.data(), top_diff.data(), bottom_diff.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kTanh:
      CudaPerformTanhBackward(bottom.data(), top.data(), top_diff.data(), bottom_diff.data(), num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "activation algorithm not supported";
  }
}

void PoolingForward(const DataList& inputs, const DataList& outputs, PoolingForwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 1) << "(pooling forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(pooling forward) #inputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  int num_images = bottom.size()[3];
  int num_channels = bottom.size()[2];
  int bottom_height = bottom.size()[1];
  int bottom_width = bottom.size()[0];
  switch (closure.algorithm) {
    case PoolingInfo::Algorithm::kMax:
      CudaPerformMaxPoolingForward(bottom.data(), top.data(), num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, context.stream, context.cudnn_handle);
      break;
    case PoolingInfo::Algorithm::kAverage:
      CudaPerformAveragePoolingForward(bottom.data(), top.data(), num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "pooling algorithm not supported";
  }
}

void PoolingBackward(const DataList& inputs, const DataList& outputs, PoolingBackwardClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(inputs.size(), 3) << "(pooling backward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(pooling backward) #inputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size()[3];
  int num_channels = top_diff.size()[2];
  int bottom_height = bottom.size()[1];
  int bottom_width = bottom.size()[0];
  switch (closure.algorithm) {
    case PoolingInfo::Algorithm::kMax:
      CudaPerformMaxPoolingBackward(bottom.data(), top.data(), top_diff.data(), bottom_diff.data(), num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, context.stream, context.cudnn_handle);
      break;
    case PoolingInfo::Algorithm::kAverage:
      CudaPerformAveragePoolingBackward(bottom.data(), top.data(), top_diff.data(), bottom_diff.data(), num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "pooling algorithm not supported";
  }
}

void ArrayLoader(const DataList& outputs, ArrayLoaderClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(outputs.size(), 1) << "(array loader) #outputs wrong";
  CHECK(closure.data) << "probably already executed";
  CUDA_CALL(cudaMemcpyAsync(outputs[0].data(), closure.data.get(), outputs[0].size().Prod() * sizeof(float), cudaMemcpyDefault));
  closure.data.reset();
}

void Randn(const DataList& outputs, RandnClosure& closure, const CudaRuntimeContext&) {
  CHECK_EQ(outputs.size(), 1) << "(randn) #outputs wrong";
  CudaPerformRandn(outputs[0].data(), outputs[0].size().Prod(), chrono::system_clock::now().time_since_epoch().count(), closure.mu, closure.var);
}

void Fill(const DataList& outputs, FillClosure& closure, const CudaRuntimeContext& context) {
  CHECK_EQ(outputs.size(), 1) << "(fill) #outputs wrong";
  CudaPerformFill(outputs[0].data(), outputs[0].size().Prod(), closure.val, context.stream);
}

}
#endif
}

