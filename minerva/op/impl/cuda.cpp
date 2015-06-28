#include "op/impl/cuda.h"
#include "op/impl/cuda/cuda_perform.h"
#include "op/context.h"
#include "op/closure.h"
#include "common/cuda_utils.h"
#include <dmlc/logging.h>
#include <chrono>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAS_PS
#include "ps.h"
#endif

using namespace std;

namespace minerva {
#ifdef HAS_CUDA
namespace cuda {

void Arithmetic(const DataList& inputs, const DataList& outputs, ArithmeticClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 2) << "Arithmetic takes 2 inputs";
  CHECK_EQ(outputs.size(), 1) << "Arithmetic takes 1 output";
  float* left = inputs[0].data_;
  float* right = inputs[1].data_;
  float* res = outputs[0].data_;
  size_t size = outputs[0].size_.Prod();
  switch (closure.type) {
    case ArithmeticType::kAdd:
      CudaPerformAdd(left, right, res, size, context.stream);
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

void LRNForward(const DataList& inputs, const DataList& outputs, LRNForwardClosure& closure, const Context & context) {
  CHECK_EQ(inputs.size(), 2) << "(LRNForward) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(LRNForward) #outputs is wrong!";
  float* bottom_data = inputs[0].data_;
  float* scale_data = inputs[1].data_;
  float* res_data = outputs[0].data_;
  int local_size = closure.local_size;
  float alpha = closure.alpha;
  float beta = closure.beta;
  int num_img = closure.data_shape[3];
  int channel = closure.data_shape[2];
  int weight = closure.data_shape[1];
  int height = closure.data_shape[0];
  CudaPerformLRNForward(bottom_data, scale_data, res_data, local_size, alpha, beta, num_img, channel, weight, height, context.stream);
}

void LRNBackward(const DataList& inputs, const DataList& outputs, LRNBackwardClosure& closure, const Context & context) {
  CHECK_EQ(inputs.size(), 4) << "(LRNBackward) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(LRNBackward) #outputs is wrong!";
  float* bottom_data = inputs[0].data_;
  float* top_data = inputs[1].data_;
  float* scale_data = inputs[2].data_;
  float* top_diff = inputs[3].data_;
  float* bottom_diff = outputs[0].data_;
  int local_size = closure.local_size;
  float alpha = closure.alpha;
  float beta = closure.beta;
  int num_img = closure.data_shape[3];
  int channel = closure.data_shape[2];
  int weight = closure.data_shape[1];
  int height = closure.data_shape[0];
  CudaPerformLRNBackward(bottom_data, top_data, scale_data, top_diff, bottom_diff, local_size, alpha, beta, num_img, channel, weight, height, context.stream);
}


void Concat(const DataList& inputs, const DataList& outputs, ConcatClosure& closure, const Context & context) {
  CHECK_GT(inputs.size(), 1) << "(Concat) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(Concat) #outputs is wrong!";
  CHECK_LE(inputs[0].size_.NumDims() - closure.catdim, 3) << "(Concat) #Currently only support concat on the last two dims!";

  size_t concat_dim = closure.catdim;
  float* top_data = outputs[0].data_;
  if (concat_dim == inputs[0].size_.NumDims() - 1) {
    int offset_num = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      float* bottom_data = inputs[i].data_;
      CudaPerformCopy(bottom_data, top_data + offset_num, inputs[i].size_.Prod(), context.cublas_handle);
      offset_num += inputs[i].size_.Prod();
    }
  }
  else {
    int offset_channel = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      float* bottom_data = inputs[i].data_;
      int bot_num_elem = 1;
      int top_num_elem = 1;
      int img_size = 1;
      int bot_channel = 1;
      for (size_t idx = 0; idx < inputs[i].size_.NumDims() - 1; idx++){
        bot_num_elem *= inputs[i].size_[idx];
        top_num_elem *= outputs[0].size_[idx];
        if(idx < inputs[i].size_.NumDims() - 2)
          img_size *= inputs[i].size_[idx];
        else
          bot_channel = inputs[i].size_[idx];
      }
      int imgnum = inputs[i].size_[inputs[i].size_.NumDims()-1];
      for (int n = 0; n < imgnum; ++n) {
        CudaPerformCopy(bottom_data + n * bot_num_elem, top_data + n * top_num_elem + offset_channel * img_size, bot_num_elem, context.cublas_handle);
      }
      offset_channel += bot_channel;
    }
  }
}

void Slice(const DataList& inputs, const DataList& outputs, SliceClosure& closure, const Context & context)
{
  CHECK_EQ(inputs.size(), 1) << "(Slice) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(Slice) #outputs is wrong!";
  CHECK_LE(inputs[0].size_.NumDims() - closure.slice_dim, 3) << "(Slice) #Currently only support concat on the last two dims!";

  size_t slice_dim = closure.slice_dim;
  float* input_data = inputs[0].data_;
  if (slice_dim == inputs[0].size_.NumDims() - 1){
    int offset_num = 0;
    float* output_data = outputs[0].data_;
    int img_size = 1;
    for (size_t i = 0; i < inputs[0].size_.NumDims() - 1; i++) {
      img_size *= inputs[0].size_[i];
    }
    offset_num = closure.st_off * img_size;
    CudaPerformCopy(input_data + offset_num, output_data, outputs[0].size_.Prod(), context.cublas_handle);
  }
  else{
    int offset_channel = closure.st_off;
    float* output_data = outputs[0].data_;
    int output_num_elem = 1;
    int input_num_elem = 1;
    int img_size = 1;
    for (size_t idx = 0; idx < outputs[0].size_.NumDims() - 1; idx++){
      output_num_elem *= outputs[0].size_[idx];
      input_num_elem *= inputs[0].size_[idx];
      if(idx < inputs[0].size_.NumDims() - 2)
        img_size *= outputs[0].size_[idx];
    }
    int imgnum = inputs[0].size_[inputs[0].size_.NumDims()-1];
    for (int n = 0; n < imgnum; ++n) {
      CudaPerformCopy(input_data + n * input_num_elem + offset_channel * img_size, output_data + n * output_num_elem, output_num_elem, context.cublas_handle);
    }
  }
}


void MatMult(const DataList& inputs, const DataList& outputs, MatMultClosure& closure, const Context & context) {
  CHECK_EQ(inputs.size(), 2) << "(matmult) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(matmult) #outputs is wrong!";
  float* left_data = inputs[0].data_;
  float* right_data = inputs[1].data_;
  float* res_data = outputs[0].data_;
  int m = inputs[0].size_[0];
  int k = inputs[0].size_[1];
  int n = outputs[0].size_[1];
  CudaPerformMatMult(left_data, right_data, res_data, m, n, k, context.cublas_handle);
}

void ArithmeticConst(const DataList& inputs, const DataList& outputs,
  ArithmeticConstClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "(arithmetic const) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(arithmetic const) #outputs is wrong!";
  float val = closure.val;
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
  size_t size = inputs[0].size_.Prod();
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
  TransposeClosure& closure, const Context& context) {
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
  int m = inputs[0].size_[0];
  int n = inputs[0].size_[1];
  CudaPerformTranspose(in_data, res_data, m, n, context.cublas_handle);
}

void NormArithmetic(const DataList& inputs, const DataList& outputs, NormArithmeticClosure& closure,
  const Context & context) {
  CHECK_EQ(inputs.size(), 2) << "NormArithmetic kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "NormArithmetic kernel wrong #output";
  // Normalizee is the chunk with full size, normalizer is the chunk with reduced dimensions
  auto normalizee_size = inputs[0].size_;
  auto normalizer_size = inputs[1].size_;
  auto normalizee_data = inputs[0].data_;
  auto normalizer_data = inputs[1].data_;
  auto res_data = outputs[0].data_;
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

void ReductionOnDim(const DataList& inputs, const DataList& outputs,
  ReductionOnDimClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "Reduction kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "Reduction kernel wrong #output";
  auto in_size = inputs[0].size_;
  auto out_size = outputs[0].size_;
  auto in_data = inputs[0].data_;
  auto out_data = outputs[0].data_;
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

void Reduction(const DataList& inputs, const DataList& outputs,
  ReductionClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "Reduction kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "Reduction kernel wrong #output";
  auto in_size = inputs[0].size_;
  auto out_size = outputs[0].size_;
  auto in_data = inputs[0].data_;
  auto out_data = outputs[0].data_;
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

void ReductionWithReshape(const DataList& inputs, const DataList& outputs,
  ReductionWithReshapeClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "Reduction kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "Reduction kernel wrong #output";
  auto in_size = inputs[0].size_;
  auto out_size = outputs[0].size_;
  auto in_data = inputs[0].data_;
  auto out_data = outputs[0].data_;
  //Check the reshape is reasonable
  int total_dim = 1;
  for (auto i : dim) {
	total_dim *= in_size[i];
  }
  CHECK_EQ(closure.newshape.NumDims(), 2) << "Only Redhape to a 2-D NArray";
  CHECK_EQ(total_dim, closure.newshape[0] * closure.newshape[1]) << "Reshape Dimension Mismatch";
  
  int m = closure.newshape[0];
  int n = closure.newshape[1];
  if (closure.dims_to_reduce == 0) {
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
  MaxIndexClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "MaxIndex kernel wrong #input";
  CHECK_EQ(outputs.size(), 1) << "MaxIndex kernel wrong #output";
  auto in_size = inputs[0].size_;
  auto out_size = outputs[0].size_;
  auto in_data = inputs[0].data_;
  auto out_data = outputs[0].data_;
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

void Reshape(const DataList& inputs, const DataList& outputs, ReshapeClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  CudaPerformReshape(inputs[0].data_, outputs[0].data_, inputs[0].size_.Prod() * sizeof(float), context.stream);
}


void Elewise(const DataList& inputs, const DataList& outputs, ElewiseClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "(elewise) #inputs is wrong!";
  CHECK_EQ(outputs.size(), 1) << "(elewise) #outputs is wrong!";
  float* in_data = inputs[0].data_;
  float* res_data = outputs[0].data_;
  int length = outputs[0].size_.Prod();
  switch (closure.type) {
    case ElewiseType::kExp:
      CudaPerformElewiseExp(in_data, res_data, length, context.stream);
      break;
    case ElewiseType::kLn:
      CudaPerformElewiseLn(in_data, res_data, length, context.stream);
      break;
    case ElewiseType::kNegative:
      CudaPerformElewiseNegative(in_data, res_data, length, context.stream);
      break;
    default:
      LOG(FATAL) << "elewise algorithm not supported";
  }
}

void SigmoidForward(const DataList& inputs, const DataList& outputs, SigmoidForwardClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "sigmoid forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "sigmoid forward #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  CudaPerformSigmoidForward(bottom.data_, top.data_, 1, 1, 1, bottom.size_.Prod(), context.stream, context.cudnn_handle);
}

void SigmoidBackward(const DataList& inputs, const DataList& outputs, SigmoidBackwardClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 3) << "sigmoid backward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "sigmoid backward #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  CudaPerformSigmoidBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, 1, 1, 1, top_diff.size_.Prod(), context.stream, context.cudnn_handle);
}

void ReluForward(const DataList& inputs, const DataList& outputs, ReluForwardClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "relu forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "relu forward #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  CudaPerformReluForward(bottom.data_, top.data_, 1, 1, 1, bottom.size_.Prod(), context.stream, context.cudnn_handle);
}

void ReluBackward(const DataList& inputs, const DataList& outputs, ReluBackwardClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 3) << "relu backward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "relu backward #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  CudaPerformReluBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, 1, 1, 1, top_diff.size_.Prod(), context.stream, context.cudnn_handle);
}

void TanhForward(const DataList& inputs, const DataList& outputs, TanhForwardClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "tanh forward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "tanh forward #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  CudaPerformTanhForward(bottom.data_, top.data_, 1, 1, 1, bottom.size_.Prod(), context.stream, context.cudnn_handle);
}

void TanhBackward(const DataList& inputs, const DataList& outputs, TanhBackwardClosure&, const Context& context) {
  CHECK_EQ(inputs.size(), 3) << "tanh backward #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "tanh backward #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  CudaPerformTanhBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, 1, 1, 1, top_diff.size_.Prod(), context.stream, context.cudnn_handle);
}

void ConvForward(const DataList& inputs, const DataList& outputs, ConvForwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 3) << "(conv forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& filter = inputs[1];
  auto& bias = inputs[2];
  auto& top = outputs[0];
  int num_images = bottom.size_[3];
  int bottom_num_channels = bottom.size_[2];
  int top_num_channels = top.size_[2];
  int bottom_height = bottom.size_[1];
  int bottom_width = bottom.size_[0];
  int filter_height = filter.size_[1];
  int filter_width = filter.size_[0];
  CudaPerformConvForward(bottom.data_, filter.data_, bias.data_, top.data_, num_images, bottom_num_channels, top_num_channels, bottom_height, bottom_width, closure.pad_height, closure.pad_width, closure.stride_vertical, closure.stride_horizontal, filter_height, filter_width, context.stream, context.cudnn_handle);
}

void ConvBackwardData(const DataList& inputs, const DataList& outputs, ConvBackwardDataClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 2) << "(conv backward data) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv backward data) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& filter = inputs[1];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size_[3];
  int bottom_num_channels = bottom_diff.size_[2];
  int top_num_channels = top_diff.size_[2];
  int top_height = top_diff.size_[1];
  int top_width = top_diff.size_[0];
  int filter_height = filter.size_[1];
  int filter_width = filter.size_[0];
  CudaPerformConvBackwardData(top_diff.data_, filter.data_, bottom_diff.data_, num_images, bottom_num_channels, top_num_channels, top_height, top_width, closure.pad_height, closure.pad_width, closure.stride_vertical, closure.stride_horizontal, filter_height, filter_width, context.stream, context.cudnn_handle);
}

void ConvBackwardFilter(const DataList& inputs, const DataList& outputs, ConvBackwardFilterClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 2) << "(conv backward filter) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv backward filter) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& bottom = inputs[1];
  auto& filter_diff = outputs[0];
  int num_images = top_diff.size_[3];
  int bottom_num_channels = bottom.size_[2];
  int top_num_channels = top_diff.size_[2];
  int bottom_height = bottom.size_[1];
  int bottom_width = bottom.size_[0];
  int filter_height = filter_diff.size_[1];
  int filter_width = filter_diff.size_[0];
  CudaPerformConvBackwardFilter(bottom.data_, top_diff.data_, filter_diff.data_, num_images, bottom_num_channels, top_num_channels, bottom_height, bottom_width, closure.pad_height, closure.pad_width, closure.stride_vertical, closure.stride_horizontal, filter_height, filter_width, context.stream, context.cudnn_handle);
}

void ConvBackwardBias(const DataList& inputs, const DataList& outputs, ConvBackwardBiasClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "(conv backward bias) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(conv backward bias) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& bias_diff = outputs[0];
  int num_images = top_diff.size_[3];
  int top_num_channels = top_diff.size_[2];
  int top_height = top_diff.size_[1];
  int top_width = top_diff.size_[0];
  CudaPerformConvBackwardBias(top_diff.data_, bias_diff.data_, num_images, top_num_channels, top_height, top_width, context.stream, context.cudnn_handle);
}

void SoftmaxForward(const DataList& inputs, const DataList& outputs, SoftmaxForwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "(softmax forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(softmax forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  int num_images = bottom.size_[3];
  int num_channels = bottom.size_[2];
  int height = bottom.size_[1];
  int width = bottom.size_[0];
  switch (closure.algorithm) {
    case SoftmaxAlgorithm::kInstance:
      CudaPerformInstanceSoftmaxForward(bottom.data_, top.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case SoftmaxAlgorithm::kChannel:
      CudaPerformChannelSoftmaxForward(bottom.data_, top.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "softmax algorithm not supported";
  }
}

void SoftmaxBackward(const DataList& inputs, const DataList& outputs, SoftmaxBackwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 2) << "(softmax backward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(softmax backward) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size_[3];
  int num_channels = top_diff.size_[2];
  int height = top_diff.size_[1];
  int width = top_diff.size_[0];
  switch (closure.algorithm) {
    case SoftmaxAlgorithm::kInstance:
      CudaPerformInstanceSoftmaxBackward(top_diff.data_, top.data_, bottom_diff.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case SoftmaxAlgorithm::kChannel:
      CudaPerformChannelSoftmaxBackward(top_diff.data_, top.data_, bottom_diff.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "softmax algorithm not supported";
  }
}

void ActivationForward(const DataList& inputs, const DataList& outputs, ActivationForwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "(activation forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(activation forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  int num_images = bottom.size_[3];
  int num_channels = bottom.size_[2];
  int height = bottom.size_[1];
  int width = bottom.size_[0];
  switch (closure.algorithm) {
    case ActivationAlgorithm::kSigmoid:
      CudaPerformSigmoidForward(bottom.data_, top.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kRelu:
      CudaPerformReluForward(bottom.data_, top.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kTanh:
      CudaPerformTanhForward(bottom.data_, top.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "activation algorithm not supported";
  }
}

void ActivationBackward(const DataList& inputs, const DataList& outputs, ActivationBackwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 3) << "(activation backward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(activation backward) #outputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size_[3];
  int num_channels = top_diff.size_[2];
  int height = top_diff.size_[1];
  int width = top_diff.size_[0];
  switch (closure.algorithm) {
    case ActivationAlgorithm::kSigmoid:
      CudaPerformSigmoidBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kRelu:
      CudaPerformReluBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    case ActivationAlgorithm::kTanh:
      CudaPerformTanhBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, num_images, num_channels, height, width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "activation algorithm not supported";
  }
}

void PoolingForward(const DataList& inputs, const DataList& outputs, PoolingForwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1) << "(pooling forward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(pooling forward) #outputs wrong";
  auto& bottom = inputs[0];
  auto& top = outputs[0];
  int num_images = bottom.size_[3];
  int num_channels = bottom.size_[2];
  int bottom_height = bottom.size_[1];
  int bottom_width = bottom.size_[0];
  switch (closure.algorithm) {
    case PoolingInfo::Algorithm::kMax:
      CudaPerformMaxPoolingForward(bottom.data_, top.data_, num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, closure.pad_height, closure.pad_width, context.stream, context.cudnn_handle);
      break;
    case PoolingInfo::Algorithm::kAverage:
      CudaPerformAveragePoolingForward(bottom.data_, top.data_, num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, closure.pad_height, closure.pad_width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "pooling algorithm not supported";
  }
}

void PoolingBackward(const DataList& inputs, const DataList& outputs, PoolingBackwardClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 3) << "(pooling backward) #inputs wrong";
  CHECK_EQ(outputs.size(), 1) << "(pooling backward) #inputs wrong";
  auto& top_diff = inputs[0];
  auto& top = inputs[1];
  auto& bottom = inputs[2];
  auto& bottom_diff = outputs[0];
  int num_images = top_diff.size_[3];
  int num_channels = top_diff.size_[2];
  int bottom_height = bottom.size_[1];
  int bottom_width = bottom.size_[0];
  switch (closure.algorithm) {
    case PoolingInfo::Algorithm::kMax:
      CudaPerformMaxPoolingBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, closure.pad_height, closure.pad_width, context.stream, context.cudnn_handle);
      break;
    case PoolingInfo::Algorithm::kAverage:
      CudaPerformAveragePoolingBackward(bottom.data_, top.data_, top_diff.data_, bottom_diff.data_, num_images, num_channels, bottom_height, bottom_width, closure.stride_vertical, closure.stride_horizontal, closure.height, closure.width, closure.pad_height, closure.pad_width, context.stream, context.cudnn_handle);
      break;
    default:
      LOG(FATAL) << "pooling algorithm not supported";
  }
}

void ArrayLoader(const DataList& outputs, ArrayLoaderClosure& closure, const Context& context) {
  CHECK_EQ(outputs.size(), 1) << "(array loader) #outputs wrong";
  CHECK(closure.data) << "probably already executed";
  CUDA_CALL(cudaMemcpyAsync(outputs[0].data_, closure.data.get(), outputs[0].size_.Prod() * sizeof(float), cudaMemcpyDefault));
  closure.data.reset();
}

void Randn(const DataList& outputs, RandnClosure& closure, const Context&) {
  CHECK_EQ(outputs.size(), 1) << "(normal) #outputs wrong";
  CudaPerformRandn(outputs[0].data_, outputs[0].size_.Prod(), chrono::system_clock::now().time_since_epoch().count(), closure.mu, closure.var);
}

void RandBernoulli(const DataList& outputs, RandBernoulliClosure& closure, const Context& context) {
  CHECK_EQ(outputs.size(), 1) << "(bernoulli) #outputs wrong";
  CudaPerformRandBernoulli(outputs[0].data_, outputs[0].size_.Prod(), chrono::system_clock::now().time_since_epoch().count(), closure.p, context.stream);
}

void Fill(const DataList& outputs, FillClosure& closure, const Context& context) {
  CHECK_EQ(outputs.size(), 1) << "(fill) #outputs wrong";
  CudaPerformFill(outputs[0].data_, outputs[0].size_.Prod(), closure.val, context.stream);
}

void SyncWithPS(const DataList& inputs, const DataList& outputs, SyncWithPSClosure& closure, const Context& context) {
  CHECK_EQ(outputs.size(), 1);
#ifdef HAS_PS
  // TODO: use memory allocator, or directly pass CPU pointer in
  // we are creating temp space on CPU for now, should use memory allocator
  size_t size = outputs[0].size_.Prod();
  vector<float> weight(size);
  if (inputs.empty())
  {
    PushGradAndPullWeight(nullptr, &weight[0], size, closure.layer_name);
  }
  else
  {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size_.Prod(), outputs[0].size_.Prod()) << "Pushed and pulled matrix must be of same dim";
    vector<float> grad(size);
    CUDA_CALL(cudaMemcpyAsync(&grad[0], inputs[0].data_, size, cudaMemcpyDefault, context.stream));
    CUDA_CALL(cudaStreamSynchronize(context.stream));
    PushGradAndPullWeight(&grad[0], &weight[0], size, closure.layer_name);
  }
  CUDA_CALL(cudaMemcpyAsync(outputs[0].data_, &weight[0], size, cudaMemcpyDefault, context.stream));
#else
  LOG(FATAL) << "HAS_PS is not enabled when you compile minerva, please enable it";
#endif
}

void Index(const DataList& inputs, const DataList& outputs, IndexClosure& closure, const Context& context) {
	cout << "a" << endl;
}

void Select(DataList const& inputs, DataList const& outputs, SelectClosure& closure, const Context& context) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(inputs[0].size_.NumDims(), 2);
  CHECK_EQ(outputs[0].size_.NumDims(), 2);
  CHECK_EQ(inputs[0].size_[0], outputs[0].size_[0]);
  CHECK_EQ(outputs[0].size_[1], closure.indices.size());
  for (auto i : closure.indices) {
    CHECK_LT(i, inputs[0].size_[1]);
  }
  CudaPerformSelect(outputs[0].data_, inputs[0].data_, closure.indices, inputs[0].size_[1], inputs[0].size_[0], context.stream);
}

}
#endif
}
