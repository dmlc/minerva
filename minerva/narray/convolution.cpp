#include "narray/convolution.h"
#include "op/physical_op.h"

namespace minerva {

ImageBatch Convolution::ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info) {
  CHECK_EQ(src.GetNumFeatureMaps(), filter.GetNumInputs()) << "#input channels mismatch";
  CHECK_EQ(bias.Size().NumDims(), 1) << "bias dimension mismatch";
  CHECK_EQ(bias.Size()[0], filter.GetNumOutputs()) << "bias size mismatch";
  //no such limit
  //CHECK_EQ((src.GetHeight() + 2 * info.pad_height - filter.GetHeight()) % info.stride_vertical, 0) << "filter height mismatch";
  //CHECK_EQ((src.GetWidth() + 2 * info.pad_width - filter.GetWidth()) % info.stride_horizontal, 0) << "filter width mismatch";
  Scale new_size {
    (src.GetWidth() + 2 * info.pad_width - filter.GetWidth()) / info.stride_horizontal + 1,
    (src.GetHeight() + 2 * info.pad_height - filter.GetHeight()) / info.stride_vertical + 1,
    filter.GetNumOutputs(),
    src.GetNumImages()
  };
  ConvForwardOp* op = new ConvForwardOp();
  op->closure = {
    info.pad_height,
    info.pad_width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({src, filter, bias}, new_size, op);
}

ImageBatch Convolution::ConvBackwardData(ImageBatch diff, ImageBatch bottom, Filter filter, ConvInfo info) {
  CHECK_EQ(diff.GetNumFeatureMaps(), filter.GetNumOutputs()) << "#output channels mismatch";
  /*
   * We can't get filter size when (top + 2*pad) % stride != 0
  Scale new_size {
    (diff.GetWidth() - 1) * info.stride_horizontal + filter.GetWidth() - 2 * info.pad_width,
    (diff.GetHeight() - 1) * info.stride_vertical + filter.GetHeight() - 2 * info.pad_height,
    filter.GetNumInputs(),
    diff.GetNumImages()
  };
  */
  ConvBackwardDataOp* op = new ConvBackwardDataOp();
  op->closure = {
    info.pad_height,
    info.pad_width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({diff, filter}, bottom.Size(), op);
}

Filter Convolution::ConvBackwardFilter(ImageBatch diff, ImageBatch bottom, Filter filter, ConvInfo info) {
  CHECK_EQ(diff.GetNumImages(), bottom.GetNumImages()) << "#images mismatch";
  /*
   * We can't get filter size when (top + 2*pad) % stride != 0
  Scale new_size {
    -(diff.GetWidth() - 1) * info.stride_horizontal + bottom.GetWidth() + 2 * info.pad_width,
    -(diff.GetHeight() - 1) * info.stride_vertical + bottom.GetHeight() + 2 * info.pad_height,
    bottom.GetNumFeatureMaps(),
    diff.GetNumFeatureMaps()
  };
  */
  ConvBackwardFilterOp* op = new ConvBackwardFilterOp();
  op->closure = {
    info.pad_height,
    info.pad_width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({diff, bottom}, filter.Size(), op);
}

NArray Convolution::ConvBackwardBias(ImageBatch diff) {
  Scale new_size {
    diff.GetNumFeatureMaps()
  };
  ConvBackwardBiasOp* op = new ConvBackwardBiasOp();
  return NArray::ComputeOne({diff}, new_size, op);
}

ImageBatch Convolution::SoftmaxForward(ImageBatch src, SoftmaxAlgorithm algorithm) {
  SoftmaxForwardOp* op = new SoftmaxForwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({src}, src.Size(), op);
}

ImageBatch Convolution::SoftmaxBackward(ImageBatch diff, ImageBatch top, SoftmaxAlgorithm algorithm) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs sizes mismatch";
  SoftmaxBackwardOp* op = new SoftmaxBackwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({diff, top}, diff.Size(), op);
}

ImageBatch Convolution::ActivationForward(ImageBatch src, ActivationAlgorithm algorithm) {
  ActivationForwardOp* op = new ActivationForwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({src}, src.Size(), op);
}

ImageBatch Convolution::ActivationBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, ActivationAlgorithm algorithm) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs sizes mismatch";
  CHECK_EQ(diff.Size(), bottom.Size()) << "inputs sizes mismatch";
  ActivationBackwardOp* op = new ActivationBackwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({diff, top, bottom}, diff.Size(), op);
}

ImageBatch Convolution::PoolingForward(ImageBatch src, PoolingInfo info) {
  //No such check
  //CHECK_EQ((src.GetHeight() - info.height) % info.stride_vertical, 0) << "window height mismatch";
  //CHECK_EQ((src.GetWidth() - info.width) % info.stride_horizontal, 0) << "window width mismatch";
  int pooled_height = static_cast<int>(ceil(static_cast<float>((src.GetHeight() + 2 * info.pad_height - info.height)) / info.stride_vertical)) + 1;
  int pooled_width = static_cast<int>(ceil(static_cast<float>((src.GetWidth() + 2 * info.pad_width - info.width)) / info.stride_horizontal)) + 1;
  if (info.pad_height > 0 || info.pad_width > 0)
  {
	if((pooled_height - 1) * info.stride_vertical >= src.GetHeight() + info.pad_height)
		--pooled_height;
	if((pooled_width - 1) * info.stride_horizontal >= src.GetWidth() + info.pad_width)
		--pooled_width;
  }

	//std::cout << "Pooled " << pooled_height << " " << pooled_width << " " << info.pad_height << " " << info.stride_vertical << std::endl; 
	
  Scale new_size {
    pooled_height,
    pooled_width,
    src.GetNumFeatureMaps(),
    src.GetNumImages()
  };
  PoolingForwardOp* op = new PoolingForwardOp();
  op->closure = {
    info.algorithm,
    info.height,
    info.width,
    info.stride_vertical,
    info.stride_horizontal,
	info.pad_height,
	info.pad_width
  };
  return NArray::ComputeOne({src}, new_size, op);
}

ImageBatch Convolution::PoolingBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, PoolingInfo info) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs sizes mismatch";
  CHECK_EQ(diff.GetNumImages(), bottom.GetNumImages()) << "#images mismatch";
  CHECK_EQ(diff.GetNumFeatureMaps(), bottom.GetNumFeatureMaps()) << "#channels mismatch";
  
  int pooled_height = static_cast<int>(ceil(static_cast<float>((bottom.GetHeight() + 2 * info.pad_height - info.height)) / info.stride_vertical)) + 1;
  int pooled_width = static_cast<int>(ceil(static_cast<float>((bottom.GetWidth() + 2 * info.pad_width - info.width)) / info.stride_horizontal)) + 1;

  if (info.pad_height > 0 || info.pad_width > 0)
  {
	if((pooled_height - 1) * info.stride_vertical >= bottom.GetHeight() + info.pad_height)
		--pooled_height;
	if((pooled_width - 1) * info.stride_horizontal >= bottom.GetWidth() + info.pad_width)
		--pooled_width;
  }

  CHECK_EQ(top.GetHeight(), pooled_height) << "height mismatch";
  CHECK_EQ(top.GetWidth(), pooled_width) << "width mismatch";
  
  PoolingBackwardOp* op = new PoolingBackwardOp();
  op->closure = {
    info.algorithm,
    info.height,
    info.width,
    info.stride_vertical,
    info.stride_horizontal,
	info.pad_height,
	info.pad_width
  };
  return NArray::ComputeOne({diff, top, bottom}, bottom.Size(), op);
}


ImageBatch Convolution::LRNForward(ImageBatch src, ImageBatch scale, int local_size, float alpha, float beta)
{
  LRNForwardOp* op = new LRNForwardOp();
  op->closure = {local_size, alpha, beta, src.Size()};
  return NArray::ComputeOne({src, scale}, src.Size(), op);
}

ImageBatch Convolution::LRNBackward(ImageBatch bottom_data, ImageBatch top_data, ImageBatch scale, ImageBatch top_diff , int local_size, float alpha, float beta)
{
  LRNBackwardOp* op = new LRNBackwardOp();
  op->closure = {local_size, alpha, beta, bottom_data.Size()};
  return NArray::ComputeOne({bottom_data, top_data, scale, top_diff}, bottom_data.Size(), op);
}


}  // namespace minerva

