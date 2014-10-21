#include "narray/convolution.h"
#include "op/physical_op.h"

namespace minerva {

ImageBatch Convolution::ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info) {
  CHECK_EQ(src.GetNumFeatureMaps(), filter.GetNumInputs()) << "#input channels mismatch";
  CHECK_EQ(bias.Size().NumDims(), 1) << "bias dimension mismatch";
  CHECK_EQ(bias.Size()[0], filter.GetNumOutputs()) << "bias size mismatch";
  CHECK_EQ((src.GetHeight() + 2 * info.pad_height - filter.GetHeight()) % info.stride_vertical, 0) << "filter height mismatch";
  CHECK_EQ((src.GetWidth() + 2 * info.pad_width - filter.GetWidth()) % info.stride_horizontal, 0) << "filter width mismatch";
  Scale new_size {
    src.GetNumImages(),
    filter.GetNumOutputs(),
    (src.GetHeight() + 2 * info.pad_height - filter.GetHeight()) / info.stride_vertical + 1,
    (src.GetWidth() + 2 * info.pad_width - filter.GetWidth()) / info.stride_horizontal + 1
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

ImageBatch Convolution::ConvBackwardData(ImageBatch diff, Filter filter, ConvInfo info) {
  CHECK_EQ(diff.GetNumFeatureMaps(), filter.GetNumOutputs()) << "#output channels mismatch";
  Scale new_size {
    diff.GetNumImages(),
    filter.GetNumInputs(),
    (diff.GetHeight() - 1) * info.stride_vertical + filter.GetHeight() - 2 * info.pad_height,
    (diff.GetWidth() - 1) * info.stride_horizontal + filter.GetWidth() - 2 * info.pad_width
  };
  ConvBackwardDataOp* op = new ConvBackwardDataOp();
  op->closure = {
    info.pad_height,
    info.pad_width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({diff, filter}, new_size, op);
}

Filter Convolution::ConvBackwardFilter(ImageBatch diff, ImageBatch bottom, ConvInfo info) {
  CHECK_EQ(diff.GetNumImages(), bottom.GetNumImages()) << "#images mismatch";
  Scale new_size {
    diff.GetNumFeatureMaps(),
    bottom.GetNumFeatureMaps(),
    (diff.GetHeight() - 1) * info.stride_vertical - bottom.GetHeight() - 2 * info.pad_height,
    (diff.GetWidth() - 1) * info.stride_horizontal - bottom.GetWidth() + 2 * info.pad_width
  };
  ConvBackwardFilterOp* op = new ConvBackwardFilterOp();
  op->closure = {
    info.pad_height,
    info.pad_width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({diff, bottom}, new_size, op);
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
  SoftmaxForwardOp* op = new SoftmaxForwardOp();
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
  CHECK_EQ((src.GetHeight() - info.height) % info.stride_vertical, 0) << "window height mismatch";
  CHECK_EQ((src.GetWidth() - info.width) % info.stride_horizontal, 0) << "window width mismatch";
  Scale new_size {
    src.GetNumImages(),
    src.GetNumFeatureMaps(),
    (src.GetHeight() - info.height) / info.stride_vertical + 1,
    (src.GetWidth() - info.width) / info.stride_horizontal + 1
  };
  PoolingForwardOp* op = new PoolingForwardOp();
  op->closure = {
    info.algorithm,
    info.height,
    info.width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({src}, new_size, op);
}

ImageBatch Convolution::PoolingBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, PoolingInfo info) {
  CHECK_EQ(diff.Size(), top.Size()) << "inputs sizes mismatch";
  CHECK_EQ(diff.GetNumImages(), bottom.GetNumImages()) << "#images mismatch";
  CHECK_EQ(diff.GetNumFeatureMaps(), bottom.GetNumFeatureMaps()) << "#channels mismatch";
  int bottom_height = (diff.GetHeight() - 1) * info.stride_vertical + info.height;
  int bottom_width = (diff.GetWidth() - 1) * info.stride_horizontal + info.width;
  CHECK_EQ(bottom.GetHeight(), bottom_height) << "height mismatch";
  CHECK_EQ(bottom.GetWidth(), bottom_width) << "width mismatch";
  PoolingBackwardOp* op = new PoolingBackwardOp();
  op->closure = {
    info.algorithm,
    info.height,
    info.width,
    info.stride_vertical,
    info.stride_horizontal
  };
  return NArray::ComputeOne({diff, top, bottom}, bottom.Size(), op);
}

}  // namespace minerva

