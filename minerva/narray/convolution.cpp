#include "narray/convolution.h"
#include "op/physical_op.h"

namespace minerva {

ImageBatch Convolution::ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info) {
  CHECK_EQ(src.GetNumFeatureMaps(), filter.GetNumInputs()) << "#input channels mismatch";
  CHECK_EQ(bias.Size().NumDims(), 1) << "bias dimension mismatch";
  CHECK_EQ(bias.Size()[0], filter.GetNumOutputs()) << "bias size mismatch";
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

ImageBatch SoftmaxForward(ImageBatch src, SoftmaxAlgorithm algorithm) {
  SoftmaxForwardOp* op = new SoftmaxForwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({src}, src.Size(), op);
}

ImageBatch SoftmaxBackward(ImageBatch diff, ImageBatch top, SoftmaxAlgorithm algorithm) {
  CHECK(diff.Size() == top.Size()) << "inputs sizes mismatch";
  SoftmaxForwardOp* op = new SoftmaxForwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({diff, top}, diff.Size(), op);
}

ImageBatch ActivationForward(ImageBatch src, ActivationAlgorithm algorithm) {
  ActivationForwardOp* op = new ActivationForwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({src}, src.Size(), op);
}

ImageBatch ActivationBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, ActivationAlgorithm algorithm) {
  CHECK(diff.Size() == top.Size() && diff.Size() == bottom.Size()) << "inputs sizes mismatch";
  ActivationBackwardOp* op = new ActivationBackwardOp();
  op->closure.algorithm = algorithm;
  return NArray::ComputeOne({diff, top, bottom}, diff.Size(), op);
}

}  // namespace minerva

