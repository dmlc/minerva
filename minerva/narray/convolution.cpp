#include "narray/convolution.h"

namespace minerva {

ImageBatch Convolution::ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info) {
  CHECK_EQ(src.GetNumFeatureMaps(), filter.GetNumInputs());
  CHECK_EQ(bias.Size().NumDims(), 1) << "bias dimension mismatch";
  CHECK_EQ(bias.Size()[0], filter.GetNumOutputs()) << "bias size mismatch";
  Scale new_size{
    src.GetNumImages(),
    filter.GetNumOutputs(),
    (src.GetHeight() + 2 * info.pad_height - filter.GetHeight()) / info.stride_vertical + 1,
    (src.GetWidth() + 2 * info.pad_width - filter.GetWidth()) / info.stride_horizontal + 1
  };
  ConvForwardOp* op = new ConvForwardOp();
  op->closure = info;
  return NArray::ComputeOne({src, filter, bias}, new_size, op);
}

}  // namespace minerva

