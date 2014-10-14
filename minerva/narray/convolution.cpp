#include "narray/convolution.h"
#include "op/physical_op.h"

namespace minerva {

ImageBatch Convolution::ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info) {
  CHECK_EQ(src.GetNumFeatureMaps(), filter.GetNumInputs()) << "num of inputs mismatch";
  CHECK_EQ(bias.Size().NumDims(), 1) << "bias dimension mismatch";
  CHECK_EQ(bias.Size()[0], filter.GetNumOutputs()) << "bias size mismatch";
  Scale new_size{
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
    info.stride_horizontal,
    src.GetNumImages(),
    filter.GetNumInputs(),
    filter.GetNumOutputs(),
    filter.GetHeight(),
    filter.GetWidth()
  };
  return NArray::ComputeOne({src, filter, bias}, new_size, op);
}

}  // namespace minerva

