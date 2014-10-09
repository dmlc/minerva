#pragma once
#include <narray/image_batch.h>

namespace minerva {

class Convolution {
 public:
  static ImageBatch Forward(ImageBatch src, Filter filter, NArray bias, ConvInfo info);
  static ImageBatch BackwardData(ImageBatch diff, Filter filter, ConvInfo info);
  static Filter BackwardFilter(ImageBatch diff, ImageBatch bottom, ConvInfo info);
  static NArray BackwardBias(ImageBatch diff);
  static ImageBatch SoftmaxForward(ImageBatch src, SoftmaxAlgorithm algorithm);
  static ImageBatch SoftmaxBackward(ImageBatch diff, ImageBatch bottom, SoftmaxAlgorithm algorithm);
  static ImageBatch ActivationForward(ImageBatch src, ActivationAlgorithm algorithm);
  static ImageBatch ActivationBackward(ImageBatch diff, ImageBatch bottom, ActivationAlgorithm algorithm);
};

}  // namespace minerva

