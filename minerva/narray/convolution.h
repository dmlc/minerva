#pragma once
#include "narray/image_batch.h"
#include "narray/convolution_info.h"

namespace minerva {

class Convolution {
 public:
  static ImageBatch ConvForward(ImageBatch src, Filter filter, NArray bias, ConvInfo info);
  static ImageBatch ConvBackwardData(ImageBatch diff, ImageBatch bottom, Filter filter, ConvInfo info);
  static Filter ConvBackwardFilter(ImageBatch diff, ImageBatch bottom, Filter filter, ConvInfo info);
  static NArray ConvBackwardBias(ImageBatch diff);
  static std::vector<ConvFwdAlgoProfResult> ConvForwardFindAlgorithm(
    Scale const& src_shape
  , Scale const& filter_shape
  , ConvInfo info);
  static std::vector<ConvBwdFilterAlgoProfResult> ConvBackwardFilterFindAlgorithm(
    Scale const& top_shape
  , Scale const& bottom_shape
  , Scale const& filter_shape
  , ConvInfo info);
  static std::vector<ConvBwdDataAlgoProfResult> ConvBackwardDataFindAlgorithm(
    Scale const& top_shape
  , Scale const& bottom_shape
  , Scale const& filter_shape
  , ConvInfo info);
  static ImageBatch SoftmaxForward(ImageBatch src, SoftmaxAlgorithm algorithm);
  static ImageBatch SoftmaxBackward(ImageBatch diff, ImageBatch top, SoftmaxAlgorithm algorithm);
  static ImageBatch ActivationForward(ImageBatch src, ActivationAlgorithm algorithm);
  static ImageBatch ActivationBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, ActivationAlgorithm algorithm);
  static ImageBatch PoolingForward(ImageBatch src, PoolingInfo info);
  static ImageBatch PoolingBackward(ImageBatch diff, ImageBatch top, ImageBatch bottom, PoolingInfo info);

  static ImageBatch LrnForward(ImageBatch src, int local_size, float alpha, float beta, float k);
  static ImageBatch LrnBackward(ImageBatch top, ImageBatch top_diff, ImageBatch bottom, int local_size, float alpha, float beta, float k);

};

}  // namespace minerva

