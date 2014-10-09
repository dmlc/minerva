#pragma once
#include <narray/narray.h>

namespace minerva {

class ImageBatch {
 public:
  NArray GetImage(int);

 private:
  NArray narray_;
  int num_images_;
  int num_feature_maps_;
  int height_;
  int width_;
};

class Filter {
 public:
  Filter();

 private:
  NArray narray_;
  int num_outputs_;
  int num_inputs_;
  int height_;
  int width_;
};

struct ConvInfo {
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
};

enum class SoftmaxAlgorithm {
  kInstance,
  kChannel
};

enum class ActivationAlgorithm {
  kSigmoid,
  kRelu,
  kTanh
};

struct PoolingInfo {
  enum Algorithm {
    kMax,
    kAverage
  };
  Algorithm algorithm;
  int height;
  int width;
  int stride_vertical;
  int stride_horizontal;
};

}  // namespace minerva

