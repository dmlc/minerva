#pragma once

namespace minerva {

struct ConvInfo {
  ConvInfo(): pad_height(0), pad_width(0),
    stride_vertical(1), stride_horizontal(1) {}
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
};

struct PoolingInfo {
  enum Algorithm {
    kMax,
    kAverage
  };
  PoolingInfo(): algorithm(kMax),
    height(0), width(0), 
    stride_vertical(1), stride_horizontal(1), 
    pad_height(0), pad_width(0) {}
  Algorithm algorithm;
  int height;
  int width;
  int stride_vertical;
  int stride_horizontal;
  int pad_height;
  int pad_width;
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

}  // namespace minerva

