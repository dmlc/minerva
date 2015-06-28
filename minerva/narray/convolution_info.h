#pragma once
#include <cstdlib>

namespace minerva {

struct ConvInfo {
  enum class DataAlgorithm {
    kImplicitGemm,
    kImplicitPrecompGemm,
    kGemm,
    kDirect,
    kFft,
    kAuto
  };
  enum class FilterAlgorithm {
    kAlgo0,
    kAlgo1,
    kFft,
    kAuto
  };
  ConvInfo(int ph = 0, int pw = 0, int sv = 1, int sh = 1)
    : pad_height{ph}
    , pad_width{pw}
    , stride_vertical{sv}
    , stride_horizontal{sh} {
  }
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
  DataAlgorithm forward_algorithm = DataAlgorithm::kAuto;
  DataAlgorithm backward_data_algorithm = DataAlgorithm::kAuto;
  FilterAlgorithm backward_filter_algorithm = FilterAlgorithm::kAuto;
};

struct ConvFwdAlgoProfResult {
  ConvInfo::DataAlgorithm algo;
  float time;
  size_t memory;
};

struct PoolingInfo {
  enum class Algorithm {
    kMax,
    kAverage
  };
  PoolingInfo(
      Algorithm alg = Algorithm::kMax
    , int h = 0
    , int w = 0
    , int sv = 1
    , int sh = 1
    , int ph = 0
    , int pw = 0)
    : algorithm(alg)
    , height(h)
    , width(w)
    , stride_vertical(sv)
    , stride_horizontal(sh)
    , pad_height(ph)
    , pad_width(pw) {
  }
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

