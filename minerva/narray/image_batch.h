#pragma once
#include <string>
#include "narray/narray.h"

namespace minerva {

class ImageBatch : public NArray {
 public:
  ImageBatch();
  ImageBatch(const ImageBatch&);
  ImageBatch(const NArray&);
  ~ImageBatch();
  ImageBatch& operator=(const ImageBatch&);
  int GetNumImages() const;
  int GetNumFeatureMaps() const;
  int GetHeight() const;
  int GetWidth() const;
};

class Filter : public NArray {
 public:
  Filter();
  Filter(const Filter&);
  Filter(const NArray&);
  ~Filter();
  Filter& operator=(const Filter&);
  int GetNumOutputs();
  int GetNumInputs();
  int GetHeight();
  int GetWidth();
};

struct ConvInfo {
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
  Algorithm algorithm;
  int height;
  int width;
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

}  // namespace minerva

