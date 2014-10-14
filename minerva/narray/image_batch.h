#pragma once
#include <string>
#include "narray/narray.h"

namespace minerva {

// `ImageBatch` should always be 4D
class ImageBatch : public NArray {
 public:
  ImageBatch(const ImageBatch&);
  ImageBatch(ImageBatch&&);
  ImageBatch(const NArray&);
  ImageBatch(NArray&&);
  ~ImageBatch();
  ImageBatch& operator=(const ImageBatch&);
  ImageBatch& operator=(ImageBatch&&);
  ImageBatch& operator=(const NArray&);
  ImageBatch& operator=(NArray&&);
  int GetNumImages() const;
  int GetNumFeatureMaps() const;
  int GetHeight() const;
  int GetWidth() const;

 private:
  ImageBatch();  // Forbidden
};

class Filter : public NArray {
 public:
  Filter(const Filter&);
  Filter(Filter&&);
  Filter(const NArray&);
  Filter(NArray&&);
  ~Filter();
  Filter& operator=(const Filter&);
  Filter& operator=(Filter&&);
  Filter& operator=(const NArray&);
  Filter& operator=(NArray&&);
  int GetNumOutputs() const;
  int GetNumInputs() const;
  int GetHeight() const;
  int GetWidth() const;

 private:
  Filter();  // Forbidden
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

