#pragma once
#include <string>
#include "narray/narray.h"

namespace minerva {

// `ImageBatch` should always be 4D
class ImageBatch final : public NArray {
 public:
  ImageBatch() = delete;  // Forbidden
  ImageBatch(const ImageBatch&);
  ImageBatch(ImageBatch&&);
  ImageBatch(NArray);
  ~ImageBatch();
  ImageBatch& operator=(const ImageBatch&);
  ImageBatch& operator=(ImageBatch&&);
  ImageBatch& operator=(NArray);
  int GetNumImages() const;
  int GetNumFeatureMaps() const;
  int GetHeight() const;
  int GetWidth() const;
};

class Filter final : public NArray {
 public:
  Filter() = delete;  // Forbidden
  Filter(const Filter&);
  Filter(Filter&&);
  Filter(NArray);
  ~Filter();
  Filter& operator=(const Filter&);
  Filter& operator=(Filter&&);
  Filter& operator=(NArray);
  int GetNumOutputs() const;
  int GetNumInputs() const;
  int GetHeight() const;
  int GetWidth() const;
};

}  // namespace minerva

