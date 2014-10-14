#include "narray/image_batch.h"
#include <memory>
#include <glog/logging.h>

namespace minerva {

ImageBatch::ImageBatch(const ImageBatch& b) : NArray(b) {
}

ImageBatch::ImageBatch(ImageBatch&& b) : NArray(std::move(b)) {
}

ImageBatch::ImageBatch(const NArray& n) : NArray(n) {
  CHECK_EQ(Size().NumDims(), 4);
}

ImageBatch::ImageBatch(NArray&& n) : NArray(std::move(n)) {
  CHECK_EQ(Size().NumDims(), 4);
}

ImageBatch::~ImageBatch() {
}

ImageBatch& ImageBatch::operator=(const ImageBatch& b) {
  if (this == &b) {
    return *this;
  }
  return *this = b;
}

ImageBatch& ImageBatch::operator=(ImageBatch&& b) {
  if (this == &b) {
    return *this;
  }
  return *this = std::move(b);
}

ImageBatch& ImageBatch::operator=(const NArray& n) {
  if (this == &n) {
    return *this;
  }
  CHECK_EQ(n.Size().NumDims(), 4);
  return *this = n;
}

ImageBatch& ImageBatch::operator=(NArray&& n) {
  if (this == &n) {
    return *this;
  }
  CHECK_EQ(n.Size().NumDims(), 4);
  return *this = std::move(n);
}

int ImageBatch::GetNumImages() const {
  return Size(0);
}

int ImageBatch::GetNumFeatureMaps() const {
  return Size(1);
}

int ImageBatch::GetHeight() const {
  return Size(2);
}

int ImageBatch::GetWidth() const {
  return Size(3);
}

Filter::Filter(const Filter& f) : NArray(f) {
}

Filter::Filter(Filter&& f) : NArray(std::move(f)) {
}

Filter::Filter(const NArray& n) : NArray(n) {
  CHECK_EQ(Size().NumDims(), 4);
}

Filter::Filter(NArray&& n) : NArray(std::move(n)) {
  CHECK_EQ(Size().NumDims(), 4);
}

Filter::~Filter() {
}

Filter& Filter::operator=(const Filter& f) {
  if (this == &f) {
    return *this;
  }
  return *this = f;
}

Filter& Filter::operator=(Filter&& f) {
  if (this == &f) {
    return *this;
  }
  return *this = std::move(f);
}

Filter& Filter::operator=(const NArray& n) {
  if (this == &n) {
    return *this;
  }
  CHECK_EQ(n.Size().NumDims(), 4);
  return *this = n;
}

Filter& Filter::operator=(NArray&& n) {
  if (this == &n) {
    return *this;
  }
  CHECK_EQ(n.Size().NumDims(), 4);
  return *this = std::move(n);
}

int Filter::GetNumOutputs() const {
  return Size(0);
}

int Filter::GetNumInputs() const {
  return Size(1);
}

int Filter::GetHeight() const {
  return Size(2);
}

int Filter::GetWidth() const {
  return Size(3);
}

}  // namespace minerva

