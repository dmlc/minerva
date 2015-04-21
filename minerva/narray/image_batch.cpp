#include "narray/image_batch.h"
#include <memory>
#include <glog/logging.h>

namespace minerva {

ImageBatch::ImageBatch(const ImageBatch& b) : NArray(b) {
}

ImageBatch::ImageBatch(ImageBatch&& b) : NArray(std::move(b)) {
}

ImageBatch::ImageBatch(NArray n) : NArray(std::move(n)) {
  CHECK_EQ(Size().NumDims(), 4);
}

ImageBatch::~ImageBatch() = default;

ImageBatch& ImageBatch::operator=(const ImageBatch& b) {
  NArray::operator=(b);
  return *this;
}

ImageBatch& ImageBatch::operator=(ImageBatch&& b) {
  NArray::operator=(std::move(b));
  return *this;
}

ImageBatch& ImageBatch::operator=(NArray n) {
  CHECK_EQ(n.Size().NumDims(), 4);
  NArray::operator=(std::move(n));
  return *this;
}

int ImageBatch::GetNumImages() const {
  return Size(3);
}

int ImageBatch::GetNumFeatureMaps() const {
  return Size(2);
}

int ImageBatch::GetHeight() const {
  return Size(1);
}

int ImageBatch::GetWidth() const {
  return Size(0);
}

Filter::Filter(const Filter& f) : NArray(f) {
}

Filter::Filter(Filter&& f) : NArray(std::move(f)) {
}

Filter::Filter(NArray n) : NArray(std::move(n)) {
  CHECK_EQ(Size().NumDims(), 4);
}

Filter::~Filter() = default;

Filter& Filter::operator=(const Filter& f) {
  NArray::operator=(f);
  return *this;
}

Filter& Filter::operator=(Filter&& f) {
  NArray::operator=(std::move(f));
  return *this;
}

Filter& Filter::operator=(NArray n) {
  CHECK_EQ(n.Size().NumDims(), 4);
  NArray::operator=(std::move(n));
  return *this;
}

int Filter::GetNumOutputs() const {
  return Size(3);
}

int Filter::GetNumInputs() const {
  return Size(2);
}

int Filter::GetHeight() const {
  return Size(1);
}

int Filter::GetWidth() const {
  return Size(0);
}

}  // namespace minerva

