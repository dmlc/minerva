#pragma once
#include "common/scale.h"

namespace minerva {

class BackendChunk {
 public:
  BackendChunk() = default;
  BackendChunk(const BackendChunk&) = default;
  BackendChunk& operator=(const BackendChunk&) = default;
  virtual ~BackendChunk() = default;
  virtual const Scale& shape() const = 0;
};

}  // namespace minerva

