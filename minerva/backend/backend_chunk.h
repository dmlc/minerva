#pragma once
#include "common/scale.h"

namespace minerva {

class BackendChunk {
 public:
  virtual ~BackendChunk() = default;
  virtual const Scale& shape() const = 0;
};

}  // namespace minerva

