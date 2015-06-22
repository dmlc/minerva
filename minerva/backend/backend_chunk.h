#pragma once
#include "common/scale.h"

namespace minerva {

class BackendChunk {
 public:
  virtual ~BackendChunk() = default;
  virtual BackendChunk* ShallowCopy() const = 0;
  virtual const Scale& shape() const = 0;
  virtual uint64_t GetDeviceId() const = 0;
};

}  // namespace minerva

