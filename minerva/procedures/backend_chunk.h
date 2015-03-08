#pragma once
#include "common/scaleh"

namespace minerva {

class BackendChunk {
 public:
  BackendChunk() = default;
  virtual ~BackendChunk() = default;
  virtual const Scale& shape() const = 0;
};

}  // namespace minerva

