#pragma once
#include "backend/backend_chunk.h"

namespace minerva {

class DagChunk : public BackendChunk {
 public:
  DagChunk();
  DagChunk(const DagChunk&);
  DagChunk& operator=(const DagChunk&);
  ~DagChunk();
  const Scale& shape() const;
};

}  // namespace minerva
