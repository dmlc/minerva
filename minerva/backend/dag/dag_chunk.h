#pragma once
#include "backend/backend_chunk.h"
#include "dag/physical_dag.h"

namespace minerva {

class DagChunk : public BackendChunk {
 public:
  DagChunk() = delete;
  DagChunk(const DagChunk&);
  DagChunk(PhysicalDataNode*);
  DagChunk& operator=(const DagChunk&);
  ~DagChunk();
  const Scale& shape() const override;
  PhysicalDataNode const* node_;
};

}  // namespace minerva
