#pragma once
#include "backend/backend_chunk.h"
#include "dag/physical_dag.h"

namespace minerva {

class DagChunk : public BackendChunk {
 public:
  DagChunk();
  DagChunk(const DagChunk&);
  DagChunk& operator=(const DagChunk&);
  ~DagChunk();
  const Scale& shape() const;

 private:
  PhysicalDataNode* node_;
};

}  // namespace minerva
