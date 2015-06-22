#pragma once
#include "backend/backend_chunk.h"
#include "dag/physical_dag.h"

namespace minerva {

class DagChunk final : public BackendChunk {
 public:
  DagChunk() = delete;
  DagChunk(PhysicalDataNode*);
  DagChunk(const DagChunk&);
  DagChunk& operator=(const DagChunk&);
  ~DagChunk();
  DagChunk* ShallowCopy() const override;
  const Scale& shape() const override;
  uint64_t GetDeviceId() const override;
  PhysicalDataNode* node() const;

 private:
  PhysicalDataNode* node_;
};

}  // namespace minerva
