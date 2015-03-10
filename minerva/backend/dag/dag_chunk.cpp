#include "dag_chunk.h"
#include "system/minerva_system.h"

namespace minerva {

static ExternRCUpdate(PhysicalDataNode* node, int delta) {
  auto& ms = MinervaSystem::Instance();
  if (ms.IsAlive()) {
    CHECK_NOTNULL(dynamic_cast<DagScheduler*>(ms.backend()))->ExternRCUpdate(node, delta);
  }
}

DagChunk::DagChunk(PhysicalDataNode* node) : node_(node) {
  CHECK_NOTNULL(node_);
  ExternRCUpdate(node_, 1);
}

DagChunk::DagChunk(const DagChunk& d) : node_(d.node_) {
  ExternRCUpdate(node_, 1);
}

DagChunk& DagChunk::operator=(const DagChunk& d) {
  if (this == &d) {
    return *this;
  }
  ExternRCUpdate(node_, -1);
  node_ = d.node_;
  ExternRCUpdate(node_, 1);
  return *this;
}

DagChunk::~DagChunk() {
  ExternRCUpdate(node_, -1);
}

const Scale& DagChunk::shape() const {
  return node_->data_.size;
}

PhysicalDataNode* DagChunk::node() const {
  return node_;
}

}  // namespace minerva

