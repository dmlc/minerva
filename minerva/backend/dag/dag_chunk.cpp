#include "dag_chunk.h"
#include "system/minerva_system.h"
#include "backend/dag/dag_scheduler.h"

namespace minerva {

static void ExternRCUpdate(PhysicalDataNode* node, int delta) {
  if (MinervaSystem::IsAlive()) {
    dynamic_cast<DagScheduler&>(MinervaSystem::Instance().backend()).ExternRCUpdate(node, delta);
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

DagChunk* DagChunk::ShallowCopy() const {
  return new DagChunk(*this);
}

const Scale& DagChunk::shape() const {
  return node_->data_.size;
}

PhysicalDataNode* DagChunk::node() const {
  return node_;
}

}  // namespace minerva

