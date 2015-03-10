#include "dag_chunk.h"

namespace minerva {

DagChunk::DagChunk(const DagChunk& d) : node_(d.node_) {
  // incr extern rf
}

DagChunk::DagChunk(PhysicalDataNode* node) : node_(node) {
  CHECK_NOTNULL(node_);
  // extern rf
}

DagChunk& DagChunk::operator=(const DagChunk& d) {
  if (this == &d) {
    return *this;
  }
  // decr extern rf
  node_ = d.node_;
  // incr extern rf
  return *this;
}

DagChunk::~DagChunk() {
  // decr extern rf
}

const Scale& DagChunk::shape() const {
  return node_->data_.size;
}

}  // namespace minerva

