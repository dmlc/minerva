#include "dag_chunk.h"

namespace minerva {

DagChunk::DagChunk() : node_(0) {
}

DagChunk::DagChunk(const DagChunk& d) : node_(d.node_) {
  CHECK_NOTNULL(node_);
  // incr extern rf
}

DagChunk& DagChunk::operator=(const DagChunk& d) {
  if (this == &d) {
    return *this;
  }
  // decr extern rf
  node_ = d.node_;
  if (node_) {
    // incr extern rf
  }
  return *this;
}

DagChunk::~DagChunk() {
  if (node_) {
    // decr extern rf
  }
}

const Scale& DagChunk::shape() const {
  return node_->data_.size;
}

}  // namespace minerva

