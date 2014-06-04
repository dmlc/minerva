#pragma once

#include "common/nvector.h"
#include "dag/dag_node.h"

namespace minerva {

class Chunk;
class ChunkOp;

class Chunk {
  friend Chunk operator * (const Chunk& a, const Chunk& b);
  friend Chunk operator + (const Chunk& a, const Chunk& b);
 public:
  static Chunk Constant(const Index& size, float val);

 public:
  Chunk();
  Index Size() const { return data_node_->meta().size; }
  DataNode* data_node() const { return data_node_; }
  void Eval();

 public:
  void operator += (const Chunk& a);

 private:
  explicit Chunk(const Index& size);
  DataNode* data_node_; // Set up in constructor
};

} // end of namespace minerva
