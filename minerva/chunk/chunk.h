#pragma once

#include "common/scale.h"
#include "common/nvector.h"
#include "dag/physical.h"

namespace minerva {

class Chunk;
class ChunkOp;

class Chunk {
  friend Chunk operator * (const Chunk& a, const Chunk& b);
  friend Chunk operator + (const Chunk& a, const Chunk& b);
 public:
  static Chunk Constant(const Scale& size, float val);

 public:
  Chunk();
  Chunk(PhysicalDataNode* node);
  Chunk(const Chunk& other);
  Scale Size() const;
  int Size(int dim) const;
  PhysicalDataNode* data_node() const { return data_node_; }
  void operator += (const Chunk& a);
  Chunk& operator = (const Chunk& other);

 private:
  PhysicalDataNode* data_node_; // Set up in constructor
};

} // end of namespace minerva
