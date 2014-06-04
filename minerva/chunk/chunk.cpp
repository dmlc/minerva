#include "chunk.h"
#include "dag/dag.h"

namespace minerva {

Chunk::Chunk(): data_node_(NULL) {
}
Chunk::Chunk(const Index& size) {
}

Chunk operator*(const Chunk& a, const Chunk& b) {
  // Check if operands match in dimension.
  /*Chunk ret;
  OpNode* op = OpNode::CreateOpNode({a.data_node(), b.data_node()});
  // Set successors and predecessors
  a.GetDataNode()->successors.push_back(op);
  b.GetDataNode()->successors.push_back(op);
  op->predecessors.push_back(a.GetDataNode());
  op->predecessors.push_back(b.GetDataNode());
  op->type = OpNode::mult;
  op->operands = {ret.GetDataNode()->storageIdx, a.GetDataNode()->storageIdx, b.GetDataNode()->storageIdx};*/
  return Chunk();
}

} // end of namespace minerva
