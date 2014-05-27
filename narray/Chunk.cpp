#include "Chunk.h"
#include "DAGNode.h"

namespace minerva {

Chunk operator*(const Chunk& a, const Chunk& b) {
    // Check if operands match in dimension.
    OpNode* op = new OpNode();
    Chunk ret;
    // Set successors and predecessors
    a.GetDataNode()->successors.push_back(op);
    b.GetDataNode()->successors.push_back(op);
    op->predecessors.push_back(a.GetDataNode());
    op->predecessors.push_back(b.GetDataNode());
    op->type = OpNode::mult;
    op->operands = {ret.GetDataNode()->storageIdx, a.GetDataNode()->storageIdx, b.GetDataNode()->storageIdx};
}

}

