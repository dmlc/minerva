#include "Chunk.h"
#include "DAGNode.h"

namespace minerva {

Chunk operator*(const Chunk& a, const Chunk& b) {
    // Check if operands match in dimension.
    OpNode* op = new OpNode();
    Chunk ret;
    // Set successors and predecessors
    op->mult;
    op->operands = {ret.storageIdx, a.storageIdx, b.storageIdx};
}

}

