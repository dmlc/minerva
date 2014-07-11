#pragma once

#include <vector>

#include "common/scale.h"
#include "common/nvector.h"
#include "chunk/chunk.h"
#include "dag.h"
#include "dag_context.h"

namespace minerva {

struct LogicalData;
struct LogicalOp;
class OpExpander;

struct LogicalData {
  Scale size;
  DataNodeContext context;
};

struct LogicalOp {
  void* closure;
  OpNodeContext context;
  OpExpander* expander;
};

class OpExpander {
 public:
  virtual std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs, LogicalOp& op) = 0;
};

typedef Dag<LogicalData, LogicalOp> LogicalDag;
typedef LogicalDag::DNode LogicalDataNode;
typedef LogicalDag::ONode LogicalOpNode;

}// end of namespace minerva
