#pragma once

#include <vector>

#include "common/scale.h"
#include "common/nvector.h"
#include "chunk/chunk.h"
#include "dag.h"
#include "dag_context.h"

namespace minerva {

struct LogicalData;
class LogicalOp;
class OpExpander;

class Closure;

struct LogicalData {
  Scale size;
  //DataNodeContext context; // TODO how to set context ?
};

class LogicalOp {
 public:
  virtual std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) = 0;
  virtual ~LogicalOp() {}
};

template<class T>
class LogicalOpWithClosure : public LogicalOp {
 public:
  T closure;
  //OpNodeContext context; // TODO how to set context ?
};

typedef Dag<LogicalData, LogicalOp*> LogicalDag;
typedef LogicalDag::DNode LogicalDataNode;
typedef LogicalDag::ONode LogicalOpNode;

}// end of namespace minerva
