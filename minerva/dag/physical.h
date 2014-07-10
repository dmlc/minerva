#pragma once

#include <vector>

#include "common/scale.h"
#include "dag.h"
#include "dag_context.h"

namespace minerva {

struct PhysicalData;
struct PhysicalOp;
class OpExecutor;

struct PhysicalData {
  Scale size, offset, chunk_index;
  DataNodeContext context;
  uint64_t data_id;
};

struct PhysicalOp {
  void* closure;
  OpNodeContext context;
  OpExecutor* executor;
};

class OpExecutor {
 public:
  virtual void Execute(std::vector<PhysicalData> inputs, std::vector<PhysicalData> outputs, PhysicalOp& op);
};

typedef Dag<PhysicalData, PhysicalOp> PhysicalDag;
typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}
