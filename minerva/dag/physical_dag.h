#pragma once

#include "dag.h"
#include "op/physical.h"
#include "op/physical_data.h"
#include "op/physical_op.h"
#include <string>

namespace minerva {

template<>
class DagHelper<PhysicalData, PhysicalOp> {
 public:
  static std::string DataToString(const PhysicalData&);
  static std::string OpToString(const PhysicalOp&);
};

typedef Dag<PhysicalData, PhysicalOp> PhysicalDag;
typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}

