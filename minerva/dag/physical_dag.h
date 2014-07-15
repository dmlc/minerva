#pragma once

#include "dag.h"
#include "op/physical.h"

namespace minerva {

typedef Dag<PhysicalData, PhysicalOp> PhysicalDag;
typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}
