#pragma once
#include <string>
#include <sstream>
#include "dag/dag.h"
#include "op/physical.h"
#include "op/physical_fn.h"

namespace minerva {

template<>
class DagHelper<PhysicalData, PhysicalOp> {
 public:
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.size;
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return o.compute_fn->Name();
  }
  static void FreeData(PhysicalData&) {
  }
  static void FreeOp(PhysicalOp& o) {
    delete o.compute_fn;
  }
};

typedef Dag<PhysicalData, PhysicalOp> PhysicalDag;
typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}  // namespace minerva
