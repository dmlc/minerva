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
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.size;
    if(d.data_gen_fn != NULL) {
      ss << d.data_gen_fn->Name();
    }
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    // TODO
    return std::string("Not implemented yet");
  }
};

typedef Dag<PhysicalData, PhysicalOp> PhysicalDag;
typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}

