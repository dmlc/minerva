#pragma once

#include "dag.h"
#include "op/physical.h"
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
    return o.compute_fn->Name();
  }
};

class DataIdPrinter {
 public:
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.data_id;
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return o.compute_fn->Name();
  }
};

class OffsetPrinter {
 public:
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.offset;
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return o.compute_fn->Name();
  }
};

typedef Dag<PhysicalData, PhysicalOp> PhysicalDag;
typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}
