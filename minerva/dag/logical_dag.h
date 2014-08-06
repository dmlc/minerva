#pragma once

#include <sstream>

#include "dag.h"
#include "op/logical.h"

namespace minerva {

template<>
class DagHelper<LogicalData, LogicalOp> {
 public:
  static std::string DataToString(const LogicalData& d) {
    std::stringstream ss;
    ss << d.size; 
    if(d.data_gen_fn != NULL) {
      ss << d.data_gen_fn->Name();
    }
    return ss.str();
  }
  static std::string OpToString(const LogicalOp& o) {
    return o.compute_fn->Name();
  }
  static void FreeData(LogicalData& d) {
    if(d.data_gen_fn != NULL)
      delete d.data_gen_fn;
  }
  static void FreeOp(LogicalOp& o) {
    delete o.compute_fn;
  }
};

typedef Dag<LogicalData, LogicalOp> LogicalDag;
typedef LogicalDag::DNode LogicalDataNode;
typedef LogicalDag::ONode LogicalOpNode;
class LogicalDagMonitor : public DagMonitor<LogicalDag> {};

}
