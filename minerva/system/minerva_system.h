#pragma once

#include "common/singleton.h"
#include "dag/logical_dag.h"
#include "dag/physical_dag.h"
#include "system/data_store.h"
#include "procedures/expand_engine.h"
#include "narray/narray.h"

namespace minerva {

class MinervaSystem : public Singleton<MinervaSystem> {
 public:
  MinervaSystem() {}
  LogicalDag& logical_dag() { return logical_dag_; }
  PhysicalDag& physical_dag() { return physical_dag_; }
  DataStore& data_store() { return data_store_; }
  void Eval(NArray narr);
 private:
  LogicalDag logical_dag_;
  PhysicalDag physical_dag_;
  DataStore data_store_;
  ExpandEngine expand_engine_;
};

} // end of namespace minerva
