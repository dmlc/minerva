#pragma once

#include "common/singleton.h"
#include "dag/logical.h"
#include "dag/physical.h"
#include "system/data_store.h"

namespace minerva {

class MinervaSystem : public Singleton<MinervaSystem> {
 public:
  MinervaSystem() {}
  LogicalDag& logical_dag() { return logical_dag_; }
  PhysicalDag& physical_dag() { return physical_dag_; }
  DataStore& data_store() { return data_store_; }
 private:
  LogicalDag logical_dag_;
  PhysicalDag physical_dag_;
  DataStore data_store_;
};

} // end of namespace minerva
