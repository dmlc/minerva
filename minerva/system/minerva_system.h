#pragma once

#include "common/singleton.h"
#include "dag/dag.h"
#include "system/data_store.h"

namespace minerva {

class MinervaSystem : public Singleton<MinervaSystem> {
 public:
  MinervaSystem() {}
  Dag& logic_dag() { return logic_dag_; }
  Dag& concrete_dag() { return concrete_dag_; }
  DataStore& data_store() { return data_store_; }
 private:
  Dag logic_dag_, concrete_dag_;
  DataStore data_store_;
};

} // end of namespace minerva
