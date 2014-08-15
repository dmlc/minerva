#pragma once
#include "dag_engine.h"
#include "system/data_store.h"

namespace minerva {

class PhysicalEngine : public DagEngine<PhysicalDag> {
 public:
  PhysicalEngine(ThreadPool& tp, DataStore& ds);
 protected:
  void FreeNodeResources(PhysicalDataNode* );
  void ProcessNode(DagNode* node);
 private:
  DataStore& data_store_;
};

}
