#pragma once
#include "dag_engine.h"
#include "system/data_store.h"
#include "impl_decider.h"

namespace minerva {

class PhysicalEngine : public DagEngine<PhysicalDag> {
 public:
  PhysicalEngine(ThreadPool& tp, DataStore& ds);
  void SetImplDecider(ImplDecider* d) { impl_decider_ = d; }
 protected:
  void SetUpReadyNodeState(DagNode* );
  void FreeDataNodeRes(PhysicalDataNode* );
  void ProcessNode(DagNode* node);
  std::unordered_set<uint64_t> FindStartFrontier(PhysicalDag& dag, const std::vector<uint64_t>& targets);
 private:
  DataStore& data_store_;
  ImplDecider* impl_decider_;
};

}
