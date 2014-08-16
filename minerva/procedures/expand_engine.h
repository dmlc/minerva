#pragma once
#include "dag_engine.h"
#include "physical_engine.h"

namespace minerva {

class ExpandEngine : public DagEngine<LogicalDag> {
 public:
  ExpandEngine(ThreadPool& tp): DagEngine<LogicalDag>(tp) {}
  bool IsExpanded(uint64_t lnode_id) const;
  const NVector<uint64_t>& GetPhysicalNodes(uint64_t) const;
  void CreateNodeState(DagNode* );
  void DeleteNodeState(DagNode* );
  void OnCreateEdge(DagNode* from, DagNode* to);
 protected:
  void ProcessNode(DagNode* node);
  std::unordered_set<uint64_t> FindStartFrontier(LogicalDag& dag, const std::vector<uint64_t>& targets);
  void FinalizeProcess();
 private:
  void MakeMapping(LogicalDag::DNode* ldnode, NVector<Chunk>& chunks);
 private:
  std::unordered_map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
  std::unordered_set<uint64_t> start_frontier_, non_froniter_;
};

}
