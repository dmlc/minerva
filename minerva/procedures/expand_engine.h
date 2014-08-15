#pragma once
#include "dag_engine.h"
#include "physical_engine.h"

namespace minerva {

class ExpandEngine : public DagEngine<LogicalDag> {
 public:
  ExpandEngine(ThreadPool& tp, PhysicalEngine& pe): DagEngine<LogicalDag>(tp), physical_engine_(pe) {}
  bool IsExpanded(uint64_t lnode_id) const;
  const NVector<uint64_t>& GetPhysicalNodes(uint64_t) const;
  void OnDeleteDataNode(LogicalDataNode* );
  void CommitExternRCChange();
 protected:
  void OnIncrExternalDep(LogicalDataNode*, int amount);
  void ProcessNode(DagNode* node);
 private:
  void MakeMapping(LogicalDag::DNode* ldnode, NVector<Chunk>& chunks);
 private:
  PhysicalEngine& physical_engine_;
  std::unordered_map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
};

}
