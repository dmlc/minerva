#pragma once

#include <unordered_map>
#include <unordered_set>
#include "dag_procedure.h"
#include "state.h"

namespace minerva {

class ExpandEngine : public LogicalDagProcedure, public LogicalDagMonitor {
 public:
  ExpandEngine(NodeStateMap<LogicalDag>& ns): node_states_(ns) {}
  void Process(LogicalDag& dag, const std::vector<uint64_t>& nodes);
  bool IsExpanded(uint64_t lnode_id) const;
  const NVector<uint64_t>& GetPhysicalNodes(uint64_t) const;

  void OnDeleteDataNode(LogicalDataNode* );
  void GCNodes(LogicalDag& dag);

 private:
  void ExpandNode(LogicalDag& dag, uint64_t lnid);
  void MakeMapping(LogicalDag::DNode* ldnode, NVector<Chunk>& chunks);

 private:
  NodeStateMap<LogicalDag>& node_states_;
  std::unordered_map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
};

}
