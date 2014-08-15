#pragma once

#include <unordered_map>
#include <unordered_set>
#include "dag_procedure.h"
#include "state.h"

namespace minerva {

class ExpandEngine : public LogicalDagProcedure, public LogicalDagMonitor {
 public:
  ExpandEngine() {}
  void Process(LogicalDag& dag, NodeStateMap<LogicalDag>&,
      const std::vector<uint64_t>& nodes);
  bool IsExpanded(uint64_t lnode_id) const;
  const NVector<uint64_t>& GetPhysicalNodes(uint64_t) const;

  void OnDeleteDataNode(LogicalDataNode* );
  void GCNodes(LogicalDag& dag, NodeStateMap<LogicalDag>&);

 private:
  void ExpandNode(LogicalDag& dag, NodeStateMap<LogicalDag>&, uint64_t lnid);
  void MakeMapping(LogicalDag::DNode* ldnode, NVector<Chunk>& chunks);

 private:
  std::unordered_map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
};

}
