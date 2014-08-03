#pragma once

#include "dag_procedure.h"

namespace minerva {

class ExpandEngine : public LogicalDagProcedure, public LogicalDagMonitor {
 public:
  void Process(LogicalDag& dag, const std::vector<uint64_t>& nodes);
  bool IsExpanded(uint64_t lnode_id) const;
  const NVector<uint64_t>& GetPhysicalNodes(uint64_t) const;
  void OnDeleteDataNode(LogicalDataNode* );
 private:
  void ExpandNode(LogicalDag& dag, uint64_t lnid);
  void MakeMapping(LogicalDag::DNode* ldnode, const NVector<Chunk>& chunks);
 private:
  std::map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
};

}
