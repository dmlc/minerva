#pragma once

#include <unordered_map>
#include <unordered_set>
#include "dag_procedure.h"

namespace minerva {

class ExpandEngine : public LogicalDagProcedure, public LogicalDagMonitor {
 public:
  void Process(LogicalDag& dag, const std::vector<uint64_t>& nodes);
  bool IsExpanded(uint64_t lnode_id) const;
  const NVector<uint64_t>& GetPhysicalNodes(uint64_t) const;
  void OnDeleteDataNode(LogicalDataNode* );

  const std::unordered_set<uint64_t>& last_expanded_nodes() const {
    return last_expanded_nodes_;
  }

 private:
  void ExpandNode(LogicalDag& dag, uint64_t lnid);
  void MakeMapping(LogicalDag::DNode* ldnode, const NVector<Chunk>& chunks);
 private:
  std::unordered_map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
  std::unordered_set<uint64_t> last_expanded_nodes_;
};

}
