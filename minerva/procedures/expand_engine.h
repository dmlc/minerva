#pragma once

#include "dag_procedure.h"

namespace minerva {

class ExpandEngine : public LogicalDagProcedure {
 public:
  void Process(LogicalDag& dag, std::vector<uint64_t>& nodes);
  std::vector<uint64_t> GetPhysicalNodes(uint64_t);
 private:
  void ExpandNode(LogicalDag& dag, uint64_t lnid);
  void MakeMapping(LogicalDag::DNode* ldnode, const NVector<Chunk>& chunks);
 private:
  std::map<uint64_t, NVector<uint64_t>> lnode_to_pnode_;
};

}
