#pragma once
#include "op/impl/impl.h"
#include "dag_procedure.h"

namespace minerva {

class SimpleImplDecider : public PhysicalDagProcedure {
 public:
  SimpleImplDecider(ImplType type): type(type) {}
  virtual void Process(PhysicalDag& dag, NodeStateMap<PhysicalDag>& states,
      const std::vector<uint64_t>&) {
    auto birth_node_set = states.GetNodesOfState(NodeState::kBirth);
    for(uint64_t nid : birth_node_set) {
      DagNode* node = dag.GetNode(nid);
      if(node->Type() == DagNode::OP_NODE) {
        PhysicalOpNode* onode = dynamic_cast<PhysicalOpNode*>(node);
        onode->op_.impl_type = type;
      }
    }
  }
 private:
  ImplType type;
};

namespace decider {
}

}
