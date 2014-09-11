#pragma once
#include "op/impl/impl.h"
#include "procedures/dag_procedure.h"
#include "procedures/dag_engine.h"

namespace minerva {

class ImplDecider {
 public:
  virtual void Decide(DagNode*, NodeStateMap&) = 0;
};

class SimpleImplDecider : public ImplDecider {
 public:
  explicit SimpleImplDecider(ImplType type) : type(type) {}
  virtual void Decide(DagNode* node, NodeStateMap& states) {
    if (node->Type() == DagNode::OP_NODE) {
      PhysicalOpNode* onode = dynamic_cast<PhysicalOpNode*>(node);
      onode->op_.impl_type = type;
    }
  }
 private:
  ImplType type;
};

}  // namespace minerva
