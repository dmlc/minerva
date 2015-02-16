#pragma once
#include "dag/dag_node.h"

namespace minerva {

template<typename DagType>
class DagMonitor {
 public:
  virtual void OnCreateNode(DagNode*) = 0;
  virtual void OnDeleteNode(DagNode*) = 0;
  virtual void OnCreateEdge(DagNode*, DagNode*) = 0;
};

}
