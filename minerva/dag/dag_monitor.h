#pragma once
#include "dag/dag_node.h"

namespace minerva {

template<typename DagType>
class DagMonitor {
 public:
  virtual void OnCreateNode(DagNode*) = 0;
  virtual void OnDeleteNode(DagNode*);
  virtual void OnCreateEdge(DagNode*, DagNode*);
  virtual void OnBeginModify();
  virtual void OnFinishModify();
};

}

