#include "dag.h"
#include <algorithm>

using namespace std;

namespace minerva {

DagNode::~DagNode() {
  for (auto succ : successors_) {
    succ->predecessors_.erase(this);
  }
  for (auto pred : predecessors_) {
    pred->successors_.erase(this);
  }
}

void DagNode::AddParent(DagNode* p) {
  p->successors_.insert(this);
  predecessors_.insert(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
  for (auto i : list) {
    AddParent(i);
  }
}

bool DagNode::DeleteSucc(DagNode* p) {
  successors_.erase(p);
  return successors_.empty();
}

bool DagNode::DeletePred(DagNode* p) {
  predecessors_.erase(p);
  return predecessors_.empty();
}

}
