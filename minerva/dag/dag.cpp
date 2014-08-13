#include <algorithm>
#include "dag.h"

using namespace std;

namespace minerva {

void DagNode::AddParent(DagNode* p) {
  p->successors_.push_back(this);
  predecessors_.push_back(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
  for (auto i: list) {
    AddParent(i);
  }
}

bool DagNodeCmp(DagNode* n1, DagNode* n2) { return n1 == n2; }

bool DagNode::DeleteSucc(DagNode* p) {
  successors_.erase(
      std::remove_if(successors_.begin(), successors_.end(),
        [p] (DagNode* n) { return n == p; }),
      successors_.end()
    );
  return successors_.empty();
}

bool DagNode::DeletePred(DagNode* p) {
  predecessors_.erase(
      std::remove_if(predecessors_.begin(), predecessors_.end(),
        [p] (DagNode* n) { return n == p; }),
      predecessors_.end()
    );
  return predecessors_.empty();
}

}
