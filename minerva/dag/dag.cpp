#include <algorithm>
#include "dag.h"

using namespace std;

namespace minerva {

void DagNode::AddParent(DagNode* p) {
  p->successors_.insert(this);
  predecessors_.insert(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
  for (auto i: list) {
    AddParent(i);
  }
}

bool DagNode::DeleteParent(DagNode* p) {
  predecessors_.erase(std::find(predecessors_.begin(), predecessors_.end(), p));
  p->successors_.erase(std::find(p->successors_.begin(), p->successors_.end(), this));
  return predecessors_.empty();
}

}
