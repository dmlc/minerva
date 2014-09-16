#include "dag.h"
#include <algorithm>
#include <glog/logging.h>

using namespace std;

namespace minerva {

DagNode::~DagNode() {
  for (auto succ : successors_) {
    CHECK_EQ(succ->predecessors_.erase(this), 1);
  }
  for (auto pred : predecessors_) {
    CHECK_EQ(pred->successors_.erase(this), 1);
  }
}

int DagNode::AddParent(DagNode* p) {
  auto pred_insert_success = p->successors_.insert(this).second;
  auto this_insert_success = predecessors_.insert(p).second;
  CHECK_EQ(pred_insert_success, this_insert_success);
  return pred_insert_success;
}

}  // namespace minerva

