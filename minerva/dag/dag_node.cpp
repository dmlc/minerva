#include "dag/dag_node.h"
#include <algorithm>
#include <glog/logging.h>

using namespace std;

namespace minerva {

uint64_t DagNode::node_id() const {
  return node_id_;
}

DagNode::DagNode(uint64_t id) : node_id_(id) {
}

int DagNode::AddParent(DagNode* p) {
  auto pred_insert_success = p->successors_.insert(this).second;
  auto this_insert_success = predecessors_.insert(p).second;
  // Either it has already been inserted, or not
  CHECK_EQ(pred_insert_success, this_insert_success);
  return pred_insert_success;
}

template<typename Data, typename Op>
DataNode::DataNode(uint64_t id, const Data& data) : DagNode(id), data_(data) {
}

template<typename Data, typename Op>
DagNode::NodeType DataNode::Type() const {
  return DagNode::NodeType::kDataNode;
}

template<typename Data, typename Op>
OpNode::OpNode(uint64_t id) : DagNode(id) {
}

template<typename Data, typename Op>
DagNode::NodeType OpNode::Type() const {
  return DagNode::NodeType::kOpNode;
}

}  // namespace minerva

