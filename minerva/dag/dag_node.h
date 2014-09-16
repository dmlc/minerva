#pragma once
#include <unordered_set>
#include <vector>
#include "common/common.h"
#include "dag/dag_helper.h"

namespace minerva {

class DagNode {
 public:
  enum class NodeType {
    kOpNode = 0,
    kDataNode
  };
  virtual ~DagNode() {}
  int AddParent(DagNode*);
  virtual NodeType Type() const = 0;
  uint64_t node_id() const {
    return node_id_;
  }
  std::unordered_set<DagNode*> successors_;
  std::unordered_set<DagNode*> predecessors_;

 protected:
  DagNode(uint64_t id) : node_id_(id) {}

 private:
  uint64_t node_id_;
  DISALLOW_COPY_AND_ASSIGN(DagNode);
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode(uint64_t id) : DagNode(id) {}
  ~DataNode() {
    DagHelper<Data, Op>::FreeData(data_);
  }
  NodeType Type() const {
    return DagNode::NodeType::kDataNode;
  }
  Data data_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
};

template<typename Data, typename Op>
class OpNode : public DagNode {
 public:
  OpNode(uint64_t id) : DagNode(id) {}
  ~OpNode() {
    DagHelper<Data, Op>::FreeOp(op_);
  }
  NodeType Type() const {
    return DagNode::NodeType::kOpNode;
  }
  Op op_;
  std::vector<DataNode<Data, Op>*> inputs_;
  std::vector<DataNode<Data, Op>*> outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

}  // namespace minerva

