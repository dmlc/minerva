#pragma once
#include <unordered_set>
#include <vector>
#include <mutex>
#include "common/common.h"

namespace minerva {

class DagNode {
 public:
  enum class NodeType {
    kOpNode = 0,
    kDataNode
  };
  DagNode() = delete;
  DISALLOW_COPY_AND_ASSIGN(DagNode);
  virtual ~DagNode() = default;
  int AddParent(DagNode*);
  virtual NodeType Type() const = 0;
  std::unordered_set<DagNode*> successors_;
  std::unordered_set<DagNode*> predecessors_;
  std::mutex m_;
  const uint64_t node_id_;

 protected:
  DagNode(uint64_t);
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode(uint64_t id, const Data& data) : DagNode(id), data_(data) {
  }
  DISALLOW_COPY_AND_ASSIGN(DataNode);
  ~DataNode() = default;
  NodeType Type() const override {
    return DagNode::NodeType::kDataNode;
  }
  Data data_;
};

template<typename Data, typename Op>
class OpNode : public DagNode {
 public:
  OpNode(uint64_t id) : DagNode(id) {
  }
  DISALLOW_COPY_AND_ASSIGN(OpNode);
  ~OpNode() = default;
  NodeType Type() const override {
    return DagNode::NodeType::kOpNode;
  }
  Op op_;
  std::vector<DataNode<Data, Op>*> inputs_;
  std::vector<DataNode<Data, Op>*> outputs_;
};

}  // namespace minerva

