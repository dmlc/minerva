#pragma once
#include <unordered_set>
#include <vector>
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
  uint64_t node_id() const;
  std::unordered_set<DagNode*> successors_;
  std::unordered_set<DagNode*> predecessors_;

 protected:
  DagNode(uint64_t);

 private:
  uint64_t node_id_;
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode(uint64_t, const Data&);
  DISALLOW_COPY_AND_ASSIGN(DataNode);
  ~DataNode() = default;
  NodeType Type() const;
  Data data_;
};

template<typename Data, typename Op>
class OpNode : public DagNode {
 public:
  OpNode(uint64_t);
  DISALLOW_COPY_AND_ASSIGN(OpNode);
  ~OpNode() = default;  // TODO autodeletion
  NodeType Type() const;
  Op op_;
  std::vector<DataNode<Data, Op>*> inputs_;
  std::vector<DataNode<Data, Op>*> outputs_;
};

}  // namespace minerva

