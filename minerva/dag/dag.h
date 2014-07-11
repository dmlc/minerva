#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <initializer_list>
#include <atomic>
#include <string>

#include "common/common.h"
#include "common/concurrent_blocking_queue.h"

namespace minerva {

class DagNode {
 public:
  enum NodeTypes {
    OP_NODE = 0,
    DATA_NODE
  };
  DagNode();
  virtual ~DagNode();
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  bool DeleteParent(DagNode*);
  // getters
  const std::set<DagNode*>& successors() const { return successors_; }
  std::set<DagNode*>& successors() { return successors_; }
  const std::set<DagNode*>& predecessors() const { return predecessors_; }
  std::set<DagNode*>& predecessors() { return predecessors_; }
  uint64_t node_id_;
  virtual NodeTypes Type() const = 0;

 protected:
  std::set<DagNode*> successors_;
  std::set<DagNode*> predecessors_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DagNode);
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode() {}
  NodeTypes Type() const { return DagNode::DATA_NODE; }
  Data data_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
};

template<typename Data, typename Op>
class OpNode : public DagNode {
  typedef DataNode<Data, Op> DNode;

 public:
  OpNode() {}
  NodeTypes Type() const { return DagNode::OP_NODE; }
  Op op_;
  std::vector<DNode*> inputs_, outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

template<typename Data, typename Op>
class Dag {
  friend class DagEngine;

 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  Dag() {}
  ~Dag();
  DNode* NewDataNode(const Data& data);
  ONode* NewOpNode(std::initializer_list<DNode*> inputs,
      std::initializer_list<DNode*> outputs, const Op& op);
  std::string PrintDag() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(Dag);
  uint64_t NewIndex();
  std::map<uint64_t, DagNode*> index_to_node_;
};

} // end of namespace minerva

#include "dag.inl"

