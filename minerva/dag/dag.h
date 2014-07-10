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
  void set_node_id(uint64_t id) { node_id_ = id; };
  uint64_t node_id() { return node_id_; };

  virtual NodeTypes Type() const = 0;

 protected:
  uint64_t node_id_;
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
  // setter/getter
  Data& data() { return data_; }
  const Data& data() const { return data_; }
  void set_data(const Data& d) { data_ = d; }
 private:
  Data data_;
};

template<typename Data, typename Op>
class OpNode : public DagNode {
  typedef DataNode<Data, Op> DNode;
 public:
  OpNode() {}
  NodeTypes Type() const { return DagNode::OP_NODE; }
  // setter/getter
  void set_inputs(const std::vector<DNode*>& in) { inputs_ = in; }
  const std::vector<DNode*>& inputs() { return inputs_; }
  void set_outputs(const std::vector<DNode*>& out) { outputs_ = out; }
  const std::vector<DNode*>& outputs() { return outputs_; }
  Op& op() { return op_; }
  const Op& op() const { return op_; }
  void set_op(const Op& op) { op_ = op; }
 private:
  Op op_;
  std::vector<DNode*> inputs_, outputs_;
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
  uint64_t NewIndex();
  DISALLOW_COPY_AND_ASSIGN(Dag);
  std::map<uint64_t, DagNode*> index_to_node_;
};

} // end of namespace minerva

#include "dag.inl"
