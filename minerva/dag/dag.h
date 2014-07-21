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
  // TODO Use unordered version for quicker access
  std::set<DagNode*> successors_;
  std::set<DagNode*> predecessors_;
  uint64_t node_id_;
  virtual NodeTypes Type() const = 0;

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

template<class DagType> class DagProcedure;

template<class Data, class Op>
class Dag {
 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  Dag() {}
  ~Dag();
  DNode* NewDataNode(const Data& data);
  ONode* NewOpNode(std::vector<DNode*> inputs,
      std::vector<DNode*> outputs, const Op& op);
  std::string PrintDag() const;
  DagNode* GetNode(uint64_t nid) const {
    auto pos = index_to_node_.find(nid);
    return pos == index_to_node_.end() ? 0 : pos->second;
  }
  ONode* GetOpNode(uint64_t nid) const {
    return dynamic_cast<ONode*>(GetNode(nid));
  }
  DNode* GetDataNode(uint64_t nid) const {
    return dynamic_cast<DNode*>(GetNode(nid));
  }
  std::map<uint64_t, DagNode*> index_to_node_;

 private:
  DISALLOW_COPY_AND_ASSIGN(Dag);
  uint64_t NewIndex();
};

template<typename Data, typename Op>
class DagHelper {
 public:
  static std::string DataToString(const Data& d) {
    return "N/A";
  }
  static std::string OpToString(const Op& o) {
    return "N/A";
  }
};

} // end of namespace minerva

#include "dag.inl"

