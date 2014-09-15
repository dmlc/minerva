#pragma once
#include "common/common.h"
#include <cstdint>
#include <initializer_list>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace minerva {

class DagNode {
 public:
  enum class NodeType {
    kOpNode = 0,
    kDataNode
  };
  virtual ~DagNode();
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  bool DeleteSucc(DagNode*);
  bool DeletePred(DagNode*);
  virtual NodeType Type() const = 0;
  uint64_t node_id() const {
    return node_id_;
  }
  std::unordered_set<DagNode*> successors_;
  std::unordered_set<DagNode*> predecessors_;

 protected:
  DagNode(uint64_t id): node_id_(id) {}

 private:
  DISALLOW_COPY_AND_ASSIGN(DagNode);
  uint64_t node_id_;
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode(uint64_t id): DagNode(id) {}
  ~DataNode();
  NodeType Type() const { return DagNode::NodeType::kDataNode; }
  Data data_;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataNode);
};

template<typename Data, typename Op>
class OpNode : public DagNode {
 public:
  OpNode(uint64_t id): DagNode(id) {}
  ~OpNode();
  NodeType Type() const { return DagNode::NodeType::kOpNode; }
  Op op_;
  std::vector<DataNode<Data, Op>*> inputs_;
  std::vector<DataNode<Data, Op>*> outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
};

template<typename Data, typename Op>
class DagHelper {
 public:
  static std::string DataToString(const Data&) {
    return "N/A";
  }
  static std::string OpToString(const Op&) {
    return "N/A";
  }
  static void FreeData(Data&) {}
  static void FreeOp(Op&) {}
};

template<typename DagType>
class DagMonitor {
 public:
  virtual void OnCreateNode(DagNode*) = 0;
  virtual void OnDeleteNode(DagNode*);
  virtual void OnCreateDataNode(typename DagType::DNode*) {}
  virtual void OnCreateOpNode(typename DagType::ONode*) {}
  virtual void OnDeleteDataNode(typename DagType::DNode*) {}
  virtual void OnDeleteOpNode(typename DagType::ONode*) {}
  virtual void OnCreateEdge(DagNode*, DagNode*) {}
};

template<typename Data, typename Op>
class Dag {
 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  typedef std::unordered_map<uint64_t, DagNode*> ContainerType;
  Dag() {}
  ~Dag() {
    // TODO Not deleting nodes for now
  }
  DNode* NewDataNode(const Data& data);
  ONode* NewOpNode(const std::vector<DNode*>& inputs,
      const std::vector<DNode*>& outputs, const Op& op);
  void DeleteNode(uint64_t);
  DagNode* GetNode(uint64_t) const;
  ONode* GetOpNode(uint64_t) const;
  DNode* GetDataNode(uint64_t) const;
  size_t NumNodes() const;
  ContainerType::iterator begin() {
    return index_to_node_.begin();
  }
  ContainerType::iterator end() {
    return index_to_node_.end();
  }
  void RegisterMonitor(DagMonitor<Dag<Data, Op>>*);
  template<typename NodePrinter=DagHelper<Data, Op>> std::string PrintDag() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(Dag);
  uint64_t NewIndex();
  std::vector<DagMonitor<Dag<Data, Op>>*> monitors_;
  ContainerType index_to_node_;
};

} // end of namespace minerva

#include "dag.inl"

