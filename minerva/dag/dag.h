#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <initializer_list>
#include <atomic>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include "device/device_info.h"

namespace minerva {

class DagNode {
 public:
  enum NodeTypes {
    OP_NODE = 0,
    DATA_NODE
  };
  virtual ~DagNode() {}
  void AddParent(DagNode*);
  void AddParents(std::initializer_list<DagNode*>);
  bool DeleteSucc(DagNode*);
  bool DeletePred(DagNode*);
  virtual NodeTypes Type() const = 0;

  uint64_t node_id() const { return node_id_; }
  void set_node_id(uint64_t id) { node_id_ = id; }
  DeviceInfo device_info() const { return device_info_; }
  void set_device_info(DeviceInfo info) { device_info_ = info; }

  std::vector<DagNode*> successors_;
  std::vector<DagNode*> predecessors_;
 private:
  uint64_t node_id_;
  DeviceInfo device_info_;
};

template<typename Data, typename Op>
class DataNode : public DagNode {
 public:
  DataNode() {}
  ~DataNode();
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
  ~OpNode();
  NodeTypes Type() const { return DagNode::OP_NODE; }
  Op op_;
  std::vector<DNode*> inputs_, outputs_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OpNode);
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
  static void FreeData(Data& d) {}
  static void FreeOp(Op& o) {}
};

template<class DagType>
class DagMonitor {
 public:
  virtual void OnCreateNode(DagNode* );
  virtual void OnDeleteNode(DagNode* );
  virtual void OnCreateDataNode(typename DagType::DNode* ) {}
  virtual void OnCreateOpNode(typename DagType::ONode* ) {}
  virtual void OnDeleteDataNode(typename DagType::DNode* ) {}
  virtual void OnDeleteOpNode(typename DagType::ONode* ) {}
  virtual void OnCreateEdge(DagNode* from, DagNode* to) {}
};

template<class Data, class Op>
class Dag {
 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  Dag() {}
  ~Dag();
  DNode* NewDataNode(const Data& data, DeviceInfo info);
  ONode* NewOpNode(const std::vector<DNode*>& inputs,
      const std::vector<DNode*>& outputs, const Op& op, DeviceInfo info);
  void DeleteNode(uint64_t );
  bool ExistNode(uint64_t ) const;
  DagNode* GetNode(uint64_t ) const;
  ONode* GetOpNode(uint64_t ) const;
  DNode* GetDataNode(uint64_t ) const;
  size_t NumNodes() const;

  typedef std::unordered_map<uint64_t, DagNode*> ContainerType;
  ContainerType::iterator begin() { return index_to_node_.begin(); }
  ContainerType::iterator end() { return index_to_node_.end(); }

  void RegisterMonitor(DagMonitor<Dag<Data, Op>>* );
  template<class NodePrinter=DagHelper<Data, Op> >
  std::string PrintDag() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(Dag);
  uint64_t NewIndex();

 private:
  std::vector<DagMonitor<Dag<Data, Op>>*> monitors_;
  ContainerType index_to_node_;
};

} // end of namespace minerva

#include "dag.inl"
