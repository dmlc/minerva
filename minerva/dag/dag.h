#pragma once
#include "dag/dag_node.h"
#include "dag/dag_monitor.h"
#include "dag/dag_helper.h"
#include "common/common.h"
#include <cstdint>
#include <initializer_list>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace minerva {

template<typename Data, typename Op>
class Dag {
 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  typedef std::unordered_map<uint64_t, DagNode*> ContainerType;
  Dag() {}
  ~Dag();
  DNode* NewDataNode(const Data& data);
  ONode* NewOpNode(const std::vector<DNode*>& inputs,
      const std::vector<DNode*>& outputs, const Op& op);
  void DeleteNode(uint64_t);
  DagNode* GetNode(uint64_t) const;
  ONode* GetOpNode(uint64_t) const;
  DNode* GetDataNode(uint64_t) const;
  size_t NumNodes() const;
  void RegisterMonitor(DagMonitor<Dag<Data, Op>>*);
  void ClearMonitor();
  template<typename NodePrinter=DagHelper<Data, Op>> std::string PrintDag() const;
  mutable std::recursive_mutex m_;

 private:
  uint64_t NewIndex();
  std::vector<DagMonitor<Dag<Data, Op>>*> monitors_;
  ContainerType index_to_node_;
  DISALLOW_COPY_AND_ASSIGN(Dag);
};

}  // end of namespace minerva

#include "dag.inl"

