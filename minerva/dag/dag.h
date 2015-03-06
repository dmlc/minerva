#pragma once
#include <algorithm>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <functional>
#include <glog/logging.h>
#include <string>
#include "dag/dag_node.h"
#include "common/common.h"

namespace minerva {

/* The DAG here is lock-free. To manipulate locks, put it into nodes.
 */
template<typename Data, typename Op>
class Dag {
 public:
  typedef DataNode<Data, Op> DNode;
  typedef OpNode<Data, Op> ONode;
  typedef std::unordered_map<uint64_t, DagNode*> ContainerType;
  Dag();
  DISALLOW_COPY_AND_ASSIGN(Dag);
  ~Dag();
  DNode* NewDataNode(const Data& data);
  ONode* NewOpNode(
      const std::vector<DNode*>&,
      const std::vector<DNode*>&, const Op&);
  void DeleteNode(uint64_t);
  DagNode* GetNode(uint64_t) const;
  ONode* GetOpNode(uint64_t) const;
  DNode* GetDataNode(uint64_t) const;
  size_t NumNodes() const;
  std::string ToDotString(
      std::function<std::string(const Data&)>,
      std::function<std::string(const Op&)>) const;
  std::string ToDotString() const;
  std::string ToString(
      std::function<std::string(const Data&)>,
      std::function<std::string(const Op&)>) const;
  std::string ToString() const;

 private:
  uint64_t NewIndex();
  ContainerType index_to_node_;
};

template<typename D, typename O>
Dag<D, O>::Dag() {
}

template<typename D, typename O>
Dag<D, O>::~Dag() {
  auto index_to_node = index_to_node_;
  for (auto i : index_to_node) {
    DeleteNode(i.first);
  }
}

template<typename D, typename O>
typename Dag<D, O>::DNode* Dag<D, O>::NewDataNode(const D& data) {
  DNode* ret = new DNode(NewIndex(), data);
  CHECK(index_to_node_.insert(std::make_pair(ret->node_id(), ret)).second);
  return ret;
}

template<typename D, typename O>
typename Dag<D, O>::ONode* Dag<D, O>::NewOpNode(
    const std::vector<DNode*>& inputs,
    const std::vector<DNode*>& outputs,
    const O& op) {
  ONode* ret = new ONode(NewIndex());
  ret->op_ = op;
  CHECK(index_to_node_.insert(std::make_pair(ret->node_id(), ret)).second);
  Iter(inputs, ret->AddParent);
  Iter(outputs, [&](DNode* output) {
    CHECK(output->AddParent(ret));
  });
  ret->inputs_ = inputs;
  ret->outputs_ = outputs;
  return ret;
}

template<typename D, typename O>
void Dag<D, O>::DeleteNode(uint64_t id) {
  DLOG(INFO) << "delete node #" << id;
  auto node = GetNode(id);
  Iter(node->successors_, [&](DagNode* succ) {
    CHECK_EQ(succ->predecessors_.erase(node), 1);
  });
  Iter(node->predecessors_, [&](DagNode* pred) {
    CHECK_CQ(pred->successors_.erase(node), 1);
  });
  CHECK_EQ(index_to_node_.erase(id), 1);
  delete node;
}

template<typename D, typename O>
DagNode* Dag<D, O>::GetNode(uint64_t nid) const {
  return index_to_node_.at(nid);
}

template<typename D, typename O>
typename Dag<D, O>::ONode* Dag<D, O>::GetOpNode(uint64_t nid) const {
  return CHECK_NOTNULL(dynamic_cast<ONode*>(GetNode(nid)));
}

template<typename D, typename O>
typename Dag<D, O>::DNode* Dag<D, O>::GetDataNode(uint64_t nid) const {
  return CHECK_NOTNULL(dynamic_cast<DNode*>(GetNode(nid)));
}

template<typename D, typename O>
size_t Dag<D, O>::NumNodes() const {
  return index_to_node_.size();
}

template<typename D, typename O>
std::string Dag<D, O>::ToDotString(
    std::function<std::string(const D& data_to_string)>,
    std::function<std::string(const O& op_to_string)>) const {
  std::ostringstream out;
  out << "digraph G {" << std::endl;
  for (auto i : index_to_node_) {
    out << "  " << i.first << " [shape=";
    if (i.second->Type() == DagNode::NodeType::kOpNode) {
      out << "ellipse";
      ONode* onode = CHECK_NOTNULL(dynamic_cast<ONode*>(i.second));
      out << " label=\"#" << i.first << "|" << op_to_string(onode->op_) << "\"";
    } else {
      out << "box";
      DNode* dnode = CHECK_NOTNULL(dynamic_cast<DNode*>(i.second));
      out << " label=\"#" << i.first << "|" << data_to_string(dnode->data_)
        << "\"";
    }
    out << "];" << std::endl;
    for (auto j : i.second->successors_) {
      out << "  " << i.first << " -> " << j->node_id() << ";" << std::endl;
    }
  }
  out << "}";
  return out.str();
}

template<typename D, typename O>
std::string Dag<D, O>::ToDotString() const {
  ToDotString([](const D&) { return ""; }, [](const O&) { return ""; });
}

template<typename D, typename O>
std::string Dag<D, O>::ToString(
    std::function<std::string(const D& data_to_string)>,
    std::function<std::string(const O& op_to_string)>) const {
  std::ostringstream ns, es;
  ns << "Nodes:" << std::endl;
  es << "Edges:" << std::endl;
  for (auto i : index_to_node_) {
    ns << i.first << ">>>>";
    if (i.second->Type() == DagNode::NodeType::kOpNode) {
      ONode* onode = CHECK_NOTNULL(dynamic_cast<ONode*>(i.second));
      ns << "type===o;;;" << op_to_string(onode->op_) << std::endl;
    } else {
      DNode* dnode = CHECK_NOTNULL(dynamic_cast<DNode*>(i.second));
      ns << "type===d;;;" << data_to_string(dnode->data_) << std::endl;
    }
    for (auto j : i.second->successors_) {
      es << i.first << " -> " << j->node_id() << std::endl;
    }
  }
  return ns.str() + es.str();
}

template<typename D, typename O>
std::string Dag<D, O>::ToString() const {
  ToString([](const D&) { return ""; }, [](const O&) { return ""; });
}

template<typename D, typename O>
uint64_t Dag<D, O>::NewIndex() {
  static uint64_t index_counter = 0;
  return index_counter++;
}

}  // end of namespace minerva

