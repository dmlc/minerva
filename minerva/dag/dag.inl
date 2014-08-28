#include <cstdint>
#include <functional>
#include <queue>
#include <sstream>
#include <glog/logging.h>
#include "device/device_info.h"

namespace minerva {

template<class D, class O>
DataNode<D, O>::~DataNode() {
  DagHelper<D, O>::FreeData(data_);
}

template<class D, class O>
OpNode<D, O>::~OpNode() {
  DagHelper<D, O>::FreeOp(op_);
}

template<class D, class O>
Dag<D, O>::~Dag() {
  // TODO currently no deletion in destructor
  /*for (auto i: index_to_node_) {
    delete i.second;
  }*/
}

template<class D, class O>
uint64_t Dag<D, O>::NewIndex() {
  static uint64_t index_counter = 0;
  return index_counter++;
}


template<class D, class O>
typename Dag<D, O>::DNode* Dag<D, O>::NewDataNode(const D& data) {
  typedef Dag<D, O>::DNode DNode;
  DNode* ret = new DNode;
  ret->data_ = data;
  ret->set_node_id(NewIndex());
  index_to_node_.insert(std::make_pair(ret->node_id(), ret));
  // notify monitors
  for(auto mon : monitors_) {
    mon->OnCreateNode(ret);
  }
  return ret;
}

template<class D, class O>
typename Dag<D, O>::ONode* Dag<D, O>::NewOpNode(
    const std::vector<DataNode<D, O>*>& inputs,
    const std::vector<DataNode<D, O>*>& outputs,
    const O& op,
    const DeviceInfo device_info) {
  typedef OpNode<D, O> ONode;
  ONode* ret = new ONode;
  ret->op_ = op;
  ret->set_node_id(NewIndex());
  ret->set_device_info(device_info);
  index_to_node_.insert(std::make_pair(ret->node_id(), ret));

  // notify monitors
  for(auto mon : monitors_) {
    mon->OnCreateNode(ret);
  }
  for(auto in : inputs) {
    ret->AddParent(in);
    // notify monitors
    for(auto mon : monitors_) {
      mon->OnCreateEdge(in, ret);
    }
  }
  for(auto out : outputs) {
    out->AddParent(ret);
    // notify monitors
    for(auto mon : monitors_) {
      mon->OnCreateEdge(ret, out);
    }
  }
  ret->inputs_ = inputs;
  ret->outputs_ = outputs;

  return ret;
}

template<class D, class O>
void Dag<D, O>::DeleteNode(uint64_t id) {
  DagNode* node = GetNode(id);
  // notify the monitors
  for(auto mon : monitors_) {
    mon->OnDeleteNode(node);
  }
  // delete the node in successors
  for(DagNode* succ : node->successors_) {
    succ->DeletePred(node);
  }
  // delete the node in predecessors
  for(DagNode* pred : node->predecessors_) {
    pred->DeleteSucc(node);
  }
  // delete the node in container
  index_to_node_.erase(id);
  delete node;
}

template<class D, class O>
bool Dag<D, O>::ExistNode(uint64_t id) const {
  return index_to_node_.find(id) != index_to_node_.end();
}

template<class D, class O>
DagNode* Dag<D, O>::GetNode(uint64_t nid) const {
  CHECK(ExistNode(nid)) << "nid=" << nid << " not found in dag!";
  return index_to_node_.find(nid)->second;
}

template<class D, class O>
typename Dag<D, O>::ONode* Dag<D, O>::GetOpNode(uint64_t nid) const {
  return CHECK_NOTNULL(dynamic_cast<ONode*>(GetNode(nid)));
}

template<class D, class O>
typename Dag<D, O>::DNode* Dag<D, O>::GetDataNode(uint64_t nid) const {
  return CHECK_NOTNULL(dynamic_cast<DNode*>(GetNode(nid)));
}

template<class D, class O>
size_t Dag<D, O>::NumNodes() const {
  return index_to_node_.size();
}

template<class D, class O>
void Dag<D, O>::RegisterMonitor(DagMonitor<Dag<D, O>>* m) {
  monitors_.push_back(m);
}

template<class D, class O>
template<class NodePrinter>
std::string Dag<D, O>::PrintDag() const {
  std::ostringstream out;
  out << "digraph G {" << std::endl;
  for (auto i: index_to_node_) {
    out << "  " << i.first << " [shape=";
    if (i.second->Type() == DagNode::OP_NODE) {
      out << "ellipse";
      Dag<D, O>::ONode* onode = dynamic_cast<Dag<D, O>::ONode*>(i.second);
      out << " label=\"#" << i.first << "|" << NodePrinter::OpToString(onode->op_) << "\"";
    } else {
      out << "box";
      Dag<D, O>::DNode* dnode = dynamic_cast<Dag<D, O>::DNode*>(i.second);
      out << " label=\"#" << i.first << "|" << NodePrinter::DataToString(dnode->data_) << "\"";
    }
    out << "];" << std::endl;
    for (auto j: i.second->successors_) {
      out << "  " << i.first << " -> " << j->node_id() << ";" << std::endl;
    }
  }
  out << "}";
  return out.str();
}

template<class DagType>
void DagMonitor<DagType>::OnCreateNode(DagNode* node) {
  switch(node->Type()) {
    case DagNode::OP_NODE:
      OnCreateOpNode(dynamic_cast<typename DagType::ONode*>(node));
      break;
    case DagNode::DATA_NODE:
      OnCreateDataNode(dynamic_cast<typename DagType::DNode*>(node));
      break;
  };
}

template<class DagType>
void DagMonitor<DagType>::OnDeleteNode(DagNode* node) {
  switch(node->Type()) {
    case DagNode::OP_NODE:
      OnDeleteOpNode(dynamic_cast<typename DagType::ONode*>(node));
      break;
    case DagNode::DATA_NODE:
      OnDeleteDataNode(dynamic_cast<typename DagType::DNode*>(node));
      break;
  };
}

} // end of namespace minerva
