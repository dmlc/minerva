#include <sstream>
#include <glog/logging.h>
#include <mutex>

namespace minerva {

template<typename D, typename O>
DataNode<D, O>::~DataNode() {
  DagHelper<D, O>::FreeData(data_);
}

template<typename D, typename O>
OpNode<D, O>::~OpNode() {
  DagHelper<D, O>::FreeOp(op_);
}

template<typename D, typename O>
typename Dag<D, O>::DNode* Dag<D, O>::NewDataNode(const D& data) {
  DNode* ret = new DNode(NewIndex());
  ret->data_ = data;
  CHECK_EQ(index_to_node_.insert(std::make_pair(ret->node_id(), ret)), 1);
  for (auto mon : monitors_) {
    mon->OnCreateNode(ret);
  }
  return ret;
}

template<typename D, typename O>
typename Dag<D, O>::ONode* Dag<D, O>::NewOpNode(
    const std::vector<DataNode<D, O>*>& inputs,
    const std::vector<DataNode<D, O>*>& outputs,
    const O& op) {
  ONode* ret = new ONode(NewIndex());
  ret->op_ = op;
  CHECK_EQ(index_to_node_.insert(std::make_pair(ret->node_id(), ret)), 1);
  for (auto mon : monitors_) {
    mon->OnCreateNode(ret);
  }
  for (auto in : inputs) {
    std::lock_guard<std::mutex>> lck(in->iterator_busy_);
    if (ret->AddParent(in)) {
      for (auto mon : monitors_) {
        mon->OnCreateEdge(in, ret);
      }
    }
  }
  for (auto out : outputs) {
    CHECK(out->AddParent(ret));
    for (auto mon : monitors_) {
      mon->OnCreateEdge(ret, out);
    }
  }
  ret->inputs_ = inputs;
  ret->outputs_ = outputs;
  return ret;
}

template<typename D, typename O>
void Dag<D, O>::DeleteNode(uint64_t id) {
  auto node = GetNode(id);
  for (auto mon : monitors_) {
    mon->OnDeleteNode(node);
  }
  CHECK_EQ(index_to_node_.erase(id), 1);
  delete node;
}

template<typename D, typename O>
DagNode* Dag<D, O>::GetNode(uint64_t nid) const {
  return index_to_node_.at(nid)->second;
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
void Dag<D, O>::RegisterMonitor(DagMonitor<Dag<D, O>>* m) {
  monitors_.push_back(m);
}

template<typename D, typename O>
template<typename NodePrinter>
std::string Dag<D, O>::PrintDag() const {
  std::ostringstream out;
  out << "digraph G {" << std::endl;
  for (auto i : index_to_node_) {
    out << "  " << i.first << " [shape=";
    if (i.second->Type() == DagNode::NodeType::kOpNode) {
      out << "ellipse";
      Dag<D, O>::ONode* onode = dynamic_cast<Dag<D, O>::ONode*>(i.second);
      out << " label=\"#" << i.first << "|" << NodePrinter::OpToString(onode->op_) << "\"";
    } else {
      out << "box";
      Dag<D, O>::DNode* dnode = dynamic_cast<Dag<D, O>::DNode*>(i.second);
      out << " label=\"#" << i.first << "|" << NodePrinter::DataToString(dnode->data_) << "\"";
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
uint64_t Dag<D, O>::NewIndex() {
  static uint64_t index_counter = 0;
  return index_counter++;
}

} // end of namespace minerva

