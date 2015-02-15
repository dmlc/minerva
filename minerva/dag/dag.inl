#include <sstream>
#include <glog/logging.h>
#include <mutex>

namespace minerva {

template<typename D, typename O>
Dag<D, O>::~Dag() {
  auto index_to_node = index_to_node_;
  for (auto i : index_to_node) {
    DeleteNode(i.first);
  }
}

template<typename D, typename O>
typename Dag<D, O>::DNode* Dag<D, O>::NewDataNode(const D& data) {
  std::lock_guard<std::recursive_mutex> lck(m_);
  DNode* ret = new DNode(NewIndex(), data);
  CHECK(index_to_node_.insert(std::make_pair(ret->node_id(), ret)).second);
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
  std::lock_guard<std::recursive_mutex> lck(m_);
  ONode* ret = new ONode(NewIndex());
  ret->op_ = op;
  CHECK(index_to_node_.insert(std::make_pair(ret->node_id(), ret)).second);
  for (auto mon : monitors_) {
    mon->OnCreateNode(ret);
  }
  for (auto in : inputs) {
    // There are possible duplicates in `inputs`
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
  std::lock_guard<std::recursive_mutex> lck(m_);
  DLOG(INFO) << "delete node #" << id;
  auto node = GetNode(id);
  for (auto succ : node->successors_) {
    CHECK_EQ(succ->predecessors_.erase(node), 1);
  }
  for (auto pred : node->predecessors_) {
    CHECK_EQ(pred->successors_.erase(node), 1);
  }
  for (auto mon : monitors_) {
    mon->OnDeleteNode(node);
  }
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
void Dag<D, O>::RegisterMonitor(DagMonitor<Dag<D, O>>* m) {
  monitors_.push_back(m);
}

template<typename D, typename O>
void Dag<D, O>::ClearMonitor() {
  monitors_.clear();
}

template<typename D, typename O>
template<typename NodePrinter>
std::string Dag<D, O>::PrintDagDot() const {
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
template<typename NodePrinter>
std::string Dag<D, O>::PrintDag() const {
  std::ostringstream ns, es;
  ns << "Nodes:" << std::endl;
  es << "Edges:" << std::endl;
  for (auto i : index_to_node_) {
    ns << i.first << ">>>>";
    if (i.second->Type() == DagNode::NodeType::kOpNode) {
      Dag<D, O>::ONode* onode = dynamic_cast<Dag<D, O>::ONode*>(i.second);
      ns << "type===o;;;" << NodePrinter::OpToString(onode->op_) << std::endl;
    } else {
      Dag<D, O>::DNode* dnode = dynamic_cast<Dag<D, O>::DNode*>(i.second);
      ns << "type===d;;;" << NodePrinter::DataToString(dnode->data_) << std::endl;
    }
    for (auto j : i.second->successors_) {
      es << i.first << " -> " << j->node_id() << std::endl;
    }
  }
  return ns.str() + es.str();
}

template<typename D, typename O>
uint64_t Dag<D, O>::NewIndex() {
  static uint64_t index_counter = 0;
  return index_counter++;
}

}  // end of namespace minerva
