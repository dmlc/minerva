#include <cstdint>
#include <functional>
#include <queue>
#include <sstream>

namespace minerva {

template<class D, class O>
Dag<D, O>::~Dag() {
  for (auto i: index_to_node_) {
    delete i.second;
  }
}

template<class D, class O>
typename Dag<D, O>::DNode* Dag<D, O>::NewDataNode(const D& data) {
  typedef Dag<D, O>::DNode DNode;
  DNode* ret = new DNode;
  ret->data_ = data;
  ret->node_id_ = NewIndex();
  index_to_node_.insert(std::make_pair(ret->node_id_, ret));
  return ret;
}

template<class D, class O>
typename Dag<D, O>::ONode* Dag<D, O>::NewOpNode(
    std::vector<DataNode<D, O>*> inputs,
    std::vector<DataNode<D, O>*> outputs, const O& op) {
  typedef OpNode<D, O> ONode;
  ONode* ret = new ONode;
  ret->op_ = op;
  ret->node_id_ = NewIndex();
  index_to_node_.insert(std::make_pair(ret->node_id_, ret));
  for(auto in : inputs) {
    ret->AddParent(in);
  }
  for(auto out : outputs) {
    out->AddParent(ret);
  }
  return ret;
}

template<class D, class O>
uint64_t Dag<D, O>::NewIndex() {
  static uint64_t index_counter = 0;
  return index_counter++;
}

template<class D, class O>
std::string Dag<D, O>::PrintDag() const {
  std::ostringstream out;
  out << "digraph G {" << std::endl;
  for (auto i: index_to_node_) {
    out << "  " << i.first << " [shape=";
    if (i.second->Type() == DagNode::OP_NODE) {
      out << "box";
    } else {
      out << "ellipse";
    }
    out << "];" << std::endl;
    for (auto j: i.second->successors_) {
      out << "  " << i.first << " -> " << j->node_id_ << ";" << std::endl;
    }
  }
  out << "}";
  return out.str();
}

} // end of namespace minerva
