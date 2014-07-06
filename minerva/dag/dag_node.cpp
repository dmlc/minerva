#include <initializer_list>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <thread>
#include <chrono>

#include "dag_node.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

DagNode::DagNode() {
}

DagNode::~DagNode() {
}

void DagNode::AddParent(DagNode* p) {
  p->successors_.insert(this);
  predecessors_.insert(p);
}

void DagNode::AddParents(initializer_list<DagNode*> list) {
  for (auto i: list) {
    AddParent(i);
  }
}

bool DagNode::DeleteParent(DagNode* p) {
  predecessors_.erase(std::find(predecessors_.begin(), predecessors_.end(), p));
  p->successors_.erase(std::find(p->successors_.begin(), p->successors_.end(), this));
  return predecessors_.empty();
}

void DataNode::Init() {
  data_id_ = MinervaSystem::Instance().data_store().GenerateDataID();
}

OpNode::OpNode() {
  /*runner_ = [this] () {
    printf("Node %llu: Op Node\n", (unsigned long long) node_id_);
  };*/
}

OpNode::~OpNode() {
}

} // end of namespace minerva
