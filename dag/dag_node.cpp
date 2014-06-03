#include "dag_node.h"
#include <initializer_list>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <thread>
#include <chrono>

using namespace std;

namespace minerva {

DagNode::DagNode() {
}

DagNode::~DagNode() {
}

void DagNode::AddParent(DagNode* p) {
  p->successors_.push_back(this);
  predecessors_.push_back(p);
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

uint64_t DataNode::data_id_gen_ = 0;

void DataNode::Init() {
  data_id_ = data_id_gen_++;
}
}

bool DataNode::CreateCPUData() {
  return true;
}
bool DataNode::CreateGPUData() {
  return false;
}
float* DataNode::GetCPUData() {
  return NULL;
}
float* DataNode::GetGPUData() {
  return NULL;
}

OpNode::OpNode() {
  /*runner_ = [this] () {
    printf("Node %llu: Op Node\n", (unsigned long long) node_id_);
  };*/
}

OpNode::~OpNode() {
}

} // end of namespace minerva
