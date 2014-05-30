#include "dag.h"
#include "dag_node.h"
#include "common/concurrent_blocking_queue.h"
#include <cstdint>
#include <functional>
#include <queue>
#include <cstdio>
#include <thread>
#include <chrono>

using namespace std;

namespace minerva {

uint64_t Dag::index_counter_ = 0;

Dag::Dag() {
}

Dag::~Dag() {
  for (auto i: index_to_node_) {
    delete i.second;
  }
}

DataNode* Dag::NewDataNode() {
  DataNode* ret = new DataNode;
  ret->node_id_ = index_counter_++;
  index_to_node_.insert(pair<uint64_t, DagNode*>(ret->node_id_, ret));
  return ret;
}

OpNode* Dag::NewOpNode() {
  OpNode* ret = new OpNode;
  ret->node_id_ = index_counter_++;
  index_to_node_.insert(pair<uint64_t, DagNode*>(ret->node_id_, ret));
  return ret;
}

} // end of namespace minerva
