#include "procedures/dag_engine.h"
#include "dag/dag.h"
#include "dag/dag_node.h"
#include "procedures/thread_pool.h"
#include "system/data_store.h"
#include <map>
#include <queue>
#include <functional>
#include <mutex>
#include <cstdio>
#include <unordered_set>

#define THREAD_NUM 4

using namespace std;

namespace minerva {

DagEngine::DagEngine(): unresolved_counter_(0), thread_pool_(THREAD_NUM, this) {
}

DagEngine::~DagEngine() {
  task_queue_.SignalForKill();
}

void DagEngine::Process(Dag& dag, vector<uint64_t>& targets) {
  // TODO Cannot add new vertices to complete nodes
  ParseDagState(dag);
  auto ready_to_execute = FindRootNodes(dag, targets);
  for (auto i: ready_to_execute) {
    AppendTask(i, bind(&DagEngine::NodeRunner, this, placeholders::_1));
  }
  {
    // Waiting execution to complete
    unique_lock<mutex> lock(unresolved_counter_mutex_);
    if (unresolved_counter_) {
      execution_finished_.wait(lock, [this]() -> bool {
        return unresolved_counter_ == 0;
      });
    }
  }
}

void DagEngine::ParseDagState(Dag& dag) {
  // Create NodeState for new nodes. Note that only new nodes are inserted.
  for (auto i: dag.index_to_node_) {
    NodeState n;
    n.state = NodeState::kNoNeed;
    n.dependency_counter = 0;
    node_states_.insert(make_pair(i.first, n));
  }
}

unordered_set<DagNode*> DagEngine::FindRootNodes(Dag& dag, vector<uint64_t>& targets) {
  queue<uint64_t> ready_node_queue;
  unordered_set<DagNode*> ready_to_execute;
  for (auto i: targets) {
    auto it = node_states_.find(i);
    if (it == node_states_.end()) { // Node not found
      continue;
    }
    ready_node_queue.push(i);
  }
  while (!ready_node_queue.empty()) {
    uint64_t cur = ready_node_queue.front();
    ready_node_queue.pop();
    auto& it = node_states_[cur];
    auto node = dag.index_to_node_[cur];
    if (it.state == NodeState::kComplete) {
      ready_to_execute.insert(node);
    } else {
      it.state = NodeState::kReady; // Set state to ready
      it.dependency_counter = node->predecessors_.size();
      if (node->predecessors_.empty()) {
        // Add root nodes to execution queue
        ready_to_execute.insert(node);
      } else {
        // Traverse predecessors
        for (auto i: node->predecessors_) {
          if (node_states_[i->node_id_].state == NodeState::kReady) { // Already visited
            continue;
          }
          ready_node_queue.push(i->node_id_);
        }
      }
    }
  }
  unresolved_counter_ = 0;
  // Mark target nodes
  for (auto i: targets) {
    node_states_[i].state = NodeState::kTarget;
    ++unresolved_counter_;
  }
  return ready_to_execute;
}

void DagEngine::NodeRunner(DagNode* node) {
  if (node->Type() == DagNode::OP_NODE) { // OpNode
    for (auto i: node->successors_) { // Allocate for each succesor
      DataNode* n = dynamic_cast<DataNode*>(i);
      DataStore::Instance().CreateData(n->data_id(), DataStore::CPU, n->meta().length);
    }
    dynamic_cast<OpNode*>(node)->runner()();
    node_states_[node->node_id_].state = NodeState::kNoNeed;
  } else {
    // Signal main process if a target is finished
    if (node_states_[node->node_id_].state == NodeState::kTarget) {
      unique_lock<mutex> lock(unresolved_counter_mutex_);
      --unresolved_counter_;
      execution_finished_.notify_one();
    }
    node_states_[node->node_id_].state = NodeState::kComplete;
  }
  {
    lock_guard<mutex> lock(node_states_mutex_);
    auto succ = node->successors_;
    for (auto i: succ) {
      auto& state = node_states_[i->node_id_];
      // Append node if all predecessors are finished
      if (state.state != NodeState::kNoNeed && (--state.dependency_counter) == 0) {
        AppendTask(i, bind(&DagEngine::NodeRunner, this, placeholders::_1));
      }
    }
  }
};

void DagEngine::AppendTask(Task node, Callback callback) {
  task_queue_.Push(make_pair(node, callback));
}

bool DagEngine::GetNewTask(thread::id id, TaskPair& task) {
  return task_queue_.Pop(task);
}

}
