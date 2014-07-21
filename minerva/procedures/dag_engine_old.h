#pragma once
#include "dag/dag_node.h"
#include "procedures/dag_procedure.h"
#include "procedures/thread_pool.h"
#include "common/common.h"
#include "common/singleton.h"
#include "common/concurrent_blocking_queue.h"
#include <cstdint>
#include <map>
#include <vector>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_set>

/*
 * DagEngine makes following assumptions:
 * 1. Each DataNode has one and only one predecessor, which should be an OpNode.
 * 2. User is only allowed to add new nodes to the dag, using existing
 *    DataNodes as predecessors.
 * 3. DagEngine is responsible for GC.
 */

namespace minerva {

struct NodeState {
  enum State {
    kNoNeed, // No need to execute
    kReady, // Need to execute
    kComplete
  } state;
  size_t dependency_counter;
  std::condition_variable* on_complete;
};

class DagEngine : public DagProcedure, public Singleton<DagEngine> {
  friend class ThreadPool;

 public:
  typedef DagNode* Task;
  typedef std::function<void(DagNode*)> Callback;
  typedef std::pair<Task, Callback> TaskPair;
  DagEngine();
  ~DagEngine();
  void EvalNodes(const std::vector<uint64_t>&);

 private:
  DISALLOW_COPY_AND_ASSIGN(DagEngine);
  // Create states for new nodes
  void CommitDagChanges();
  // Find execution entry point
  std::unordered_set<DagNode*> FindRootNodes(const std::vector<uint64_t>&);
  // Callback when a node finishes execution
  void NodeRunner(DagNode*);
  void AppendTask(Task, Callback);
  bool GetNewTask(std::thread::id, TaskPair&);
  std::map<uint64_t, NodeState> node_states_;
  std::mutex node_states_mutex_;
  ConcurrentBlockingQueue<TaskPair> task_queue_;
  ThreadPool thread_pool_;
};

}

