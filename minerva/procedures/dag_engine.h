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

namespace minerva {

struct NodeState {
  enum State {
    kNoNeed, // No need to execute
    kReady, // Need to execute
    kTarget // Target node
  } state;
  size_t dependency_counter;
};

class DagEngine : public DagProcedure, public Singleton<DagEngine> {
  friend class ThreadPool;

 public:
  typedef DagNode* Task;
  typedef std::function<void(DagNode*)> Callback;
  typedef std::pair<Task, Callback> TaskPair;
  DagEngine();
  ~DagEngine();
  void Process(Dag&, std::vector<uint64_t>&);

 private:
  DISALLOW_COPY_AND_ASSIGN(DagEngine);
  // Create states for new nodes
  void ParseDagState(Dag&);
  // Find execution entry point
  std::queue<DagNode*> FindRootNodes(Dag&, std::vector<uint64_t>&);
  // Callback when a node finishes execution
  void NodeRunner(DagNode*);
  void AppendTask(Task, Callback);
  bool GetNewTask(std::thread::id, TaskPair&);
  std::map<uint64_t, NodeState> node_states_;
  std::mutex node_states_mutex_;
  size_t unresolved_counter_;
  std::mutex unresolved_counter_mutex_;
  std::condition_variable execution_finished_;
  ConcurrentBlockingQueue<TaskPair> task_queue_;
  ThreadPool thread_pool_;
};

}

