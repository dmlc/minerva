#pragma once
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <condition_variable>
#include <thread>
#include <mutex>

#include "procedures/dag_procedure.h"
#include "procedures/thread_pool.h"
#include "procedures/state.h"
#include "common/common.h"

namespace minerva {

/*
 * PhysicalEngine makes following assumptions:
 * 1. Each PhysicalDataNode has one or zero predecessor.
 * 2. User is only allowed to add new nodes to the dag.
 * 3. PhysicalEngine is responsible for GC.
 */

/*struct NodeState {
  enum State {
    kNoNeed, // No need to execute
    kReady, // Need to execute
    kComplete
  } state;
};*/

class PhysicalEngine: public PhysicalDagProcedure, public PhysicalDagMonitor {
  friend class ThreadPool;

 public:
  typedef DagNode* Task;
  typedef std::function<void(Task)> Callback;
  typedef std::pair<Task, Callback> TaskPair;
  PhysicalEngine(NodeStateMap<PhysicalDag>& ns);
  ~PhysicalEngine();
  void Process(PhysicalDag&, const std::vector<uint64_t>&);
  void OnCreateNode(DagNode* node);
  void OnDeleteNode(DagNode* node);
  void GCNodes(PhysicalDag& );

 private:
  struct RuntimeState {
    int dependency_counter;
    std::condition_variable* on_complete;
  };

 private:
  DISALLOW_COPY_AND_ASSIGN(PhysicalEngine);
  void Init();
  std::unordered_set<DagNode*> FindRootNodes(PhysicalDag& dag, const std::vector<uint64_t>&);
  void NodeRunner(DagNode*);
  void AppendTask(Task, Callback);
  bool GetNewTask(std::thread::id, TaskPair&);

  std::mutex node_states_mutex_;
  std::unordered_map<uint64_t, RuntimeState> rt_states_;
  NodeStateMap<PhysicalDag>& node_states_;

  ConcurrentBlockingQueue<TaskPair> task_queue_;
  ThreadPool thread_pool_;
};

}
