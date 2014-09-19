#pragma once
#include "procedures/dag_procedure.h"
#include "procedures/runtime_info_map.h"
#include "procedures/device_listener.h"
#include "dag/dag.h"
#include "dag/dag_monitor.h"
#include "dag/physical_dag.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace minerva {

/*
 * Public methods of DAG scheduler are thread safe,
 * in the sense of multiple devices and a single user thread.
 */
class DagScheduler : public DagProcedure<PhysicalDag>, public DagMonitor<PhysicalDag>, public DeviceListener {
 public:
  enum class TaskType {
    kToRun,
    kToComplete
  };
  DagScheduler(PhysicalDag*);
  virtual ~DagScheduler();
  // Wait for evaluation to finish
  void WaitForFinish();
  void GCNodes();
  // Monitor external reference changes
  void OnIncrExternRC(PhysicalDataNode*, int);
  // DAG monitor
  void OnCreateNode(DagNode*);
  void OnDeleteNode(DagNode*);
  void OnCreateEdge(DagNode*, DagNode*);
  void OnBeginModify();
  void OnFinishModify();
  // Device listener
  void OnOperationComplete(uint64_t);
  // DAG procedure
  void Process(const std::vector<uint64_t>&);

 private:
  int CalcTotalReferenceCount(PhysicalDataNode*);
  void FreeDataNodeRes(PhysicalDataNode*);
  std::recursive_mutex m_;
  // Runtime information
  RuntimeInfoMap rt_info_;
  // Scheduler dispatcher
  ConcurrentBlockingQueue<std::pair<TaskType, uint64_t>> dispatcher_queue_;
  void DispatcherRoutine();
  std::thread dispatcher_;
  // Evaluation finishing signal
  std::atomic<int> num_nodes_yet_to_finish_;
  std::mutex finish_mutex_;
  std::condition_variable finish_cond_;
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
};

}  // namespace minerva

