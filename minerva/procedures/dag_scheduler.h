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
  void WaitForFinish(uint64_t);
  void GCNodes();
  // Monitor external reference changes
  void OnExternRCUpdate(PhysicalDataNode*);
  // DAG monitor
  void OnCreateNode(DagNode*);
  void OnDeleteNode(DagNode*);
  void OnCreateEdge(DagNode*, DagNode*);
  // Device listener
  void OnOperationComplete(PhysicalOpNode*);
  // DAG procedure
  void Process(const std::vector<uint64_t>&);

 private:
  void FreeDataNodeRes(PhysicalDataNode*);
  // Runtime information
  RuntimeInfoMap rt_info_;
  // Scheduler dispatcher
  ConcurrentBlockingQueue<std::pair<TaskType, uint64_t>> dispatcher_queue_;
  void DispatcherRoutine();
  std::thread dispatcher_;
  // Evaluation finishing signal
  std::atomic<int> num_nodes_yet_to_finish_;
  uint64_t target_ = -1;
  std::mutex finish_mutex_;
  std::condition_variable finish_cond_;
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
};

}  // namespace minerva

