#pragma once
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include "backend/dag/runtime_info_map.h"
#include "backend/backend.h"
#include "device/device_listener.h"
#include "dag/dag.h"
#include "dag/physical_dag.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include "device/device_manager.h"
#include "backend/dag/task_type.h"
#include "backend/dag/priority_dispatcher_queue.h"

namespace minerva {

class DagScheduler :
  public Backend,
  public DeviceListener {
 public:
  DagScheduler() = delete;
  DagScheduler(PhysicalDag*, DeviceManager*);
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
  ~DagScheduler();
  // Backend
  std::vector<BackendChunk*> Create(const std::vector<BackendChunk*>&,
      const std::vector<Scale>&, std::shared_ptr<ComputeFn>) override;
  void Wait(BackendChunk*) override;
  void WaitForAll() override;
  std::shared_ptr<float> GetValue(BackendChunk*) override;
  // Device listener
  void OnOperationComplete(Task*) override;
  // Interface for `DagChunk`
  void ExternRCUpdate(PhysicalDataNode*, int);

 private:
  void FreeDataNodeRes(PhysicalDataNode*);
  void OnCreateNode(DagNode*);
  void OnDeleteNode(DagNode*);
  void OnCreateEdge(DagNode*, DagNode*);
  void ProcessIfReady(PhysicalOpNode*);
  // Dag
  PhysicalDag* dag_;
  // Device manager
  DeviceManager* dm_;
  // Runtime information
  RuntimeInfoMap rt_info_;
  // Scheduler dispatcher
  PriorityDispatcherQueue dispatcher_queue_;
  void DispatcherRoutine();
  void DecrNumNodesYetToFinish(uint64_t);
  std::thread dispatcher_;
  // Evaluation finishing signal
  std::atomic<int> num_nodes_yet_to_finish_;
  uint64_t target_ = -1;
  std::mutex finish_mutex_;
  std::condition_variable finish_cond_;
};

}  // namespace minerva

