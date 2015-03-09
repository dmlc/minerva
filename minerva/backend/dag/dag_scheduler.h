#pragma once
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "backend/dag/runtime_info_map.h"
#include "backend/backend.h"
#include "device/device_listener.h"
#include "dag/dag.h"
#include "dag/physical_dag.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"

namespace minerva {

class DeviceManager;

class DagScheduler :
  public Backend,
  public DeviceListener {
 public:
  enum class TaskType {
    kToRun,
    kToComplete
  };
  DagScheduler() = delete;
  DagScheduler(PhysicalDag*, DeviceManager*);
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
  ~DagScheduler();
  // Backend
  std::vector<BackendChunk*> Create(const std::vector<BackendChunk*>&,
      const std::vector<Scale>&, ComputeFn*) override;
  void Wait(BackendChunk*) override;
  void WaitForAll() override;
  std::shared_ptr<float> GetValue(BackendChunk*) override;
  // Device listener
  void OnOperationComplete(Task*) override;

 private:
  void FreeDataNodeRes(PhysicalDataNode*);
  void OnExternRCUpdate(PhysicalDataNode*);
  void Process(const std::vector<uint64_t>&);
  std::mutex m_;
  // Dag
  PhysicalDag* dag_;
  // Device manager
  DeviceManager* dm_;
  // Runtime information
  RuntimeInfoMap rt_info_;
  // Scheduler dispatcher
  ConcurrentBlockingQueue<std::pair<TaskType, uint64_t>> dispatcher_queue_;
  void DispatcherRoutine();
  std::thread dispatcher_;
  // Evaluation finishing signal
  std::atomic<int> num_nodes_yet_to_finish_;
  uint64_t target_ = -1;
  std::condition_variable finish_cond_;
};

}  // namespace minerva

