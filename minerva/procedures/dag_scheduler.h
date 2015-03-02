#pragma once
#include "procedures/runtime_info_map.h"
#include "procedures/device_listener.h"
#include "dag/dag.h"
#include "dag/dag_monitor.h"
#include "dag/physical_dag.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include "system/backend.h"

#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace minerva {

class DeviceManager;
/*
 * Public methods of DAG scheduler are thread safe,
 * in the sense of multiple devices and a single user thread.
 */
class DagScheduler :
  public IBackend,
  public DagMonitor<PhysicalDag>,
  public DeviceListener {
 public:
  enum class TaskType {
    kToRun,
    kToComplete
  };
  DagScheduler(PhysicalDag*, DeviceManager*);
  ~DagScheduler();
  ////////////////////////// backend interfaces
  std::vector<MData*> Create(const std::vector<MData*>& params,
      const std::vector<Scale>& result_sizes, ComputeFn* fn) override;
  //virtual MData* RecordCreateInplace(MData* param, ComputeFn* fn) = 0;
  void ShallowCopy(MData*&, MData* from) override;
  void Destroy(MData* ) override;
  void Issue(MData* ) override;
  void Wait(MData* ) override;
  //void Wait(const std::vector<MData*>& ) = 0;
  void WaitForAll() override;
  std::shared_ptr<float> GetValue(MData* ) override;
  /////////////////////////
  
  // Wait for evaluation to finish
  //void WaitForFinish();
  //void WaitForFinish(uint64_t);
  void GCNodes();
  // Monitor external reference changes
  // DAG monitor
  void OnCreateNode(DagNode*) override;
  void OnDeleteNode(DagNode*) override;
  void OnCreateEdge(DagNode*, DagNode*) override;
  // Device listener
  void OnOperationComplete(PhysicalOpNode*) override;

 private:
  void FreeDataNodeRes(PhysicalDataNode*);
  void OnExternRCUpdate(PhysicalDataNode*);
  void Process(const std::vector<uint64_t>&);
  // Dag
  PhysicalDag* dag_;
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

