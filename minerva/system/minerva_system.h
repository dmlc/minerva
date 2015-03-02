#pragma once
#include <unordered_set>
#include <memory>
#include "common/singleton.h"
#include "common/inspector.h"
#include "dag/physical_dag.h"
#include "procedures/dag_scheduler.h"
#include "narray/narray.h"
#include "device/device_manager.h"
#include "device/device.h"
#include "profiler/execution_profiler.h"
#include "system/backend.h"

namespace minerva {

class MinervaSystem :
  public EverlastingSingleton<MinervaSystem> {
  friend class NArray;
  friend class EverlastingSingleton<MinervaSystem>;
  friend class Inspector<MinervaSystem>;

 public:
  static void UniversalMemcpy(std::pair<Device::MemType, float*>, std::pair<Device::MemType, float*>, size_t);
  ~MinervaSystem();
  PhysicalDag& physical_dag() {
    return *physical_dag_;
  }
  DeviceManager& device_manager() {
    return *device_manager_;
  }
  DagScheduler& dag_scheduler() {
    //return *dag_scheduler_;
    return *((DagScheduler*)backend_);
  }
  ExecutionProfiler& profiler() {
    return *profiler_;
  }
  uint64_t CreateCpuDevice();
#ifdef HAS_CUDA
  uint64_t CreateGpuDevice(int gid);
  int GetGpuDeviceCount();
#endif
  //std::shared_ptr<float> GetValue(const NArray& narr);
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);
  //void IncrExternRC(PhysicalDataNode*);
  //void DecrExternRC(PhysicalDataNode*);
  //void WaitForEval(const std::vector<NArray>& narrs);
  //void StartEval(const std::vector<NArray>& narrs);
  uint64_t GenerateDataId();
  uint64_t current_device_id_;

  ////// interfaces for calling backends
  std::vector<MData*> Create(const std::vector<MData*>& params, const std::vector<Scale>& result_sizes, ComputeFn* fn);
  MData* CreateOne(MData* param, const Scale& result_size, ComputeFn* fn);
  //virtual MData* RecordCreateInplace(MData* param, ComputeFn* fn);
  void ShallowCopy(MData*& to, MData* from);
  void Destroy(MData* );
  void Issue(MData* );
  void Wait(MData* );
  //virtual void Wait(const std::vector<MData*>& );
  void WaitForAll();
  std::shared_ptr<float> GetValue(MData* );

 private:
  MinervaSystem(int*, char***);
  //void ExecutePhysicalDag(const std::vector<uint64_t>& pids);
  PhysicalDag* physical_dag_;
  //DagScheduler* dag_scheduler_;
  IBackend* backend_;
  ExecutionProfiler* profiler_;
  DeviceManager* device_manager_;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
};

}  // end of namespace minerva

