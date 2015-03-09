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
  ExecutionProfiler& profiler() {
    return *profiler_;
  }
#ifdef HAS_CUDA
  int GetGpuDeviceCount();
#endif
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);
  uint64_t GenerateDataId();
  uint64_t current_device_id_;

 private:
  MinervaSystem(int*, char***);
  PhysicalDag* physical_dag_;
  IBackend* backend_;
  ExecutionProfiler* profiler_;
  DeviceManager* device_manager_;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
};

}  // end of namespace minerva

