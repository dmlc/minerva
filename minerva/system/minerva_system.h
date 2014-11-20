#pragma once
#include <unordered_set>
#include <memory>
#include "common/singleton.h"
#include "dag/physical_dag.h"
#include "procedures/dag_scheduler.h"
#include "narray/narray.h"
#include "device/device_manager.h"
#include "common/inspector.h"
#include "device/device.h"

namespace minerva {

class MinervaSystem :
  public EverlastingSingleton<MinervaSystem> {
  friend class NArray;
  friend class EverlastingSingleton<MinervaSystem>;
  friend class Inspector<MinervaSystem>;

 public:
  static void UniversalMemcpy(std::pair<Device::MemType, float*>, std::pair<Device::MemType, float*>, size_t);
  ~MinervaSystem();
  void Initialize(int* argc, char*** argv);
  void Finalize();
  PhysicalDag& physical_dag() {
    return *physical_dag_;
  }
  DeviceManager& device_manager() {
    return *device_manager_;
  }
  DagScheduler& dag_scheduler() {
    return *dag_scheduler_;
  }
  uint64_t CreateCpuDevice();
#ifdef HAS_CUDA
  uint64_t CreateGpuDevice(int gid);
#endif
  std::shared_ptr<float> GetValue(const NArray& narr);
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);
  void IncrExternRC(PhysicalDataNode*);
  void DecrExternRC(PhysicalDataNode*);
  void Eval(const std::vector<NArray>& narrs);
  void EvalAsync(const std::vector<NArray>& narrs);
  uint64_t GenerateDataId();
  uint64_t current_device_id_;

 private:
  MinervaSystem();
  void LoadBuiltinDagMonitors();
  void ExecutePhysicalDag(const std::vector<uint64_t>& pids);
  PhysicalDag* physical_dag_;
  DagScheduler* dag_scheduler_;
  DeviceManager* device_manager_;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
};

}  // end of namespace minerva

