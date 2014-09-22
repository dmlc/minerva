#pragma once
#include <unordered_set>
#include <memory>
#include "common/singleton.h"
#include "dag/physical_dag.h"
#include "procedures/dag_scheduler.h"
#include "narray/narray.h"
#include "device/device_factory.h"
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
  DeviceFactory& device_factory() {
    return *device_factory_;
  }
  DagScheduler& dag_scheduler() {
    return *dag_scheduler_;
  }
  uint64_t CreateCPUDevice();
#ifdef HAS_CUDA
  uint64_t CreateGPUDevice(int gid);
#endif
  std::shared_ptr<float> GetValue(NArray& narr);
  std::pair<Device::MemType, float*> GetPtr(uint64_t, uint64_t);
  void Eval(const std::vector<NArray>& narrs);
  void EvalAsync(const std::vector<NArray>& narrs);
  void WaitForEvalFinish();
  uint64_t GenerateDataId();
  uint64_t current_device_id_;

 private:
  MinervaSystem();
  void LoadBuiltinDagMonitors();
  void ExecutePhysicalDag(const std::vector<uint64_t>& pids);
  PhysicalDag* physical_dag_;
  DagScheduler* dag_scheduler_;
  DeviceFactory* device_factory_;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
};

}  // end of namespace minerva

