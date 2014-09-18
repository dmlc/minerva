#pragma once
#include <unordered_set>
#include "common/singleton.h"
#include "dag/physical_dag.h"
#include "procedures/dag_scheduler.h"
#include "narray/narray.h"
#include "device/device_factory.h"
#include "common/inspector.h"

namespace minerva {

class MinervaSystem :
  public EverlastingSingleton<MinervaSystem> {
  friend class NArray;
  friend class EverlastingSingleton<MinervaSystem>;
  friend class Inspector<MinervaSystem>;

 public:
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
  float* GetValue(NArray& narr);
  void Eval(const std::vector<NArray>& narrs);
  void EvalAsync(const std::vector<NArray>& narrs);
  void WaitForEvalFinish();
  void set_device_id(uint64_t id);
  uint64_t device_id() const;
  uint64_t CreateCPUDevice();
  uint64_t CreateGPUDevice(int gid);
  uint64_t CreateGPUDevice(int gid, int num_stream);
  Device* GetDevice(uint64_t id);

 private:
  MinervaSystem();
  void LoadBuiltinDagMonitors();
  void IncrExternRC(LogicalDag::DNode*, int amount = 1);
  void GeneratePhysicalDag(const std::vector<uint64_t>& lids);
  void ExecutePhysicalDag(const std::vector<uint64_t>& pids);
  PhysicalDag* physical_dag_;
  DagScheduler* dag_scheduler_;
  DeviceFactory* device_factory_;
  uint64_t device_id_;

  std::unordered_set<uint64_t> extern_rc_changed_ldnodes_;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
};

}  // end of namespace minerva

