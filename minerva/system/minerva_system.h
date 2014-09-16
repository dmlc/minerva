#pragma once
#include "common/singleton.h"
#include "dag/physical_dag.h"
#include "procedures/dag_procedure.h"
#include "narray/narray.h"
#include "device/device_info.h"
#include "common/inspector.h"
#include <unordered_set>

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
  PhysicalDag& physical_dag() { return physical_dag_; }
  PhysicalEngine& physical_engine() { return *physical_engine_; }

  float* GetValue(NArray& narr);
  void Eval(const std::vector<NArray>& narrs);
  void EvalAsync(const std::vector<NArray>& narrs);
  void WaitForEvalFinish();

  void set_device_info(DeviceInfo info);
  DeviceInfo device_info() const;
  DeviceInfo CreateGpuDevice(int gid);
  DeviceInfo CreateGpuDevice(int gid, int num_stream);

 private:
  MinervaSystem();
  void LoadBuiltinDagMonitors();
  void IncrExternRC(LogicalDag::DNode*, int amount = 1);
  void GeneratePhysicalDag(const std::vector<uint64_t>& lids);
  void ExecutePhysicalDag(const std::vector<uint64_t>& pids);
  PhysicalDag physical_dag_;

  PhysicalEngine* physical_engine_;

  DeviceFactory df_;
  DeviceInfo device_info_;

  std::unordered_set<uint64_t> extern_rc_changed_ldnodes_;
  DISALLOW_COPY_AND_ASSIGN(MinervaSystem);
};

}  // end of namespace minerva

