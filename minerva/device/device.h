#pragma once
#include <cublas.h>
#include <cuda.h>
#include <vector>
#include "device_info.h"
#include "system/data_store.h"
#include "op/physical.h"
#include "op/physical_fn.h"
#include "common/inspector.h"

namespace minerva {

class MinervaSystem;

class Device {
  friend class Inspector<MinervaSystem>;
 public:
  enum DeviceTypes {
   CPU_DEVICE = 0,
   GPU_DEVICE
  };
  Device();
  Device(uint64_t id, DeviceInfo info);
  DeviceInfo GetInfo();
  void Execute(uint64_t nid, std::vector<PhysicalData> inputs, std::vector<PhysicalData> outputs, const PhysicalOp Op); // called by Physical_Engine::ProcessNode()
  virtual DeviceTypes Type() const = 0;
  virtual float* GetData(uint64_t data_id) = 0;
  virtual void CopyData(void * dst, void * src, size_t numBytes);

 protected:
  std::set<uint64_t> local_data_;
  DeviceInfo device_info_;
  uint64_t device_id_;
  virtual void CreateData(uint64_t data_id, int size) = 0;
  virtual void Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op) = 0;
};

class GpuDevice : public Device {
 public:
  GpuDevice(uint64_t id, DeviceInfo info);
  ~GpuDevice();
  DeviceTypes Type() const { return GPU_DEVICE; }
  void CreateData(uint64_t data_id, int size);
  float* GetData(uint64_t data_id);
  void Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op);
private:
  std::vector<cudaStream_t> streams_;
  std::vector<cublasHandle_t> cublas_handles_;
  int stream_ptr_;
  const static int num_streams_ = 16;
  float * gpu_buffer_one_;
  float * gpu_buffer_zero_;
};

class CpuDevice : public Device {
 public:
  CpuDevice(uint64_t id, DeviceInfo info) : Device(id, info) {}
  DeviceTypes Type() const { return CPU_DEVICE; }
  void CreateData(uint64_t data_id, int size);
  float* GetData(uint64_t data_id);
  void Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op);
};

}
