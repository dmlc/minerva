#pragma once
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <mutex>
#include "device/data_store.h"
#include "op/physical.h"
#include "op/physical_fn.h"
#include "procedures/device_listener.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include "common/thread_pool.h"
#include "common/concurrent_unordered_set.h"
#include "common/concurrent_unordered_map.h"
#ifdef HAS_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#endif

namespace minerva {

class Device {
 public:
  enum class MemType {
    kCpu,
    kGpu
  };
  Device(uint64_t, DeviceListener*);
  virtual ~Device();
  virtual void PushTask(uint64_t) = 0;
  virtual std::pair<MemType, float*> GetPtr(uint64_t);
  virtual std::string Name() const = 0;
  virtual MemType GetMemType() const = 0;
  virtual void FreeDataIfExist(uint64_t);

 protected:
  ConcurrentUnorderedSet<uint64_t> local_data_;
  ConcurrentUnorderedSet<uint64_t> remote_data_;
  uint64_t device_id_;
  DataStore* data_store_;
  DeviceListener* listener_;

 private:
  Device();
  DISALLOW_COPY_AND_ASSIGN(Device);
};

class ThreadedDevice : public Device {
 public:
  ThreadedDevice(uint64_t, DeviceListener*, size_t);
  ~ThreadedDevice();
  void PushTask(uint64_t);
  void FreeDataIfExist(uint64_t);

 protected:
  virtual void Execute(uint64_t, int);
  virtual void DoCopyRemoteData(float*, float*, size_t, int) = 0;
  virtual void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) = 0;
  ConcurrentUnorderedMap<uint64_t, std::mutex> copy_locks_;
  ThreadPool pool_;

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadedDevice);
};

#ifdef HAS_CUDA
class GpuDevice : public ThreadedDevice {
 public:
  GpuDevice(uint64_t, DeviceListener*, int);
  ~GpuDevice();
  MemType GetMemType() const;
  std::string Name() const;

 private:
  static const size_t kParallelism = 16;
  const int device_;
  void DoCopyRemoteData(float*, float*, size_t, int);
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int);
  cudaStream_t stream_[kParallelism];
  cublasHandle_t handle_[kParallelism];
  DISALLOW_COPY_AND_ASSIGN(GpuDevice);
};
#endif

class CpuDevice : public ThreadedDevice {
 public:
  CpuDevice(uint64_t, DeviceListener*);
  ~CpuDevice();
  MemType GetMemType() const;
  std::string Name() const;

 private:
  static const size_t kDefaultThreadNum = 8;
  void DoCopyRemoteData(float*, float*, size_t, int);
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int);
  DISALLOW_COPY_AND_ASSIGN(CpuDevice);
};

}  // namespace minerva

