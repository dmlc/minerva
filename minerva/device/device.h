#pragma once
#include <string>
#include <utility>
#include <mutex>
#include "device/task.h"
#include "device/data_store.h"
#include "device/device_listener.h"
#include "op/physical_fn.h"
#include "common/common.h"
#include "common/thread_pool.h"
#include "common/concurrent_blocking_queue.h"
#include "common/concurrent_unordered_set.h"
#include "common/concurrent_unordered_map.h"
#ifdef HAS_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>
#endif

namespace minerva {

class Device {
 public:
  enum class MemType {
    kCpu,
    kGpu
  };
  Device() = delete;
  DISALLOW_COPY_AND_ASSIGN(Device);
  Device(uint64_t device_id, DeviceListener*);
  virtual ~Device() = default;
  virtual void PushTask(Task*) = 0;
  virtual std::pair<MemType, float*> GetPtr(uint64_t data_id);
  virtual void FreeDataIfExist(uint64_t data_id);
  virtual std::string GetMemUsage() const;
  virtual std::string Name() const = 0;
  virtual MemType GetMemType() const = 0;

 protected:
  ConcurrentUnorderedSet<uint64_t> local_data_;
  ConcurrentUnorderedSet<uint64_t> remote_data_;
  uint64_t device_id_;
  DataStore* data_store_;
  DeviceListener* listener_;
};

class ThreadedDevice : public Device {
 public:
  ThreadedDevice() = delete;
  ThreadedDevice(uint64_t device_id, DeviceListener*, size_t parallelism);
  DISALLOW_COPY_AND_ASSIGN(ThreadedDevice);
  ~ThreadedDevice() = default;
  void PushTask(Task*) override;
  void FreeDataIfExist(uint64_t data_id) override;

 protected:
  virtual void Execute(Task*, int thrid);
  virtual void PreExecute();
  virtual void Barrier(int);
  virtual void DoCopyRemoteData(float*, float*, size_t, int) = 0;
  virtual void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) = 0;
  ConcurrentUnorderedMap<uint64_t, std::mutex> copy_locks_;
  ThreadPool pool_;
};

#ifdef HAS_CUDA
class GpuDevice : public ThreadedDevice {
 public:
  GpuDevice(uint64_t device_id, DeviceListener*, int gpu_id);
  DISALLOW_COPY_AND_ASSIGN(GpuDevice);
  ~GpuDevice();
  MemType GetMemType() const override;
  std::string Name() const override;

 private:
  static const size_t kParallelism = 4;  // TODO change me
  const int device_;
  cudaStream_t stream_[kParallelism];
  cublasHandle_t cublas_handle_[kParallelism];
  cudnnHandle_t cudnn_handle_[kParallelism];
  void PreExecute() override;
  void Barrier(int) override;
  void DoCopyRemoteData(float*, float*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
};
#endif

class CpuDevice : public ThreadedDevice {
 public:
  CpuDevice(uint64_t device_id, DeviceListener*);
  DISALLOW_COPY_AND_ASSIGN(CpuDevice);
  ~CpuDevice();
  MemType GetMemType() const override;
  std::string Name() const override;

 private:
  static const size_t kDefaultThreadNum = 4;
  void DoCopyRemoteData(float*, float*, size_t, int) override;
  void DoExecute(const DataList&, const DataList&, PhysicalOp&, int) override;
};

}  // namespace minerva

