#include <utility>
#include <cstdlib>
#include <mutex>
#include <sstream>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "device.h"
#include "device/task.h"
#include "system/minerva_system.h"
#include "op/context.h"
#include "common/cuda_utils.h"
#include "device/pooled_data_store.h"
#include "profiler/wall_timer.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

#define DEFAULT_POOL_SIZE ((size_t) 5.8 * 1024 * 1024 * 1024)
DEFINE_bool(no_execute, false, "Disable the actual computation (for performance debuggin)");

using namespace std;

namespace minerva {

Device::Device(uint64_t device_id, DeviceListener* l) : device_id_(device_id), data_store_(0), listener_(l) {
}

pair<Device::MemType, float*> Device::GetPtr(uint64_t data_id) {
  return make_pair(GetMemType(), data_store_->GetData(data_id));
}

void Device::FreeDataIfExist(uint64_t data_id) {
  auto d = local_data_.Erase(data_id) + remote_data_.Erase(data_id);
  if (d == 1) {
    data_store_->FreeData(data_id);
  } else if (d != 0) {
    LOG(FATAL) << "duplicate data";
  }
}

string Device::GetMemUsage() const {
  stringstream ss;
  ss << "device #" << device_id_ << " used " << data_store_->GetTotalBytes() << "B";
  return ss.str();
}

ThreadedDevice::ThreadedDevice(uint64_t device_id, DeviceListener* l, size_t parallelism) : Device(device_id, l), pool_(parallelism) {
}

void ThreadedDevice::PushTask(Task* task) {
  pool_.Push(bind(&ThreadedDevice::Execute, this, task, placeholders::_1));
}

void ThreadedDevice::FreeDataIfExist(uint64_t data_id) {
  copy_locks_.Erase(data_id);
  Device::FreeDataIfExist(data_id);
}

void ThreadedDevice::Execute(Task* task, int thrid) {
  PreExecute();
#ifndef NDEBUG
  WallTimer memory_timer;
  memory_timer.Start();
#endif
  DataList input_shards;
  for (auto& i : task->inputs) {
    auto& input_data = i.physical_data;
    if (input_data.device_id == device_id_) {  // Input is local
      DLOG(INFO) << Name() << " input task data #" << i.id << " is local";
      CHECK_EQ(local_data_.Count(input_data.data_id), 1);
    } else {
      lock_guard<mutex> lck(copy_locks_[input_data.data_id]);
      if (!remote_data_.Count(input_data.data_id)) {  // Input is remote and not copied
        DLOG(INFO) << Name() << " input task data #" << i.id << " is remote and not copied";
        size_t size = input_data.size.Prod() * sizeof(float);
        auto ptr = data_store_->CreateData(input_data.data_id, size);
        DoCopyRemoteData(ptr, MinervaSystem::Instance().GetPtr(input_data.device_id, input_data.data_id).second, size, thrid);
        CHECK(remote_data_.Insert(input_data.data_id));
      }
    }
    input_shards.emplace_back(data_store_->GetData(input_data.data_id), input_data.size);
  }
  DataList output_shards;
  for (auto& i : task->outputs) {
    size_t size = i.physical_data.size.Prod() * sizeof(float);
    DLOG(INFO) << Name() << " create output for task data #" << i.id;
    auto ptr = data_store_->CreateData(i.physical_data.data_id, size);
    CHECK(local_data_.Insert(i.physical_data.data_id));
    output_shards.emplace_back(ptr, i.physical_data.size);
  }
  auto& op = task->op;
  CHECK(op.compute_fn);
  if(!FLAGS_no_execute) {
#ifndef NDEBUG
    Barrier(thrid);
    memory_timer.Stop();
    MinervaSystem::Instance().profiler().RecordTime(TimerType::kMemory, op.compute_fn->Name(), memory_timer);
    WallTimer calculate_timer;
    calculate_timer.Start();
#endif
    DLOG(INFO) << Name() << " execute task #" << task->id << ": " << op.compute_fn->Name();
    DoExecute(input_shards, output_shards, op, thrid);
    DLOG(INFO) << Name() << " finished execute task #" << task->id << ": " << op.compute_fn->Name();
#ifndef NDEBUG
    calculate_timer.Stop();
    MinervaSystem::Instance().profiler().RecordTime(TimerType::kCalculation, op.compute_fn->Name(), calculate_timer);
#endif
  }
  listener_->OnOperationComplete(task);
}

void ThreadedDevice::PreExecute() {
}

void ThreadedDevice::Barrier(int) {
}

#ifdef HAS_CUDA
GpuDevice::GpuDevice(uint64_t device_id, DeviceListener* l, int gpu_id) : ThreadedDevice(device_id, l, kParallelism), device_(gpu_id) {
  CUDA_CALL(cudaSetDevice(device_));
  cudaFree(0);  // Initialize
  auto allocator = [this](size_t len) -> void* {
    void* ret;
    CUDA_CALL(cudaSetDevice(device_));
    CUDA_CALL(cudaMalloc(&ret, len));
    return ret;
  };
  auto deallocator = [this](void* ptr) {
    CUDA_CALL(cudaSetDevice(device_));
    CUDA_CALL(cudaFree(ptr));
  };
  data_store_ = new PooledDataStore(DEFAULT_POOL_SIZE, allocator, deallocator);
  for (size_t i = 0; i < kParallelism; ++i) {
    CUDA_CALL(cudaStreamCreate(&stream_[i]));
    CUBLAS_CALL(cublasCreate(&cublas_handle_[i]));
    CUBLAS_CALL(cublasSetStream(cublas_handle_[i], stream_[i]));
    CUDNN_CALL(cudnnCreate(&cudnn_handle_[i]));
    CUDNN_CALL(cudnnSetStream(cudnn_handle_[i], stream_[i]));
  }
}

GpuDevice::~GpuDevice() {
  CUDA_CALL(cudaSetDevice(device_));
  pool_.WaitForAllFinished();
  for (size_t i = 0; i < kParallelism; ++i) {
    CUDNN_CALL(cudnnDestroy(cudnn_handle_[i]));
    CUBLAS_CALL(cublasDestroy(cublas_handle_[i]));
    CUDA_CALL(cudaStreamDestroy(stream_[i]));
  }
  delete data_store_;
}

Device::MemType GpuDevice::GetMemType() const {
  return MemType::kGpu;
}

string GpuDevice::Name() const {
  stringstream ss;
  ss << "GPU device #" << device_id_;
  return ss.str();
}

void GpuDevice::PreExecute() {
  CUDA_CALL(cudaSetDevice(device_));
}

void GpuDevice::Barrier(int thrid) {
  CUDA_CALL(cudaStreamSynchronize(stream_[thrid]));
}

void GpuDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int thrid) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream_[thrid]));
  CUDA_CALL(cudaStreamSynchronize(stream_[thrid]));
}

void GpuDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int thrid) {
  Context ctx;
  ctx.impl_type = ImplType::kCuda;
  ctx.stream = stream_[thrid];
  ctx.cublas_handle = cublas_handle_[thrid];
  ctx.cudnn_handle = cudnn_handle_[thrid];
  op.compute_fn->Execute(in, out, ctx);
  CUDA_CALL_MSG(op.compute_fn->Name(), cudaStreamSynchronize(stream_[thrid]));
}

#endif

CpuDevice::CpuDevice(uint64_t device_id, DeviceListener* l) : ThreadedDevice(device_id, l, kDefaultThreadNum) {
  auto allocator = [](size_t len) -> void* {
    void* ret = malloc(len);
    return ret;
  };
  auto deallocator = [](void* ptr) {
    free(ptr);
  };
  data_store_ = new DataStore(allocator, deallocator);
}

CpuDevice::~CpuDevice() {
  pool_.WaitForAllFinished();
  delete data_store_;
}

Device::MemType CpuDevice::GetMemType() const {
  return MemType::kCpu;
}

string CpuDevice::Name() const {
  stringstream ss;
  ss << "CPU device #" << device_id_;
  return ss.str();
}

void CpuDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
#else
  memcpy(dst, src, size);
#endif
}

void CpuDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int) {
  Context ctx;
  ctx.impl_type = ImplType::kBasic;
  op.compute_fn->Execute(in, out, ctx);
}

}  // namespace minerva

