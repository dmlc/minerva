#include "device/device.h"
#include <utility>
#include <cstdlib>
#include <mutex>
#include <glog/logging.h>
#include <sstream>
#include "system/minerva_system.h"
#include "op/context.h"
#include "common/cuda_utils.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

using namespace std;

namespace minerva {

Device::Device(uint64_t d, DeviceListener* l) : device_id_(d), data_store_(0), listener_(l) {
}

Device::~Device() {
}

pair<Device::MemType, float*> Device::GetPtr(uint64_t id) {
  return make_pair(GetMemType(), data_store_->GetData(id));
}

void Device::FreeDataIfExist(uint64_t id) {
  auto d = local_data_.Erase(id) + remote_data_.Erase(id);
  if (d == 1) {
    data_store_->FreeData(id);
  } else if (d != 0) {
    CHECK(false) << "duplicate data";
  }
}

ThreadedDevice::ThreadedDevice(uint64_t id, DeviceListener* l, size_t p) : Device(id, l), pool_(p) {
}

ThreadedDevice::~ThreadedDevice() {
  pool_.WaitForAllFinished();
}

void ThreadedDevice::PushTask(uint64_t id) {
  pool_.Push(bind(&ThreadedDevice::Execute, this, id, placeholders::_1));
}

void ThreadedDevice::FreeDataIfExist(uint64_t id) {
  copy_locks_.Erase(id);
  Device::FreeDataIfExist(id);
}

void ThreadedDevice::PreExecute() {
}

void ThreadedDevice::Execute(uint64_t nid, int thrid) {
  PreExecute();
  auto node = MinervaSystem::Instance().physical_dag().GetNode(nid);
  if (node->Type() == DagNode::NodeType::kOpNode) {  // Op node
    auto op_node = CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(node));
    DataList input_shards;
    for (auto i : op_node->inputs_) {
      auto& input_data = i->data_;
      if (input_data.device_id == device_id_) {  // Input is local
        DLOG(INFO) << Name() << " input node #" << i->node_id() << " data #" << input_data.data_id << " is local";
        CHECK_EQ(local_data_.Count(input_data.data_id), 1);
      } else {
        lock_guard<mutex> lck(copy_locks_[input_data.data_id]);
        if (!remote_data_.Count(input_data.data_id)) {  // Input is remote and not copied
          DLOG(INFO) << Name() << " input node #" << i->node_id() << " is remote and not copied";
          size_t size = input_data.size.Prod() * sizeof(float);
          auto ptr = data_store_->CreateData(input_data.data_id, size);
          DoCopyRemoteData(ptr, MinervaSystem::Instance().GetPtr(input_data.device_id, input_data.data_id).second, size, thrid);
          CHECK(remote_data_.Insert(input_data.data_id));
        }
      }
      input_shards.emplace_back(data_store_->GetData(i->data_.data_id), i->data_.size);
    }
    DataList output_shards;
    for (auto i : op_node->outputs_) {
      size_t size = i->data_.size.Prod() * sizeof(float);
      DLOG(INFO) << "create output data node #" << i->node_id();
      auto ptr = data_store_->CreateData(i->data_.data_id, size);
      CHECK(local_data_.Insert(i->data_.data_id));
      output_shards.emplace_back(ptr, i->data_.size);
    }
    auto& op = op_node->op_;
    CHECK_NOTNULL(op.compute_fn);
    DLOG(INFO) << Name() << " execute node #" << nid << ": " << op.compute_fn->Name();
    DoExecute(input_shards, output_shards, op, thrid);
  }
  listener_->OnOperationComplete(nid);
}

#ifdef HAS_CUDA

GpuDevice::GpuDevice(uint64_t id, DeviceListener* l, int gid) : ThreadedDevice(id, l, kParallelism), device_(gid) {
  CUDA_CALL(cudaSetDevice(device_));
  cudaFree(0);  // Initialize
  auto allocator = [](size_t len) -> void* {
    void* ret;
    CUDA_CALL(cudaMalloc(&ret, len));
    return ret;
  };
  auto deallocator = [](void* ptr) {
    CUDA_CALL(cudaFree(ptr));
  };
  data_store_ = new DataStore(allocator, deallocator);
  for (size_t i = 0; i < kParallelism; ++i) {
    CUDA_CALL(cudaStreamCreate(&stream_[i]));
    CUBLAS_CALL(cublasCreate(&cublas_handle_[i]));
    CUBLAS_CALL(cublasSetStream(cublas_handle_[i], stream_[i]));
    CUDNN_CALL(cudnnCreate(&cudnn_handle_[i]));
    CUDNN_CALL(cudnnSetStream(cudnn_handle_[i], stream_[i]));
  }
}

GpuDevice::~GpuDevice() {
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

void GpuDevice::DoCopyRemoteData(float* dst, float* src, size_t size, int thrid) {
  CUDA_CALL(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream_[thrid]));
  CUDA_CALL(cudaStreamSynchronize(stream_[thrid]));
}

void GpuDevice::DoExecute(const DataList& in, const DataList& out, PhysicalOp& op, int thrid) {
  CudaRuntimeContext ctx;
  ctx.impl_type = ImplType::kCuda;
  ctx.stream = stream_[thrid];
  ctx.cublas_handle = cublas_handle_[thrid];
  ctx.cudnn_handle = cudnn_handle_[thrid];
  op.compute_fn->Execute(in, out, ctx);
  CUDA_CALL(cudaStreamSynchronize(stream_[thrid]));
}

#endif

CpuDevice::CpuDevice(uint64_t id, DeviceListener* l) : ThreadedDevice(id, l, kDefaultThreadNum) {
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

