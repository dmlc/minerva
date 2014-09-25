#include "device/device.h"
#include <utility>
#include <cstdlib>
#include <glog/logging.h>
#include <sstream>
#include "system/minerva_system.h"
#include "op/context.h"
#include "common/cuda_utils.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

namespace minerva {

Device::Device(uint64_t d, DeviceListener* l) : device_id_(d), data_store_(0), listener_(l) {
}

Device::~Device() {
}

void Device::FreeDataIfExist(uint64_t id) {
  auto d = local_data_.erase(id) + remote_data_.erase(id);
  if (d == 1) {
    data_store_->FreeData(id);
  } else if (d != 0) {
    CHECK(false) << "duplicate data";
  }
}

#ifdef HAS_CUDA
GpuDevice::GpuDevice(uint64_t id, DeviceListener* l, int gid) : Device(id, l), device_(gid), pool_(1) {
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
  for (size_t i = 0; i < kDefaultStreamNum; ++i) {
    CUDA_CALL(cudaStreamCreate(&stream_[i]));
    CUBLAS_CALL(cublasCreate(&handle_[i]));
  }
}

GpuDevice::~GpuDevice() {
  pool_.WaitForAllFinished();
  for (size_t i = 0; i < kDefaultStreamNum; ++i) {
    CUBLAS_CALL(cublasDestroy(handle_[i]));
    CUDA_CALL(cudaStreamDestroy(stream_[i]));
  }
  delete data_store_;
}

void GpuDevice::PushTask(uint64_t id) {
  pool_.Push(bind(&GpuDevice::Execute, this, id));
}

pair<Device::MemType, float*> GpuDevice::GetPtr(uint64_t id) {
  return make_pair(MemType::kGpu, data_store_->GetData(id));
}

string GpuDevice::Name() const {
  stringstream ss;
  ss << "GPU device " << device_id_;
  return ss.str();
}

size_t GpuDevice::RoundRobinAlloc() {
  static int s = 0;
  int ret = s;
  s = (s + 1) % kDefaultStreamNum;
  return ret;
}

struct CallbackData {
  DeviceListener* listener;
  uint64_t id;
};

void CUDART_CB cudaStreamCallback(cudaStream_t, cudaError_t, void* user_data) {
  CallbackData* d = reinterpret_cast<CallbackData*>(user_data);
  d->listener->OnOperationComplete(d->id);
  delete d;
}

void GpuDevice::Execute(uint64_t nid) {
  auto node = MinervaSystem::Instance().physical_dag().GetNode(nid);
  if (node->Type() == DagNode::NodeType::kDataNode) {  // Data node
    listener_->OnOperationComplete(node->node_id());
  } else {  // Op node
    auto op_node = CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(node));
    auto rr = RoundRobinAlloc();
    DataList input_shards;
    for (auto i : op_node->inputs_) {
      auto& r_data = i->data_;
      if (r_data.device_id == device_id_) {  // Input is local
        CHECK_EQ(local_data_.count(r_data.data_id), 1);
      } else if (!remote_data_.count(r_data.data_id)) {  // Input is remote and not copied
        DLOG(INFO) << "GPU device input node #" << i->node_id() << " is remote and to be copied";
        size_t size = r_data.size.Prod() * sizeof(float);
        auto ptr = data_store_->CreateData(r_data.data_id, size);
        CUDA_CALL(cudaMemcpyAsync(ptr, MinervaSystem::Instance().GetPtr(r_data.device_id, r_data.data_id).second, size, cudaMemcpyDefault, stream_[rr]));
        CHECK(remote_data_.insert(r_data.data_id).second);
      }
      input_shards.emplace_back(data_store_->GetData(r_data.data_id), r_data.size);
    }
    DataList output_shards;
    for (auto i : op_node->outputs_) {
      size_t size = i->data_.size.Prod() * sizeof(float);
      auto ptr = data_store_->CreateData(i->data_.data_id, size);
      CHECK(local_data_.insert(i->data_.data_id).second);
      output_shards.emplace_back(ptr, i->data_.size);
    }
    auto& op = op_node->op_;
    CHECK_NOTNULL(op.compute_fn);
    DLOG(INFO) << "GPU device execute node #" << nid << ": " << op.compute_fn->Name();
    CudaRuntimeContext ctx;
    ctx.impl_type = ImplType::kCuda;
    ctx.stream = stream_[rr];
    ctx.handle = handle_[rr];
    op.compute_fn->Execute(input_shards, output_shards, ctx);
    CallbackData* d = new CallbackData{listener_, nid};
    CUDA_CALL(cudaStreamAddCallback(stream_[rr], cudaStreamCallback, d, 0));
  }
}

#endif

CpuDevice::CpuDevice(uint64_t id, DeviceListener* l) : Device(id, l), pool_(kDefaultThreadNum) {
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

void CpuDevice::PushTask(uint64_t id) {
  pool_.Push(bind(&CpuDevice::Execute, this, id));
}

pair<Device::MemType, float*> CpuDevice::GetPtr(uint64_t id) {
  return make_pair(MemType::kCpu, data_store_->GetData(id));
}

string CpuDevice::Name() const {
  stringstream ss;
  ss << "CPU device " << device_id_;
  return ss.str();
}

void CpuDevice::Execute(uint64_t nid) {
  auto node = MinervaSystem::Instance().physical_dag().GetNode(nid);
  if (node->Type() == DagNode::NodeType::kOpNode) {  // Op node
    auto op_node = CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(node));
    DataList input_shards;
    for (auto i : op_node->inputs_) {
      auto& r_data = i->data_;
      if (r_data.device_id == device_id_) {  // Input is local
        DLOG(INFO) << "CPU device input node #" << i->node_id() << " data #" << r_data.data_id << " is local";
        CHECK_EQ(local_data_.count(r_data.data_id), 1);
      } else if (!remote_data_.count(r_data.data_id)) { // Input is remote and not copied
        DLOG(INFO) << "CPU device input node #" << i->node_id() << " is remote and not copied";
        size_t size = r_data.size.Prod() * sizeof(float);
        auto ptr = data_store_->CreateData(r_data.data_id, size);
#ifdef HAS_CUDA
        CUDA_CALL(cudaMemcpy(ptr, MinervaSystem::Instance().GetPtr(r_data.device_id, r_data.data_id).second, size, cudaMemcpyDefault));
#else
        memcpy(ptr, MinervaSystem::Instance().GetPtr(r_data.device_id, r_data.data_id).second, size);
#endif
        CHECK(remote_data_.insert(r_data.data_id).second);
      }
      input_shards.emplace_back(data_store_->GetData(i->data_.data_id), i->data_.size);
    }
    DataList output_shards;
    for (auto i : op_node->outputs_) {
      size_t size = i->data_.size.Prod() * sizeof(float);
      DLOG(INFO) << "create output data for node #" << i->node_id();
      auto ptr = data_store_->CreateData(i->data_.data_id, size);
      CHECK(local_data_.insert(i->data_.data_id).second);
      output_shards.emplace_back(ptr, i->data_.size);
    }
    auto& op = op_node->op_;
    CHECK_NOTNULL(op.compute_fn);
    DLOG(INFO) << "CPU device execute node #" << nid << ": " << op.compute_fn->Name();
    Context ctx;
    ctx.impl_type = ImplType::kBasic;
    op.compute_fn->Execute(input_shards, output_shards, ctx);
  }
  listener_->OnOperationComplete(nid);
}

}  // namespace minerva

