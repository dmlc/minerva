#include "device/device.h"
#include <utility>
#include <cstdlib>
#include <glog/logging.h>
#include <sstream>
#include "system/minerva_system.h"
#include "op/context.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

namespace minerva {

Device::Device(uint64_t d, DeviceListener* l) : device_id_(d), data_store_(0), listener_(l) {
}

Device::~Device() {
}

#ifdef HAS_CUDA

GpuDevice::GpuDevice(uint64_t id, DeviceListener* l, int gid) : Device(id, l), pool_(1), device_(gid) {
  CHECK_EQ(cudaSetDevice(device_), cudaSuccess);
  cudaFree(0);
  auto allocator = [](size_t len) -> void* {
    void* ret;
    CHECK_EQ(cudaMalloc(&ret, len), cudaSuccess);
    return ret;
  };
  auto deallocator = [](void* ptr) {
    CHECK_EQ(cudaFree(ptr), cudaSuccess);
  };
  data_store_ = new DataStore(allocator, deallocator);
  for (int i = 0; i < kDefaultStreamNum; ++i) {
    CHECK_EQ(cudaStreamCreate(&stream_[i]), cudaSuccess);
  }
}

GpuDevice::~GpuDevice() {
  pool_.WaitForAllFinished();
  for (int i = 0; i < kDefaultStreamNum; ++i) {
    CHECK_EQ(cudaStreamDestroy(stream_[i]), cudaSuccess);
  }
  delete data_store_;
}

void GpuDevice::PushTask(uint64_t id) {
  pool_.Push(bind(&GpuDevice::Execute, this, id));
}

pair<MemType, void*> GpuDevice::GetPtr(uint64_t id) {
  return make_pair(MemType::kGpu, data_store_->GetData(id));
}

string GpuDevice::Name() const {
  stringstream ss;
  ss << "GPU device " << device_id_;
  return ss.str();
}

void GpuDevice::GetSomeStream() {
  static int s = 0;
  int ret = s;
  s = (++s) % kDefaultStreamNum;
  return ret;
}

void GpuDevice::Execute(uint64_t nid) {
  auto node = MinervaSystem::Instance().physical_dag().GetNode(nid);
  if (node->Type() == DagNode::NodeType::kDataNode) {  // Data node
    auto data_node = dynamic_cast<PhysicalDataNode*>(node);
    CHECK_NOTNULL(data_node);
    auto& data = data_node->data_;
    if (data.device_id == device_id_) {  // Local
      DLOG(INFO) << "GPU device input data #" << nid << " is local";
      CHECK_EQ(local_data_.count(data.data_id), 1);
    } else if (!remote_data_.count(data.data_id)){  // Remote and not copied
      DLOG(INFO) << "GPU device input data #" << nid << " is remote";
      size_t size = data.size.Prod() * sizeof(float);
      auto ptr = data_store_->CreateData(data.data_id, size);
      auto stream = GetSomeStream();
      CHECK_EQ(cudaMemcpyAsync(ptr, MinervaSystem::Instance().GetPtr(data.device_id, data.data_id).second, size, cudaMemcpyDefault, stream), cudaSuccess);
      remote_data_.insert(data.data_id);
      cudaStreamAddCallback(stream, [&](cudaStream_t, cudaError_t, void*) {
        listener_->OnOperationComplete(nid);
      }, 0, 0);
    }
  } else {
    auto op_node = dynamic_cast<PhysicalOpNode*>(node);
    CHECK_NOTNULL(op_node);
    DataList input_shards;
    for (auto i : op_node->inputs_) {
      if (i->data_.device_id == device_id_) {  // Input is local
        CHECK_EQ(local_data_.count(i->data_.data_id), 1);
      } else { // Input is remote
        CHECK_EQ(remote_data_.count(i->data_.data_id), 1);
      }
      input_shards.emplace_back(data_store_->GetData(i->data_.data_id), i->data_.size);
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
    ctx.stream = GetSomeStream();
    op.compute_fn->Execute(input_shards, output_shards, ctx);
    cudaStreamAddCallback(stream, [&](cudaStream_t, cudaError_t, void*) {
      listener_->OnOperationComplete(nid);
    }, 0, 0);
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
  if (node->Type() == DagNode::NodeType::kDataNode) {  // Data node
    auto data_node = dynamic_cast<PhysicalDataNode*>(node);
    CHECK_NOTNULL(data_node);
    auto& data = data_node->data_;
    if (data.device_id == device_id_) {  // Local
      DLOG(INFO) << "CPU device input data #" << nid << " is local";
      CHECK_EQ(local_data_.count(data.data_id), 1);
    } else if (!remote_data_.count(data.data_id)){  // Remote and not copied
      DLOG(INFO) << "CPU device input data #" << nid << " is remote";
      size_t size = data.size.Prod() * sizeof(float);
      auto ptr = data_store_->CreateData(data.data_id, size);
#ifdef HAS_CUDA
      CHECK_EQ(cudaMemcpy(ptr, MinervaSystem::Instance().GetPtr(data.device_id, data.data_id), size, cudaMemcpyDefault), cudaSuccess);
#else
      memcpy(ptr, MinervaSystem::Instance().GetPtr(data.device_id, data.data_id).second, size);
#endif
      remote_data_.insert(data.data_id);
    }
  } else {
    auto op_node = dynamic_cast<PhysicalOpNode*>(node);
    CHECK_NOTNULL(op_node);
    DataList input_shards;
    for (auto i : op_node->inputs_) {
      if (i->data_.device_id == device_id_) {  // Input is local
        CHECK_EQ(local_data_.count(i->data_.data_id), 1);
      } else { // Input is remote
        CHECK_EQ(remote_data_.count(i->data_.data_id), 1);
      }
      input_shards.emplace_back(data_store_->GetData(i->data_.data_id), i->data_.size);
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
    DLOG(INFO) << "CPU device execute node #" << nid << ": " << op.compute_fn->Name();
    Context ctx;
    ctx.impl_type = ImplType::kBasic;
    op.compute_fn->Execute(input_shards, output_shards, ctx);
  }
  listener_->OnOperationComplete(nid);
}

}  // namespace minerva

