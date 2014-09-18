#include "gpu_device.h"

namespace minerva {

GpuDevice::GpuDevice() : Device() {}

GpuDevice::GpuDevice(uint64_t id, DeviceInfo info) : Device(id, info) {}

void GpuDevice::CreateData(uint64_t data_id, int size) {
  //DataStore& data_store_ = MinervaSystem::Instance().data_store();
  data_store_->CreateData(data_id, DataStore::GPU, size, 1);
}

float* GpuDevice::GetData(uint64_t data_id) {
  //DataStore& data_store_ = MinervaSystem::Instance().data_store();
  return data_store_->GetData(data_id, DataStore::GPU);
}

void GpuDevice::Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op) {
#ifdef HAS_CUDA
  Context cxt;
  cxt.impl_type = Op.impl_type;
  Op.compute_fn->Execute(inputShards, outputShards, cxt);
#endif
}


}

