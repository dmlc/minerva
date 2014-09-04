#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "device.h"
#include "op/physical_fn.h"
#include "system/minerva_system.h"
#include <iostream>

DEFINE_bool(enable_execute, true, "enable concrete computation");

namespace minerva {

Device::Device() {}

Device::Device(DeviceInfo info) {
  device_info_ = info;
}

DeviceInfo Device::GetInfo() {
  return device_info_;
}

void Device::Execute(uint64_t nid, std::vector<PhysicalData> inputs, std::vector<PhysicalData> outputs, const PhysicalOp Op) {
  std::vector<DataShard> inputShards;
  for (std::vector<PhysicalData>::iterator input = inputs.begin(); input != inputs.end(); ++ input) {
/*
    uint64_t data_id = input->data_id;
    int size = input->size.Prod();
    if (local_data_.find(data_id) == local_data_.end()) { // data not found in this device
       uint64_t input_device_id = input->device_info.id;
       CreateData(data_id, size);
       float* local_pointer = this->GetData(data_id);
       float* remote_pointer = MinervaSystem::Instance().GetDevice(input_device_id)->GetData(data_id);
       // TODO data copy
       // cudaMemcpy(local_pointer, remote_pointer, size, ...);
       local_data_.insert(data_id);
    }
*/
    inputShards.push_back(DataShard(*input));
  }

  std::vector<DataShard> outputShards;
  for (std::vector<PhysicalData>::iterator output = outputs.begin(); output != outputs.end(); ++ output) {
    CreateData(output->data_id, output->size.Prod());
    local_data_.insert(output->data_id);
    outputShards.push_back(DataShard(*output));
  }

  CHECK_NOTNULL(Op.compute_fn);
  if (FLAGS_enable_execute) {
    DLOG(INFO) << "Execute node#" << nid << " compute fn: " << Op.compute_fn->Name();
    Op.compute_fn->Execute(inputShards, outputShards, Op.impl_type);
  }
}

void GpuDevice::CreateData(uint64_t data_id, int size) {
  DataStore& data_store_ = MinervaSystem::Instance().data_store();
  data_store_.CreateData(data_id, DataStore::GPU, size, 1);
}

void CpuDevice::CreateData(uint64_t data_id, int size) {
  DataStore& data_store_ = MinervaSystem::Instance().data_store();
  data_store_.CreateData(data_id, DataStore::CPU, size, 1);
}

float* GpuDevice::GetData(uint64_t data_id) {
  DataStore& data_store_ = MinervaSystem::Instance().data_store();
  return data_store_.GetData(data_id, DataStore::GPU);
}

float* CpuDevice::GetData(uint64_t data_id) {
  DataStore& data_store_ = MinervaSystem::Instance().data_store();
  return data_store_.GetData(data_id, DataStore::CPU);
}

}
