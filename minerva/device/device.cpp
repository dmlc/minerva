#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "device.h"
#include "system/minerva_system.h"
#include "op/context.h"
#include <iostream>
#ifdef HAS_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#endif

DEFINE_bool(enable_execute, true, "enable concrete computation");

namespace minerva {

Device::Device() {}

Device::Device(uint64_t device_id, DeviceInfo info) {
  device_id_ = device_id;
  device_info_ = info;
}

DeviceInfo Device::GetInfo() {
  return device_info_;
}

void Device::CopyData(void * dst, void * src, size_t numBytes)
{
	// TODO: make sure the code works without cuda 6.0
#ifdef HAS_CUDA
#if CUDA_VERSION < 6000
    assert(0);
#endif
    cudaMemcpy(dst, src, numBytes, cudaMemcpyDefault);
#else
    memcpy(dst, src, numBytes);
#endif
}

void Device::Execute(uint64_t nid, std::vector<PhysicalData> inputs, std::vector<PhysicalData> outputs, const PhysicalOp Op) {
  std::vector<DataShard> inputShards;
  for (std::vector<PhysicalData>::iterator input = inputs.begin(); input != inputs.end(); ++ input) {
    uint64_t data_id = input->data_id;
    if (local_data_.find(data_id) == local_data_.end()) { // data not found in this device
      uint64_t input_device_id = input->device_id;
      int size = input->size.Prod();
      CreateData(data_id + 10000, size);
      float* local_pointer = this->GetData(data_id + 10000);
      float* remote_pointer = MinervaSystem::Instance().GetDevice(input_device_id)->GetData(data_id);
      CopyData(local_pointer, remote_pointer, size*sizeof(float));
      DLOG(INFO) << "Data copy from device " << input_device_id << " to device " << device_id_;
      local_data_.insert(data_id);
      inputShards.push_back(DataShard(local_pointer, input->size, input->offset));
    }
    else
      inputShards.push_back(DataShard(this->GetData(data_id), input->size, input->offset));
  }

  std::vector<DataShard> outputShards;
  for (std::vector<PhysicalData>::iterator output = outputs.begin(); output != outputs.end(); ++ output) {
    CreateData(output->data_id, output->size.Prod());
    float* data = this->GetData(output->data_id);
    local_data_.insert(output->data_id);
    outputShards.push_back(DataShard(data, output->size, output->offset));
  }

  CHECK_NOTNULL(Op.compute_fn);
  if (FLAGS_enable_execute) {
    DLOG(INFO) << "Execute node#" << nid << " compute fn: " << Op.compute_fn->Name();
    Execute_Op(inputShards, outputShards, Op);
  }
}

GpuDevice::GpuDevice(uint64_t id, DeviceInfo info) 
  : Device(id, info), stream_ptr_(0){
  streams_.resize(num_streams_);
  cublas_handles_.resize(num_streams_);
  for (int i = 0; i < num_streams_; i++)
  {
    cudaStreamCreate(&streams_[i]);
    cublasCreate(&cublas_handles_[i]);
    cublasSetStream(cublas_handles_[i], streams_[i]);
	cublasSetPointerMode(cublas_handles_[i], CUBLAS_POINTER_MODE_DEVICE);
  }
  cudaMalloc((void**)&gpu_buffer_one_, sizeof(float));
  float one = 1.0;
  cudaMemcpy(gpu_buffer_one_, &one, sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gpu_buffer_zero_, sizeof(float));
  float zero = 1.0;
  cudaMemcpy(gpu_buffer_zero_, &zero, sizeof(float), cudaMemcpyHostToDevice);  
}

GpuDevice::~GpuDevice() {
  for (int i = 0; i < num_streams_; i++)
  {
    cublasDestroy(cublas_handles_[i]);
    cudaStreamDestroy(streams_[i]);
  }
  cudaFree(gpu_buffer_one_);
  cudaFree(gpu_buffer_zero_);
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

void CpuDevice::Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op) {
  Context cxt;
  cxt.impl_type = Op.impl_type;
  Op.compute_fn->Execute(inputShards, outputShards, cxt);
}

void GpuDevice::Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op) {
#ifdef HAS_CUDA
  CudaRuntimeContext cxt;
  cxt.impl_type = ImplType::kCuda;
  cxt.stream = streams_[stream_ptr_];
  cxt.handle = cublas_handles_[stream_ptr_];
  cxt.one = gpu_buffer_one_;
  cxt.zero = gpu_buffer_zero_;
  stream_ptr_++;
  Op.compute_fn->Execute(inputShards, outputShards, cxt);
#endif
}

}

