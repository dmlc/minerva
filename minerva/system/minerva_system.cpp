#include "system/minerva_system.h"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cstdlib>
#include <mutex>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include "common/thread_pool.h"
#include "op/impl/basic.h"
#include "dag/dag_printer.h"
#include "procedures/dag_scheduler.h"
#include "procedures/simple_backend.h"
#include "common/cuda_utils.h"

DEFINE_bool(use_dag, true, "Use dag engine");

using namespace std;

namespace minerva {

void MinervaSystem::UniversalMemcpy(pair<Device::MemType, float*> to, pair<Device::MemType, float*> from, size_t size) {
#ifdef HAS_CUDA
  CUDA_CALL(cudaMemcpy(to.second, from.second, size, cudaMemcpyDefault));
#else
  CHECK_EQ(static_cast<int>(to.first), static_cast<int>(Device::MemType::kCpu));
  CHECK_EQ(static_cast<int>(from.first), static_cast<int>(Device::MemType::kCpu));
  memcpy(to.second, from.second, size);
#endif
}

MinervaSystem::~MinervaSystem() {
  backend_->WaitForAll();
  physical_dag_->ClearMonitor();
  delete backend_;
  delete device_manager_;
  delete profiler_;
  delete physical_dag_;
  google::ShutdownGoogleLogging();
}

uint64_t MinervaSystem::CreateCpuDevice() {
  return device_manager_->CreateCpuDevice();
}

#ifdef HAS_CUDA

uint64_t MinervaSystem::CreateGpuDevice(int gid) {
  return device_manager_->CreateGpuDevice(gid);
}

int MinervaSystem::GetGpuDeviceCount() {
  return device_manager_->GetGpuDeviceCount();
}

#endif

pair<Device::MemType, float*> MinervaSystem::GetPtr(uint64_t device_id, uint64_t data_id) {
  return device_manager_->GetDevice(device_id)->GetPtr(data_id);
}

uint64_t MinervaSystem::GenerateDataId() {  // TODO contention
  static uint64_t data_id = 0;
  return data_id++;
}

MinervaSystem::MinervaSystem(int* argc, char*** argv) {
  gflags::ParseCommandLineFlags(argc, argv, true);
#ifndef HAS_PS
  // XXX workaround
  // glog is initialized in PS::main, and also here, so we will hit a 
  // double-initalize error when compiling with PS
  google::InitGoogleLogging((*argv)[0]);
#endif
  physical_dag_ = new PhysicalDag();
  profiler_ = new ExecutionProfiler();
  device_manager_ = new DeviceManager();
  if (FLAGS_use_dag) {
    LOG(INFO) << "Enable dag engine";
    backend_ = new DagScheduler(physical_dag_, device_manager_);
  } else {
    LOG(INFO) << "Disable dag engine";
    backend_ = new SimpleBackend(*device_manager_);
  }
  current_device_id_ = 0;
}

}  // end of namespace minerva
