#include "minerva_system.h"
#include <cstdlib>
#include <mutex>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "backend/dag/dag_scheduler.h"
#include "backend/simple_backend.h"
#include "common/cuda_utils.h"

DEFINE_bool(use_dag, true, "Use dag engine");
DEFINE_bool(no_init_glog, false, "Skip initializing Google Logging");

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
  delete backend_;
  delete device_manager_;
  delete profiler_;
  delete physical_dag_;
  google::ShutdownGoogleLogging();
}

pair<Device::MemType, float*> MinervaSystem::GetPtr(uint64_t device_id, uint64_t data_id) {
  return device_manager_->GetDevice(device_id)->GetPtr(data_id);
}

uint64_t MinervaSystem::GenerateDataId() {
  return data_id_counter_++;
}

MinervaSystem::MinervaSystem(int* argc, char*** argv) : current_device_id_(0), data_id_counter_(0) {
  cout << "init" << endl;
  gflags::ParseCommandLineFlags(argc, argv, true);
#ifndef HAS_PS
  // glog is initialized in PS::main, and also here, so we will hit a
  // double-initalize error when compiling with PS
  if (!FLAGS_no_init_glog)
    google::InitGoogleLogging((*argv)[0]);
#endif
  physical_dag_ = new PhysicalDag();
  profiler_ = new ExecutionProfiler();
  device_manager_ = new DeviceManager();
  if (FLAGS_use_dag) {
    LOG(INFO) << "dag engine enabled";
    backend_ = new DagScheduler(physical_dag_, device_manager_);
  } else {
    LOG(INFO) << "dag engine disabled";
    backend_ = new SimpleBackend(*device_manager_);
  }
}

}  // end of namespace minerva

