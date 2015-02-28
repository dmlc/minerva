#include "system/minerva_system.h"
#include <glog/logging.h>
#include <cstdlib>
#include <mutex>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include "common/thread_pool.h"
#include "op/impl/basic.h"
#include "dag/dag_printer.h"
#include "procedures/dag_scheduler.h"
#include "common/cuda_utils.h"

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
  dag_scheduler_->WaitForFinish();
  physical_dag_->ClearMonitor();
  delete device_manager_;
  delete profiler_;
  delete dag_scheduler_;
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

shared_ptr<float> MinervaSystem::GetValue(const NArray& narr) {
  auto& data = narr.data_node_->data_;
  shared_ptr<float> ret(new float[data.size.Prod()], [](float* p) {
    delete[] p;
  });
  MinervaSystem::UniversalMemcpy(make_pair(Device::MemType::kCpu, ret.get()), GetPtr(data.device_id, data.data_id), data.size.Prod() * sizeof(float));
  return ret;
}

pair<Device::MemType, float*> MinervaSystem::GetPtr(uint64_t device_id, uint64_t data_id) {
  return device_manager_->GetDevice(device_id)->GetPtr(data_id);
}

void MinervaSystem::IncrExternRC(PhysicalDataNode* node) {
  lock_guard<recursive_mutex> lck(physical_dag_->m_);
  ++(node->data_.extern_rc);
  dag_scheduler_->OnExternRCUpdate(node);
}

void MinervaSystem::DecrExternRC(PhysicalDataNode* node) {
  lock_guard<recursive_mutex> lck(physical_dag_->m_);
  --(node->data_.extern_rc);
  dag_scheduler_->OnExternRCUpdate(node);
}

void MinervaSystem::WaitForEval(const vector<NArray>& narrs) {
  LOG(INFO) << "evaluation (synchronous) start...";
  vector<uint64_t> pid_to_eval = Map<uint64_t>(narrs, [](const NArray& n) {
    return n.data_node_->node_id();
  });
  ExecutePhysicalDag(pid_to_eval);
  for (auto i : pid_to_eval) {
    dag_scheduler_->WaitForFinish(i);
  }
  dag_scheduler_->GCNodes();
  LOG(INFO) << "Evaluation completed!";
}

void MinervaSystem::StartEval(const vector<NArray>& narrs) {
  LOG(INFO) << "evaluation (asynchronous) start...";
  vector<uint64_t> pid_to_eval = Map<uint64_t>(narrs, [](const NArray& n) {
    return n.data_node_->node_id();
  });
  ExecutePhysicalDag(pid_to_eval);
}

uint64_t MinervaSystem::GenerateDataId() {
  static uint64_t data_id = 0;
  return data_id++;
}

MinervaSystem::MinervaSystem(int* argc, char*** argv) {
#ifndef HAS_PS
  // workaround
  // glog is initialized in PS::main, and also here, so we will hit a 
  // double-initalize error when compiling with PS
  google::InitGoogleLogging((*argv)[0]);
#endif
  physical_dag_ = new PhysicalDag();
  dag_scheduler_ = new DagScheduler(physical_dag_);
  profiler_ = new ExecutionProfiler();
  device_manager_ = new DeviceManager(dag_scheduler_);
  current_device_id_ = 0;
  LoadBuiltinDagMonitors();
}

void MinervaSystem::LoadBuiltinDagMonitors() {
  physical_dag_->RegisterMonitor(dag_scheduler_);
}

void MinervaSystem::ExecutePhysicalDag(const vector<uint64_t>& pids) {
  dag_scheduler_->Process(pids);
}

}  // end of namespace minerva
