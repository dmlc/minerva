#include "system/minerva_system.h"
#include <glog/logging.h>
#include <cstdlib>
#include <iostream>
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif
#include "common/thread_pool.h"
#include "op/impl/basic.h"
#include "dag/dag_printer.h"
#include "procedures/dag_scheduler.h"

using namespace std;

namespace minerva {

void MinervaSystem::UniversalMemcpy(pair<Device::MemType, float*> to, pair<Device::MemType, float*> from, size_t size) {
#ifdef HAS_CUDA
  CHECK_EQ(cudaMemcpy(to.second, from.second, size, cudaMemcpyDefault), cudaSuccess);
#else
  CHECK_EQ(static_cast<int>(to.first), static_cast<int>(Device::MemType::kCpu));
  CHECK_EQ(static_cast<int>(from.first), static_cast<int>(Device::MemType::kCpu));
  memcpy(to.second, from.second, size);
#endif
}

MinervaSystem::~MinervaSystem() {
}

void MinervaSystem::Initialize(int* argc, char*** argv) {
  google::InitGoogleLogging((*argv)[0]);
  physical_dag_ = new PhysicalDag();
  dag_scheduler_ = new DagScheduler(physical_dag_);
  device_factory_ = new DeviceFactory(dag_scheduler_);
  current_device_id_ = -1;
  LoadBuiltinDagMonitors();
}

void MinervaSystem::Finalize() {
  physical_dag_->ClearMonitor();
  delete device_factory_;
  delete dag_scheduler_;
  delete physical_dag_;
}

uint64_t MinervaSystem::CreateCPUDevice() {
  return device_factory_->CreateCpuDevice();
}

#ifdef HAS_CUDA

uint64_t MinervaSystem::CreateGPUDevice(int gid) {
  return device_factory_->CreateGpuDevice(gid);
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
  return device_factory_->GetDevice(device_id)->GetPtr(data_id);
}

void MinervaSystem::IncrExternRC(PhysicalDataNode* node) {
  ++(node->data_.extern_rc);
  dag_scheduler_->OnExternRCUpdate(node);
}

void MinervaSystem::DecrExternRC(PhysicalDataNode* node) {
  --(node->data_.extern_rc);
  dag_scheduler_->OnExternRCUpdate(node);
}

void MinervaSystem::Eval(const vector<NArray>& narrs) {
  LOG(INFO) << "Evaluation(synchronous) start...";
  vector<uint64_t> pid_to_eval = Map<uint64_t>(narrs, [](const NArray& n) {
    return n.data_node_->node_id();
  });
  ExecutePhysicalDag(pid_to_eval);
  WaitForEvalFinish();
  LOG(INFO) << "Evaluation completed!";
}

void MinervaSystem::EvalAsync(const vector<NArray>& narrs) {
  LOG(INFO) << "Evaluation(a-synchronous) start...";
  vector<uint64_t> pid_to_eval = Map<uint64_t>(narrs, [](const NArray& n) {
    return n.data_node_->node_id();
  });
  ExecutePhysicalDag(pid_to_eval);
}

void MinervaSystem::WaitForEvalFinish() {
  dag_scheduler_->WaitForFinish();
  dag_scheduler_->GCNodes();  // GC useless physical nodes
}

uint64_t MinervaSystem::GenerateDataId() {
  static uint64_t data_id = 0;
  return data_id++;
}

MinervaSystem::MinervaSystem() {
}

void MinervaSystem::LoadBuiltinDagMonitors() {
  physical_dag_->RegisterMonitor(dag_scheduler_);
}

void MinervaSystem::ExecutePhysicalDag(const std::vector<uint64_t>& pids) {
  dag_scheduler_->Process(pids);
}

}  // end of namespace minerva
