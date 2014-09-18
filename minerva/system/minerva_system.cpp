#include "system/minerva_system.h"
#include <glog/logging.h>
#include <cstdlib>
#include <iostream>
#include "common/thread_pool.h"
#include "op/impl/basic.h"
#include "dag/dag_printer.h"
#include "procedures/dag_scheduler.h"
#include "system/data_store.h"

using namespace std;

namespace minerva {

MinervaSystem::~MinervaSystem() {}

void MinervaSystem::Initialize(int* argc, char*** argv) {
  google::InitGoogleLogging((*argv)[0]);
  physical_dag_ = new PhysicalDag();
  dag_scheduler_ = new DagScheduler(physical_dag_);
  device_factory_ = new DeviceFactory();
  device_id_ = 0;
  LoadBuiltinDagMonitors();
}

void MinervaSystem::Finalize() {}

float* MinervaSystem::GetValue(NArray& narr) {
  narr.data_node_
  NVector<uint64_t> phy_nid = expand_engine_->GetPhysicalNodes(narr.data_node_->node_id());
  float* rstptr = new float[narr.Size().Prod()];
  Scale srcstart = Scale::Origin(narr.Size().NumDims());
  for(uint64_t nid : phy_nid) {
    PhysicalData& pdata = physical_dag_.GetDataNode(nid)->data_;
    float* srcptr = data_store_->GetData(pdata.data_id, DataStore::CPU);
    basic::NCopy(srcptr, pdata.size, srcstart,
        rstptr, narr.Size(), pdata.offset, pdata.size);
  }
  return rstptr;
}

MinervaSystem::MinervaSystem() {}


void MinervaSystem::set_device_id(uint64_t id) {
  device_id_ = id;
}

uint64_t MinervaSystem::device_id() const {
  return device_id_;
}

uint64_t MinervaSystem::CreateCPUDevice() {
  return device_factory_->CreateCPUDevice();
}

uint64_t MinervaSystem::CreateGPUDevice(int gid) {
  return device_factory_->CreateGPUDevice(gid);
}

uint64_t MinervaSystem::CreateGPUDevice(int gid, int num_stream) {
  return device_factory_->CreateGPUDevice(gid, num_stream);
}

Device* MinervaSystem::GetDevice(uint64_t id) {
  return device_factory_->GetDevice(id);
}

void MinervaSystem::LoadBuiltinDagMonitors() {
  physical_dag_->RegisterMonitor(dag_scheduler_);
}

void MinervaSystem::GeneratePhysicalDag(const std::vector<uint64_t>& lids) {
  expand_engine_->Process(logical_dag_, lids);
  expand_engine_->WaitForFinish();
  // commit extern rc change
  for(auto changed_lnid : extern_rc_changed_ldnodes_) {
    //cout << "changed_ldnode: " << changed_lnid << endl;
    auto changed_ldnode = logical_dag_.GetDataNode(changed_lnid);
    auto pnids = expand_engine_->GetPhysicalNodes(changed_ldnode->node_id());
    for(uint64_t pnid : pnids) {
      auto pnode = physical_dag_.GetDataNode(pnid);
      int changed_amount = changed_ldnode->data_.extern_rc - pnode->data_.extern_rc;
      pnode->data_.extern_rc += changed_amount;
      physical_engine_->OnIncrExternRC(pnode, changed_amount);
    }
  }
  extern_rc_changed_ldnodes_.clear();
  expand_engine_->GCNodes(logical_dag_);// GC useless logical nodes
  LOG(INFO) << "Physical dag generated";
  //cout << physical_dag().PrintDag<ExternRCPrinter>() << endl;
}

void MinervaSystem::WaitForEvalFinish() {
  physical_engine_->WaitForFinish();
  physical_engine_->GCNodes(physical_dag_);// GC useless physical nodes
}

void MinervaSystem::ExecutePhysicalDag(const std::vector<uint64_t>& pids) {
  physical_engine_->Process(physical_dag_, pids);
}

void MinervaSystem::Eval(const vector<NArray>& narrs) {
  LOG(INFO) << "Evaluation(synchronous) start...";

  std::vector<uint64_t> lid_to_eval;
  for(auto na : narrs) {
    lid_to_eval.push_back(na.data_node_->node_id());
  }
  // logical dag
  GeneratePhysicalDag(lid_to_eval);
  // physical dag
  vector<uint64_t> pid_to_eval;
  for(uint64_t id : lid_to_eval) {
    const vector<uint64_t>& pnids = expand_engine_->GetPhysicalNodes(id).ToVector();
    pid_to_eval.insert(pid_to_eval.end(), pnids.begin(), pnids.end());
  }
  ExecutePhysicalDag(pid_to_eval);
  WaitForEvalFinish();

  LOG(INFO) << "Evaluation completed!";
}

void MinervaSystem::EvalAsync(const std::vector<NArray>& narrs) {
  LOG(INFO) << "Evaluation(a-synchronous) start...";
  std::vector<uint64_t> lid_to_eval;
  for(auto na : narrs) {
    lid_to_eval.push_back(na.data_node_->node_id());
  }
  // logical dag
  GeneratePhysicalDag(lid_to_eval);
  // physical dag
  vector<uint64_t> pid_to_eval;
  for(uint64_t id : lid_to_eval) {
    const vector<uint64_t>& pnids = expand_engine_->GetPhysicalNodes(id).ToVector();
    pid_to_eval.insert(pid_to_eval.end(), pnids.begin(), pnids.end());
  }
  ExecutePhysicalDag(pid_to_eval);
}

void MinervaSystem::IncrExternRC(LogicalDag::DNode* dnode, int amount) {
  CHECK_NOTNULL(dnode);
  dnode->data_.extern_rc += amount;
  if(expand_engine_->node_states().GetState(dnode->node_id()) == NodeState::kCompleted) {
    extern_rc_changed_ldnodes_.insert(dnode->node_id());
  }
  expand_engine_->OnIncrExternRC(dnode, amount);
}

} // end of namespace minerva
