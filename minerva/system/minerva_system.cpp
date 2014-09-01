#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cstdlib>
//#include <fstream>

#include "minerva_system.h"
#include "common/thread_pool.h"
#include "op/impl/basic.h"
#include "dag/dag_printer.h"
#include "procedures/impl_decider.h"
#include "procedures/expand_engine.h"
#include "procedures/physical_engine.h"
#include "procedures/impl_decider.h"
#include "system/data_store.h"

using namespace std;

/////////////////////// flag definitions //////////////////////
static bool IsValidImplType(const char* flag, const std::string& value) {
  return strcmp(flag, "basic") || strcmp(flag, "mkl") || strcmp(flag, "cuda");
}
DEFINE_string(impl, "basic", "use basic|mkl|cuda kernels");
static const bool impl_valid = gflags::RegisterFlagValidator(&FLAGS_impl, &IsValidImplType);
static bool IsValidNumThreads(const char* flag, int n) {
  return n > 0;
}
DEFINE_int32(numthreads, 1, "number of threads used in execution");
static const bool numthreads_valid = gflags::RegisterFlagValidator(&FLAGS_numthreads, &IsValidNumThreads);
/////////////////////// member function definitions //////////////////////
namespace minerva {

void MinervaSystem::Initialize(int* argc, char*** argv) {
  google::InitGoogleLogging((*argv)[0]);
  gflags::ParseCommandLineFlags(argc, argv, true);
  ThreadPool* execute_pool = new ThreadPool(FLAGS_numthreads);
  ThreadPool* expand_pool = new ThreadPool(1);
  data_store_ = new DataStore();
  physical_engine_ = new PhysicalEngine(*execute_pool, *data_store_);
  expand_engine_ = new ExpandEngine(*expand_pool);
  static SimpleImplDecider all_basic_impl(ImplType::kBasic);
  static SimpleImplDecider all_mkl_impl(ImplType::kMkl);
  static SimpleImplDecider all_cuda_impl(ImplType::kCuda);
  if (FLAGS_impl == "mkl") {
    physical_engine_->SetImplDecider(&all_mkl_impl);
  } else if (FLAGS_impl == "cuda") {
    physical_engine_->SetImplDecider(&all_cuda_impl);
  } else {
    physical_engine_->SetImplDecider(&all_basic_impl);
  }
  LoadBuiltinDagMonitors();
  df_ = DeviceFactory::Instance();
  df_.Reset();
  device_info_ = df_.DefaultInfo();
}
void MinervaSystem::Finalize() { }
MinervaSystem::MinervaSystem() { }
MinervaSystem::~MinervaSystem() { }

void MinervaSystem::set_device_info(DeviceInfo info) {
  device_info_ = info;
}

DeviceInfo MinervaSystem::device_info() const {
  return device_info_;
}

DeviceInfo MinervaSystem::CreateGpuDevice(int gid) {
  return df_.GpuDeviceInfo(gid);
}

DeviceInfo MinervaSystem::CreateGpuDevice(int gid, int num_stream) {
  return df_.GpuDeviceInfo(gid, num_stream);
}

void MinervaSystem::LoadBuiltinDagMonitors() {
  logical_dag_.RegisterMonitor(expand_engine_);
  physical_dag_.RegisterMonitor(physical_engine_);
}

void MinervaSystem::SetImplDecider(ImplDecider* decider) {
  physical_engine_->SetImplDecider(decider);
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

float* MinervaSystem::GetValue(NArray& narr) {
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

void MinervaSystem::IncrExternRC(LogicalDag::DNode* dnode, int amount) {
  CHECK_NOTNULL(dnode);
  dnode->data_.extern_rc += amount;
  if(expand_engine_->node_states().GetState(dnode->node_id()) == NodeState::kCompleted) {
    extern_rc_changed_ldnodes_.insert(dnode->node_id());
  }
  expand_engine_->OnIncrExternRC(dnode, amount);
}

} // end of namespace minerva
