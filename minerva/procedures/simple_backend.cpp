#include <memory>

#include "glog/logging.h"
#include "procedures/simple_backend.h"
#include "device/device_manager.h"
#include "op/physical.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

class MDataSimple : public MData {
 public:
  MDataSimple(const PhysicalData& data): data(data) {}
  const Scale& shape() const { return data.size; }
  PhysicalData data;
};

SimpleBackend::SimpleBackend(DeviceManager& dm): device_manager_(dm) {
  dm.RegisterListener(this);
}

vector<MData*> SimpleBackend::Create(const vector<MData*>& params, const vector<Scale>& result_sizes, ComputeFn* fn) {
  static uint64_t node_id_gen = 0;
  auto current_device_id = MinervaSystem::Instance().current_device_id_;
  PhysicalOpNode* op_node = new PhysicalOpNode(node_id_gen++);
  // make op
  op_node->op_.compute_fn = fn;
  op_node->op_.device_id = current_device_id;
  // make inputs
  for(size_t i = 0; i < params.size(); ++i) {
    MDataSimple* mds = CHECK_NOTNULL(dynamic_cast<MDataSimple*>(params[i]));
    op_node->inputs_.push_back(new PhysicalDataNode(0/*fake id*/, mds->data));
  }
  // make outputs
  vector<PhysicalData> rst_phy_data;
  for(size_t i = 0; i < result_sizes.size(); ++i) {
    rst_phy_data.push_back(PhysicalData(result_sizes[i], current_device_id, MinervaSystem::Instance().GenerateDataId()));
    op_node->outputs_.push_back(new PhysicalDataNode(0/*fake id*/, rst_phy_data[i]));
  }
  // call device to execute
  LOG(INFO) << "Executing function: " << fn->Name();
  device_manager_.GetDevice(current_device_id)->PushTask(op_node);
  // wait for finish
  {
    unique_lock<mutex> ul(finish_mutex_);
    finish_cond_.wait(ul);
  }
  // delete temporary nodes
  for (auto ptr : op_node->inputs_) {
    delete ptr;
  }
  for (auto ptr : op_node->outputs_) {
    delete ptr;
  }
  delete op_node;
  //delete fn; XXX [This has been deleted in the above line]
  return Map<MData*>(rst_phy_data, [](const PhysicalData& phd) { return new MDataSimple(phd); });
}
//MData* RecordCreateInplace(MData* param, ComputeFn* fn) { }
void SimpleBackend::ShallowCopy(MData*& to, MData* from) {
  if(to) {
    Destroy(to);
  }
  to = from;
  ++(CHECK_NOTNULL(dynamic_cast<MDataSimple*>(from))->data.extern_rc);
}
void SimpleBackend::Destroy(MData* md) {
  if(md) {
    auto& pdata = CHECK_NOTNULL(dynamic_cast<MDataSimple*>(md))->data;
    if(--pdata.extern_rc == 0) {
      // release the storage
      device_manager_.FreeData(pdata.data_id);
      // delete
      delete md;
    }
  }
}
void SimpleBackend::Issue(MData* ) {
  // Do nothing
}
void SimpleBackend::Wait(MData* ) {
  // Do nothing
}
//void Wait(const vector<MData*>& ) { }
void SimpleBackend::WaitForAll() {
  // Do nothing
}
shared_ptr<float> SimpleBackend::GetValue(MData* md) {
  auto& data = dynamic_cast<MDataSimple*>(md)->data;
  shared_ptr<float> ret(new float[data.size.Prod()], [](float* p) {
    delete[] p;
  });
  auto dev_pair = MinervaSystem::Instance().GetPtr(data.device_id, data.data_id);
  MinervaSystem::UniversalMemcpy(make_pair(Device::MemType::kCpu, ret.get()), dev_pair, data.size.Prod() * sizeof(float));
  return ret;
}
void SimpleBackend::OnOperationComplete(PhysicalOpNode*) {
  lock_guard<mutex> ul(finish_mutex_);
  finish_cond_.notify_all();
}

}
