#include <memory>

#include <glog/logging.h>
#include "simple_backend.h"
#include "device/device_manager.h"
#include "op/physical.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

class SimpleChunk : public BackendChunk {
 public:
  SimpleChunk(std::shared_ptr<PhysicalData> data): data_(data) {}
  const Scale& shape() const override { return data_->size; }
  BackendChunk* ShallowCopy() const override { return new SimpleChunk(data_); }
  PhysicalData& data() { return *data_; }
 private:
  std::shared_ptr<PhysicalData> data_;
};

SimpleBackend::SimpleBackend(DeviceManager& dm): device_manager_(dm) {
  finished_flag_.store(false);
  device_manager_.RegisterListener(this);
}

std::vector<BackendChunk*> SimpleBackend::Create(const std::vector<BackendChunk*>& input,
    const std::vector<Scale>& result_sizes, std::shared_ptr<ComputeFn> fn) {
  auto current_device_id = MinervaSystem::Instance().current_device_id();
  std::vector<BackendChunk*> result_chunks;
  Task* task = new Task();
  task->light = true;
  for (auto i : input) {
    auto c = CHECK_NOTNULL(dynamic_cast<SimpleChunk*>(i));
    task->inputs.emplace_back(c->data(), 0);
  }
  for (auto s : result_sizes) {
    auto data_id = MinervaSystem::Instance().GenerateDataId();
    std::shared_ptr<PhysicalData> data_ptr( new PhysicalData(s, current_device_id, data_id),
       [&] (PhysicalData* d) { device_manager_.FreeData(d->data_id); delete d; } );
    SimpleChunk* o = new SimpleChunk(data_ptr);
    result_chunks.emplace_back(o);
    task->outputs.emplace_back(o->data(), 0);
  }
  task->op = PhysicalOp{fn, current_device_id};
  task->id = 0;
  DLOG(INFO) << "executing task name=" << fn->Name() << " to device #" << current_device_id;
  // wait for finish
//  unique_lock<mutex> ul(finish_mutex_);
//  finished_flag_.store(false);;
  device_manager_.GetDevice(current_device_id)->PushTask(task);
//  while(! finished_flag_.load()) {
//    finish_cond_.wait(ul);
//  }
  return result_chunks;
}

void SimpleBackend::Wait(BackendChunk*) {
  // do nothing
}

void SimpleBackend::WaitForAll() {
  // do nothing
}

std::shared_ptr<float> SimpleBackend::GetValue(BackendChunk* chunk) {
  auto& data = CHECK_NOTNULL(dynamic_cast<SimpleChunk*>(chunk))->data();
  shared_ptr<float> ret(new float[data.size.Prod()], [](float* p) {
    delete[] p;
  });
  auto dev_pair = MinervaSystem::Instance().GetPtr(data.device_id, data.data_id);
  MinervaSystem::UniversalMemcpy(make_pair(Device::MemType::kCpu, ret.get()), dev_pair, data.size.Prod() * sizeof(float));
  return ret;
}

void SimpleBackend::OnOperationComplete(Task* task) {
  //lock_guard<mutex> ul(finish_mutex_);
  //std::cout << "Callback flag=" << finished_flag_ << std::endl;
//  finished_flag_.store(true);
  //finish_cond_.notify_all();
  delete task;
}

}
