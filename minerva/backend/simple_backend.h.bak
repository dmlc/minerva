#pragma once
#include <mutex>
#include <condition_variable>

#include "system/backend.h"
#include "procedures/device_listener.h"

namespace minerva {

class DeviceManager;

class SimpleBackend : public IBackend, public DeviceListener {
 public:
  SimpleBackend(DeviceManager& dm);
  std::vector<MData*> Create(const std::vector<MData*>& params, const std::vector<Scale>& result_sizes, ComputeFn* fn) override;
  //MData* RecordCreateInplace(MData* param, ComputeFn* fn) override;
  void ShallowCopy(MData*& to, MData* from) override;
  void Destroy(MData* ) override;
  void Issue(MData* ) override;
  void Wait(MData* ) override;
  //void Wait(const std::vector<MData*>& ) override;
  void WaitForAll() override;
  std::shared_ptr<float> GetValue(MData* ) override;

  void OnOperationComplete(PhysicalOpNode*) override;

 private:
  std::mutex finish_mutex_;
  std::condition_variable finish_cond_;
  DeviceManager& device_manager_;
  DISALLOW_COPY_AND_ASSIGN(SimpleBackend);
};

}
