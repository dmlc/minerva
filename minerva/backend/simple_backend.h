#pragma once
#include <mutex>
#include <condition_variable>

#include "backend.h"
#include "device/device_listener.h"

namespace minerva {

class DeviceManager;

class SimpleBackend : public Backend, public DeviceListener {
 public:
  SimpleBackend(DeviceManager& dm);
  std::vector<BackendChunk*> Create(const std::vector<BackendChunk*>&, const std::vector<Scale>&, std::shared_ptr<ComputeFn>) override;
  void Wait(BackendChunk*) override;
  void WaitForAll() override;
  std::shared_ptr<float> GetValue(BackendChunk*) override;

  void OnOperationComplete(Task*) override;

 private:
  DeviceManager& device_manager_;

  std::mutex finish_mutex_;
  bool finished_flag_;
  std::condition_variable finish_cond_;

  DISALLOW_COPY_AND_ASSIGN(SimpleBackend);
};

}
