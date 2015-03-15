#pragma once
#include "device/task.h"

namespace minerva {

class DeviceListener {
 public:
  virtual ~DeviceListener() = default;
  virtual void OnOperationComplete(Task*) = 0;
};

}  // namespace minerva

