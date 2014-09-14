#pragma once
#include <cstdint>

namespace minerva {

class DeviceListener {
 public:
  virtual void OnOperationComplete(uint64_t) = 0;
};

}  // namespace minerva

