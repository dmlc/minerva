#pragma once
#include <cstdint>
#include "dag/physical_dag.h"

namespace minerva {

class DeviceListener {
 public:
  virtual void OnOperationComplete(PhysicalOpNode*) = 0;
  virtual ~DeviceListener() = default;
};

}  // namespace minerva

