#pragma once
#include <string>
#include <cstdint>
#include "op/physical.h"

namespace minerva {

struct DeviceData {
  PhysicalData physical_data;
  // `id` is only meaningful to the issuer of the task
  uint64_t id;
};

}  // namespace minerva

