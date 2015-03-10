#pragma once
#include <string>
#include <cstdint>
#include "op/physical.h"

namespace minerva {

struct TaskData {
  TaskData(const PhysicalData& p, uint64_t i) : physical_data(p), id(i) {
  }
  PhysicalData physical_data;
  // `id` is only meaningful to the issuer of the task
  uint64_t id;
};

}  // namespace minerva

