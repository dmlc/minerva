#pragma once
#include <vector>
#include <cstdint>
#include "op/physical.h"
#include "device/task_data.h"

namespace minerva {

struct Task {
  std::vector<TaskData> inputs;
  std::vector<TaskData> outputs;
  PhysicalOp op;
  // `id` is only meaningful to the issuer of the task
  uint64_t id;
};

}  // namespace minerva
