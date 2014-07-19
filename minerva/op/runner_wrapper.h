#pragma once
#include "op/physical_data.h"
#include <vector>

namespace minerva {

struct RunnerWrapper {
  typedef std::function<void(const std::vector<PhysicalData*>&, const std::vector<PhysicalData*>&)> RunnerType;
  std::string name;
  RunnerType runner;
};

}

