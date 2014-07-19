#pragma once
#include "op/physical_data.h"
#include "op/closure.h"
#include "op/op.h"
#include <vector>
#include <cstdint>

namespace minerva {

struct PhysicalData;

struct RunnerWrapper {
  typedef uint64_t ID;
  typedef const std::vector<PhysicalData*>& Operands;
  typedef std::function<void(Operands, Operands, ClosureBase*)> Runner;
  std::string name;
  Runner runner;
};

}

