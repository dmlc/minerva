#pragma once
#include "procedures/dag_procedure.h"
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace minerva {

class PhysicalEngine: public PhysicalDagProcedure {
 public:
  typedef uint64_t RunnerID;
  PhysicalEngine& RegisterRunner(std::string, RunnerWrapper::RunnerType);
  RunnerID GetRunner(std::string);

 private:
  std::vector<RunnerWrapper> runners_;
  std::map<std::string, RunnerID> reverse_lookup_;
};

}

