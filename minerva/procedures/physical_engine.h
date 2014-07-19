#pragma once
#include "procedures/dag_procedure.h"
#include "op/runner_wrapper.h"
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace minerva {

class PhysicalEngine: public PhysicalDagProcedure {
 public:
  PhysicalEngine& RegisterRunner(std::string, RunnerWrapper::Runner);
  RunnerWrapper::ID GetRunner(std::string);
  void Init();

 private:
  void LoadBuiltinRunners();
  std::vector<RunnerWrapper> runners_;
  std::map<std::string, RunnerWrapper::ID> reverse_lookup_;
};

}

