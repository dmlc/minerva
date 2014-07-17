#include "procedures/physical_engine.h"

using namespace std;

namespace minerva {

PhysicalEngine& PhysicalEngine::RegisterRunner(string name, RunnerWrapper::RunnerType runner) {
  assert(reverse_lookup_.find(name) == reverse_lookup_.end());
  RunnerWrapper runner_wrapper;
  runner_wrapper.name = name;
  runner_wrapper.runner = runner;
  runners_.push_back(runner_wrapper);
  reverse_lookup_[name] = runners_.size() - 1;
  return *this;
}

PhysicalEngine::RunnerID PhysicalEngine::GetRunner(string name) {
  auto it = reverse_lookup_.find(name);
  assert(it != reverse_lookup_.end());
  return it->second;
}

}

