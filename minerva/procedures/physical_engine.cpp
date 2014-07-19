#include "procedures/physical_engine.h"

using namespace std;

namespace minerva {

PhysicalEngine::PhysicalEngine() {
  Init();
}

PhysicalEngine::~PhysicalEngine() {
}

PhysicalEngine& PhysicalEngine::RegisterRunner(string name, RunnerWrapper::Runner runner) {
  assert(reverse_lookup_.find(name) == reverse_lookup_.end());
  RunnerWrapper runner_wrapper;
  RunnerWrapper::ID index = ++index_;
  runner_wrapper.name = name;
  runner_wrapper.runner = runner;
  runners_[index] = runner_wrapper;
  reverse_lookup_[name] = index;
  return *this;
}

RunnerWrapper::ID PhysicalEngine::GetRunnerID(string name) {
  auto it = reverse_lookup_.find(name);
  assert(it != reverse_lookup_.end());
  return it->second;
}

void PhysicalEngine::Init() {
  LoadBuiltinRunners();
  // Then we can load user defined runners
}

void PhysicalEngine::LoadBuiltinRunners() {
  RegisterRunner("add", [](RunnerWrapper::Operands inputs, RunnerWrapper::Operands outputs, Closure* closure) {
    assert(outputs.size() == 1);
  });
}

}

