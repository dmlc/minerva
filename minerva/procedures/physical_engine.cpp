#include "procedures/physical_engine.h"
#include "op/op.h"
#include "system/minerva_system.h"

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
  return 0;
  auto it = reverse_lookup_.find(name);
  assert(it != reverse_lookup_.end());
  return it->second;
}

void PhysicalEngine::Process(PhysicalDag& dag, std::vector<uint64_t>& targets) {
}

void PhysicalEngine::Init() {
  LoadBuiltinRunners();
  // Then we can load user defined runners
}

void PhysicalEngine::LoadBuiltinRunners() {
  RegisterRunner("fill", [](RunnerWrapper::Operands inputs, RunnerWrapper::Operands outputs, ClosureBase* closure_base) {
    assert(inputs.size() == 0); // This is how we define generators for now
    assert(outputs.size() == 1);
    auto& closure = GetClosureFromBase<FillClosure>(closure_base); // Do runtime checking of type
    size_t size = inputs[0]->size.Prod();
    auto data = MinervaSystem::Instance().data_store().GetData(inputs[0]->data_id, DataStore::CPU);
    for (size_t i = 0; i < size; ++i) {
      data[i] = closure.val;
    }
  });
}

}

