#include "procedures/physical_engine.h"
#include "op/op.h"
#include "system/minerva_system.h"
#include <random>

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
  if (it == reverse_lookup_.end()) {
    cout << name << " not defined";
  }
  assert(it != reverse_lookup_.end());
  return it->second;
}

RunnerWrapper PhysicalEngine::GetRunnerWrapper(RunnerWrapper::ID id) {
  auto it = runners_.find(id);
  assert(it != runners_.end());
  return it->second;
}

void PhysicalEngine::Process(PhysicalDag& dag, std::vector<uint64_t>& targets) {
}

void PhysicalEngine::Init() {
  LoadBuiltinRunners();
  // Then we can load user defined runners
}

#define LAMBDA_SIG \
  [](RunnerWrapper::Operands inputs, RunnerWrapper::Operands outputs, ClosureBase* closure_base)

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
  RegisterRunner("randn", [](RunnerWrapper::Operands inputs, RunnerWrapper::Operands outputs, ClosureBase* closure_base) {
    assert(inputs.size() == 0);
    assert(outputs.size() == 1);
    auto& closure = GetClosureFromBase<RandnClosure>(closure_base);
    size_t size = inputs[0]->size.Prod();
    auto data = MinervaSystem::Instance().data_store().GetData(inputs[0]->data_id, DataStore::CPU);
    default_random_engine generator;
    normal_distribution<float> distribution(closure.mu, closure.var); // TODO only float for now
    for (size_t i = 0; i < size; ++i) {
      data[i] = distribution(generator);
    }
  });
  RegisterRunner("matMult", LAMBDA_SIG {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    auto& left = *inputs[0];
    auto& right = *inputs[1];
    auto& res = *outputs[0];
    auto left_data = MinervaSystem::Instance().data_store().GetData(left.data_id, DataStore::CPU);
    auto right_data = MinervaSystem::Instance().data_store().GetData(right.data_id, DataStore::CPU);
    auto res_data = MinervaSystem::Instance().data_store().GetData(res.data_id, DataStore::CPU);
    int m = res.size[0];
    int n = res.size[1];
    int o = left.size[1];
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        res_data[i * n + j] = 0;
        for (int k = 0; k < o; ++k) {
          res_data[i * n + j] += left_data[i * o + k] * right_data[k * n + j];
        }
      }
    }
  });
  RegisterRunner("arithmetic", LAMBDA_SIG {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    auto& closure = GetClosureFromBase<ArithmeticClosure>(closure_base);
    auto& left = *inputs[0];
    auto& right = *inputs[1];
    auto& res = *outputs[0];
    auto left_data = MinervaSystem::Instance().data_store().GetData(left.data_id, DataStore::CPU);
    auto right_data = MinervaSystem::Instance().data_store().GetData(right.data_id, DataStore::CPU);
    auto res_data = MinervaSystem::Instance().data_store().GetData(res.data_id, DataStore::CPU);
    size_t size = res.size.Prod();
    if (closure.type == ADD) {
      for (size_t i = 0; i < size; ++i) {
        res_data[i] = left_data[i] + right_data[i];
      }
    } else if (closure.type == SUB) {
      for (size_t i = 0; i < size; ++i) {
        res_data[i] = left_data[i] - right_data[i];
      }
    } else if (closure.type == MULT) {
      for (size_t i = 0; i < size; ++i) {
        res_data[i] = left_data[i] * right_data[i];
      }
    } else if (closure.type == DIV) {
      for (size_t i = 0; i < size; ++i) {
        res_data[i] = left_data[i] / right_data[i];
      }
    } else {
      assert(false);
    }
  });
  RegisterRunner("arithmeticConstant", LAMBDA_SIG {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto& closure = GetClosureFromBase<ArithmeticConstClosure>(closure_base);
    float val = closure.val;
    auto& in = *inputs[0];
    auto& res = *outputs[0];
    auto in_data = MinervaSystem::Instance().data_store().GetData(in.data_id, DataStore::CPU);
    auto res_data = MinervaSystem::Instance().data_store().GetData(res.data_id, DataStore::CPU);
    size_t size = res.size.Prod();
    if (closure.type == ADD) {
      for (size_t i = 0; i < size; ++i) {
        res_data[i] = in_data[i] + val;
      }
    } else if (closure.type == SUB) {
      if (!closure.side) {
        for (size_t i = 0; i < size; ++i) {
          res_data[i] = val - in_data[i];
        }
      } else {
        for (size_t i = 0; i < size; ++i) {
          res_data[i] = in_data[i] - val;
        }
      }
    } else if (closure.type == MULT) {
      for (size_t i = 0; i < size; ++i) {
        res_data[i] = in_data[i] * val;
      }
    } else if (closure.type == DIV) {
      if (!closure.side) {
        for (size_t i = 0; i < size; ++i) {
          res_data[i] = val / in_data[i];
        }
      } else {
        for (size_t i = 0; i < size; ++i) {
          res_data[i] = in_data[i] / val;
        }
      }
    } else {
      assert(false);
    }
  });
  RegisterRunner("trans", LAMBDA_SIG {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto& in = *inputs[0];
    auto& res = *outputs[0];
    auto in_data = MinervaSystem::Instance().data_store().GetData(in.data_id, DataStore::CPU);
    auto res_data = MinervaSystem::Instance().data_store().GetData(res.data_id, DataStore::CPU);
    int m = res.size[0];
    int n = res.size[1];
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        res_data[i * n + j] = in_data[j * m + i];
      }
    }
  });
}

}

