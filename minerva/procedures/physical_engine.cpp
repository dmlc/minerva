#include "procedures/physical_engine.h"
#include "op/op.h"
#include "system/minerva_system.h"
#include <random>
#include <queue>

#define THREAD_NUM 4

using namespace std;

namespace minerva {

PhysicalEngine::PhysicalEngine(): thread_pool_(THREAD_NUM, this) {
  Init();
}

PhysicalEngine::~PhysicalEngine() {
  task_queue_.SignalForKill();
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

const RunnerWrapper& PhysicalEngine::GetRunnerWrapper(RunnerWrapper::ID id) {
  auto it = runners_.find(id);
  assert(it != runners_.end());
  return it->second;
}

void PhysicalEngine::Process(PhysicalDag&, std::vector<uint64_t>& targets) {
  // TODO Ignoring PhysicalDag, use MinervaSystem instead
  CommitDagChanges();
  auto ready_to_execute = FindRootNodes(targets);
  for (auto i: ready_to_execute) {
    AppendTask(i, bind(&PhysicalEngine::NodeRunner, this, placeholders::_1));
  }
  {
    // Waiting execution to complete
    for (auto i: targets) {
      unique_lock<mutex> lock(node_states_mutex_);
      if (node_states_[i].on_complete) {
        node_states_[i].on_complete->wait(lock);
        delete node_states_[i].on_complete;
        node_states_[i].on_complete = 0;
      }
    }
  }
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

void PhysicalEngine::CommitDagChanges() {
  lock_guard<mutex> lock(node_states_mutex_);
  auto& dag = MinervaSystem::Instance().physical_dag();
  // Create NodeState for new nodes. Only new nodes are inserted.
  for (auto& i: dag.index_to_node_) {
    if (node_states_.find(i.first) == node_states_.end()) {
      NodeState n;
      n.state = NodeState::kNoNeed;
      n.dependency_counter = 0;
      n.on_complete = 0;
      node_states_.insert(make_pair(i.first, n));
    }
  }
}

unordered_set<DagNode*> PhysicalEngine::FindRootNodes(const vector<uint64_t>& targets) {
  lock_guard<mutex> lock(node_states_mutex_);
  auto& dag = MinervaSystem::Instance().physical_dag();
  queue<uint64_t> ready_node_queue;
  unordered_set<DagNode*> ready_to_execute;
  for (auto i: targets) {
    // Don't push complete nodes
    if (node_states_[i].state != NodeState::kComplete) {
      ready_node_queue.push(i);
    }
  }
  while (!ready_node_queue.empty()) {
    uint64_t cur = ready_node_queue.front();
    ready_node_queue.pop();
    auto& it = node_states_[cur];
    auto node = dag.index_to_node_[cur];
    it.state = NodeState::kReady;
    it.dependency_counter = 0;
    for (auto i: node->predecessors_) {
      switch (node_states_[i->node_id_].state) {
        // Count dependency and recursively search predecessors
        case NodeState::kNoNeed:
          ready_node_queue.push(i->node_id_);
        case NodeState::kReady:
          ++it.dependency_counter;
          break;
        default:
          break;
      }
    }
    // All successors of OpNode will be set ready. Successor of an incomplete OpNode could not be complete.
    if (node->Type() == DagNode::OP_NODE) {
      for (auto i: node->successors_) {
        node_states_[i->node_id_].state = NodeState::kReady;
      }
    }
    // All predecessors are complete, or there are no predecessors at all
    if (!it.dependency_counter) {
      ready_to_execute.insert(node);
    }
  }
  for (auto i: targets) {
    if (node_states_[i].state != NodeState::kComplete) {
      node_states_[i].on_complete = new condition_variable;
    }
  }
  return ready_to_execute;
}

void PhysicalEngine::NodeRunner(DagNode* node) {
  if (node->Type() == DagNode::OP_NODE) { // OpNode
    vector<PhysicalData*> input;
    vector<PhysicalData*> output;
    for (auto i: node->predecessors_) {
      input.push_back(&(dynamic_cast<PhysicalDataNode*>(i)->data_));
    }
    for (auto i: node->successors_) { // Allocate storage for each successor
      auto& n = dynamic_cast<PhysicalDataNode*>(i)->data_;
      MinervaSystem::Instance().data_store().CreateData(n.data_id, DataStore::CPU, n.size.Prod());
      output.push_back(&n);
    }
    auto& op = dynamic_cast<PhysicalOpNode*>(node)->op_;
    auto runner_wrapper = GetRunnerWrapper(op.runner_id);
    runner_wrapper.runner(input, output, op.closure);
  } else { // DataNode
    if (!node->predecessors_.size()) { // Headless data node
      auto& data = dynamic_cast<PhysicalDataNode*>(node)->data_;
      MinervaSystem::Instance().data_store().CreateData(data.data_id, DataStore::CPU, data.size.Prod());
      auto& runner_wrapper = GetRunnerWrapper(data.generator_id);
      runner_wrapper.runner({}, {&data}, data.closure);
    }
  }
  {
    lock_guard<mutex> lock(node_states_mutex_);
    auto succ = node->successors_;
    for (auto i: succ) {
      auto state = node_states_.find(i->node_id_);
      if (state == node_states_.end()) { // New nodes, not committed yet
        continue;
      }
      // Append node if all predecessors are finished
      if (state->second.state == NodeState::kReady && !(--state->second.dependency_counter)) {
        AppendTask(i, bind(&PhysicalEngine::NodeRunner, this, placeholders::_1));
      }
    }
    node_states_[node->node_id_].state = NodeState::kComplete;
    if (node_states_[node->node_id_].on_complete) {
      node_states_[node->node_id_].on_complete->notify_all();
    }
  }
}

void PhysicalEngine::AppendTask(Task node, Callback callback) {
  task_queue_.Push(make_pair(node, callback));
}

bool PhysicalEngine::GetNewTask(thread::id id, TaskPair& task) {
  return task_queue_.Pop(task);
}

}

