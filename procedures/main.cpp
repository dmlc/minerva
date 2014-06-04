#include "dag/dag.h"
#include "dag/dag_node.h"
#include "procedures/dag_engine.h"
#include <functional>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

int main() {
  minerva::Dag d;
  minerva::DagNode* nodes[5];
  minerva::DagEngine engine;
  vector<uint64_t> targets{nodes[4]->node_id()};
  engine.Process(d, targets);
  return 0;
}

