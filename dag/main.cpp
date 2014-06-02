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
  for (int i = 0; i < 5; ++i) {
    nodes[i] = d.NewDataNode();
  }
  // nodes[2] is never used!
  nodes[3]->AddParents({nodes[0], nodes[1]});
  nodes[4]->AddParent(nodes[3]);
  minerva::DagEngine engine;
  vector<uint64_t> targets{nodes[4]->node_id()};
  engine.Process(d, targets);
  this_thread::sleep_for(chrono::seconds(3));
  return 0;
}
