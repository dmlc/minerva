#include "dag.h"
#include "physical_dag.h"
#include "system/minerva_system.h"
#include <sstream>

using namespace std;

namespace minerva {

string DagHelper<PhysicalData, PhysicalOp>::DataToString(const PhysicalData& d) {
  stringstream ss;
  ss << d.size;
  if (d.generator_id) {
    ss << MinervaSystem::Instance().physical_engine().GetRunnerWrapper(d.generator_id).name;
  }
  return ss.str();
}

string DagHelper<PhysicalData, PhysicalOp>::OpToString(const PhysicalOp& op) {
  return MinervaSystem::Instance().physical_engine().GetRunnerWrapper(op.runner_id).name;
}

}

