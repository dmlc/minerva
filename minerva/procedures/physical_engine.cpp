#include "physical_engine.h"
#include "op/context.h"
#include <gflags/gflags.h>

DEFINE_bool(enable_execute, true, "enable concrete computation");

using namespace std;

namespace minerva {
  
PhysicalEngine::PhysicalEngine(ThreadPool& tp, DataStore& ds): DagEngine<PhysicalDag>(tp), data_store_(ds) {
}
  
}
