#pragma once
#include "dag_engine.h"
#include "system/data_store.h"
#include "impl_decider.h"

namespace minerva {

class PhysicalEngine : public DagEngine<PhysicalDag> {
 public:
  PhysicalEngine(ThreadPool& tp, DataStore& ds);
 protected:
 private:
};

}
