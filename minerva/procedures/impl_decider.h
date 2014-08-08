#pragma once
#include "op/impl/impl.h"
#include "dag_procedure.h"

namespace minerva {

class ImplDecider : public PhysicalDagProcedure {
 public:
  virtual void Process(PhysicalDag&, const std::vector<uint64_t>&) = 0;
};

class SimpleImplDecider : public ImplDecider {
 public:
  SimpleImplDecider(IMPL_TYPE type): type(type) {}
  void Process(PhysicalDag&, const std::vector<uint64_t>&);
 private:
  IMPL_TYPE type;
};

}
