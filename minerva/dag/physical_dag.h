#pragma once
#include <string>
#include <sstream>
#include <mutex>
#include "dag/dag.h"
#include "op/physical.h"
#include "op/compute_fn.h"

namespace minerva {

class PhysicalDag : public Dag<PhysicalData, PhysicalOp> {
 public:
  using Dag<PhysicalData, PhysicalOp>::ToDotString;
  using Dag<PhysicalData, PhysicalOp>::ToString;
  std::string ToDotString() const override;
  std::string ToString() const override;

 private:
  static std::string DataToString(const PhysicalData&);
  static std::string OpToString(const PhysicalOp&);
};

typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}  // namespace minerva
