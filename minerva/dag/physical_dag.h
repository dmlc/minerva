#pragma once
#include <string>
#include <sstream>
#include "dag/dag.h"
#include "op/physical.h"
#include "op/physical_fn.h"

namespace minerva {

class PhysicalDag : public Dag<PhysicalData, PhysicalOp> {
 public:
  using Dag<PhysicalData, PhysicalOp>::ToDotString;
  using Dag<PhysicalData, PhysicalOp>::ToString;
  std::string ToDotString() const;
  std::string ToString() const;

 private:
  static std::string DataToString(const PhysicalData&);
  static std::string OpToString(const PhysicalOp&);
};

typedef PhysicalDag::DNode PhysicalDataNode;
typedef PhysicalDag::ONode PhysicalOpNode;

}  // namespace minerva
