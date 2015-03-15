#include "physical_dag.h"
#include <string>
#include <sstream>
using namespace std;

namespace minerva {

string PhysicalDag::ToDotString() const {
  return ToDotString(DataToString, OpToString);
}

string PhysicalDag::ToString() const {
  return ToString(DataToString, OpToString);
}

string PhysicalDag::DataToString(const PhysicalData& d) {
  stringstream ss;
  ss << d.size;
  return ss.str();
}

string PhysicalDag::OpToString(const PhysicalOp& o) {
  return o.compute_fn->Name();
}

}  // namespace minerva
