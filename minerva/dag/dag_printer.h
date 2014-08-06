#include "op/logical.h"
#include "op/physical.h"

namespace minerva {

class ExternRCPrinter {
 public:
  static std::string DataToString(const LogicalData& d) {
    std::stringstream ss;
    ss << d.extern_rc;
    return ss.str();
  }
  static std::string OpToString(const LogicalOp& o) {
    return o.compute_fn->Name();
  }
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.extern_rc;
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return o.compute_fn->Name();
  }
};

}
