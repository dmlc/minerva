#include <string>
#include <sstream>
#include "op/physical.h"
#include "op/compute_fn.h"

namespace minerva {

class ExternRCPrinter {
 public:
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.extern_rc;
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return o.compute_fn->Name();
  }
};

class DataIdPrinter {
 public:
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << d.data_id << "!" << d.size;
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return o.compute_fn->Name();
  }
};

class AllInfoPrinter {
 public:
  static std::string DataToString(const PhysicalData& d) {
    std::stringstream ss;
    ss << "size===" << d.size << ";;;device_id===" << d.device_id
      << ";;;data_id===" << d.data_id << ";;;extern_rc===" << d.extern_rc << ";;;";
    return ss.str();
  }
  static std::string OpToString(const PhysicalOp& o) {
    return "name===" + o.compute_fn->Name() + ";;;";
  }
};

}  // namespace minerva

