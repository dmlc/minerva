#pragma once
#include <string>

namespace minerva {

template<typename Data, typename Op>
class DagHelper {
 public:
  static std::string DataToString(const Data&) {
    return "N/A";
  }
  static std::string OpToString(const Op&) {
    return "N/A";
  }
  static void FreeData(Data&) {}
  static void FreeOp(Op&) {}
};

}  // namespace minerva

