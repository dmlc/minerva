#pragma once
#include <vector>
#include <glog/logging.h>

namespace minerva {

template<typename DagType>
class DagProcedure {
 public:
  virtual void Process(const std::vector<uint64_t>&) = 0;
 protected:
  DagType* dag_;
};

}  // namespace minerva

