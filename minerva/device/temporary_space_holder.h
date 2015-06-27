#pragma once
#include <memory>
#include "common/common.h"

namespace minerva {

class TemporarySpaceHolder {
 public:
  using Deallocator = std::function<void()>;
  TemporarySpaceHolder(void*, size_t, Deallocator);
  DISALLOW_COPY_AND_MOVE(TemporarySpaceHolder);
  virtual ~TemporarySpaceHolder();
  float* ptr();
  size_t size() const;

 private:
  void* ptr_;
  size_t size_;
  Deallocator deallocator_;
};

}  // namespace minerva

