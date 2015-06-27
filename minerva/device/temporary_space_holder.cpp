#include "./temporary_space_holder.h"

namespace minerva {

TemporarySpaceHolder::TemporarySpaceHolder(void* ptr, size_t size,
    Deallocator deallocator)
  : ptr_{ptr}
  , size_{size}
  , deallocator_{deallocator} {
}

TemporarySpaceHolder::~TemporarySpaceHolder() {
  deallocator_();
}

float* TemporarySpaceHolder::ptr() {
  return static_cast<float*>(ptr_);
}

size_t TemporarySpaceHolder::size() const {
  return size_;
}

}  // namespace minerva

