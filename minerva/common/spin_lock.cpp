#include "./spin_lock.h"

using namespace std;

namespace minerva {
namespace common {

void SpinLock::Lock() {
  while (m_.test_and_set(memory_order_acquire));
}

void SpinLock::Unlock() {
  m_.clear(memory_order_release);
}

}  // namespace common
}  // namespace minerva

