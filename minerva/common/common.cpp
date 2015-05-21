#include "./common.h"
#include <cstdarg>
#include <dmlc/logging.h>

namespace minerva {
namespace common {

void FatalError(char const* format, ...) {
  char buffer[4096];
  va_list va;
  va_start(va, format);
  vsprintf(buffer, format, va);
  va_end(va);
  LOG(FATAL) << buffer;
  abort();
}

}  // namespace common
}  // namespace minerva

