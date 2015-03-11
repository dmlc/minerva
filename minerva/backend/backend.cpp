#include "backend.h"
#include <memory>
#include "backend/backend_chunk.h"
#include "op/physical_fn.h"

using namespace std;

namespace minerva {

BackendChunk* Backend::CreateOne(BackendChunk* param, const Scale& result_size, shared_ptr<ComputeFn> fn) {
  return Create({param}, {result_size}, fn)[0];
}

}  // namespace minerva

