#include "backend.h"
#include "backend/backend_chunk.h"
#include "op/physical_fn.h"

BackendChunk* CreateOne(BackendChunk* param, const Scale& result_size, ComputeFn* fn) {
  return Create({param}, {result_size}, fn)[0];
}

