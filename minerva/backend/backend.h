#pragma once
#include <vector>
#include <memory>
#include "op/compute_fn.h"
#include "backend/backend_chunk.h"
#include "common/scale.h"
#include "common/common.h"

namespace minerva {

class Backend {
 public:
  Backend() = default;
  DISALLOW_COPY_AND_MOVE(Backend);
  virtual ~Backend() = default;
  virtual std::vector<BackendChunk*> Create(std::vector<BackendChunk*> const&,
      std::vector<Scale> const&, std::shared_ptr<ComputeFn>) = 0;
  virtual BackendChunk* CreateOne(BackendChunk*, Scale const&,
      std::shared_ptr<ComputeFn>);
  virtual void Wait(BackendChunk*) = 0;
  virtual void WaitForAll() = 0;
  virtual std::shared_ptr<float> GetValue(BackendChunk*) = 0;
};

}  // namespace minerva

