#pragma once
#include <vector>
#include "common/nvector.h"
#include "chunk/chunk.h"
#include "op/basic_fn.h"

namespace minerva {

class LogicalDataGenFn: public BasicFn {
 public:
  virtual NVector<Chunk> Expand(const NVector<Scale>& partition_shapes) = 0;
};

class LogicalComputeFn: public BasicFn {
 public:
  virtual std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>>) = 0;
};

template<int num_outputs, int num_inputs>
class LogicalComputeFnTemp: public LogicalComputeFn {
};

template<>
class LogicalComputeFnTemp<1, 1>: public LogicalComputeFn {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    return {ExpandReal(inputs[0])};
  }
  virtual NVector<Chunk> ExpandReal(NVector<Chunk>) = 0;
};

template<>
class LogicalComputeFnTemp<1, 2>: public LogicalComputeFn {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    return {ExpandReal(inputs[0], inputs[1])};
  }
  virtual NVector<Chunk> ExpandReal(NVector<Chunk>, NVector<Chunk>) = 0;
};

}
