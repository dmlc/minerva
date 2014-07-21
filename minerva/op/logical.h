#pragma once

#include <vector>

#include "common/scale.h"
#include "common/nvector.h"
#include "chunk/chunk.h"
#include "op.h"
#include "context.h"

namespace minerva {

struct LogicalData;
struct LogicalOp;
class LogicalDataFn;
class LogicalComputeFn;

class LogicalDataGenFn : public BasicFn {
 public:
   virtual NVector<Chunk> Expand(const Scale& rst_size) = 0;
   virtual ~LogicalDataGenFn() {}
};

class LogicalComputeFn : public BasicFn {
 public:
  virtual std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) = 0;
};

struct LogicalData {
  Scale size;
  LogicalDataGenFn* data_gen_fn;
  //DataNodeContext context; // TODO how to set context ?
};

struct LogicalOp {
  LogicalComputeFn* compute_fn;
};

/*
template<class T>
class LogicalComputeFnWithClosure : public LogicalComputeFn {
 public:
  T closure; // Q: why we isolate the closure here ? A: simply because i'm too lazy
             // to write constructors for every operators, and it's quite elegant to
             // use brace-initializer.
  //OpNodeContext context; // TODO how to set context ?
};*/

}// end of namespace minerva
