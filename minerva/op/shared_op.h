#pragma once
#include "op/shared_fn.h"
#include "logical.h"
#include "physical.h"
#include "closure.h"
#include "impl/bundle.h"
#include "device/device_info.h"
#include <sstream>
#include <vector>
#include <glog/logging.h>

namespace minerva {

///////////////////////////////////////////////////
// Data generate functions
///////////////////////////////////////////////////
class RandnOp : public SharedDataGenFnWithClosure<RandnClosure> {
 public:
  NVector<Chunk> Expand(const NVector<Scale>& part_sizes) {
    return part_sizes.Map<Chunk>(
        [&] (const Scale& psize) {
          return Chunk::Randn(psize, closure.mu, closure.var);
        });
  }
  std::string Name() const {
    return ":randn";
  }
};

class FillOp : public SharedDataGenFnWithClosure<FillClosure> {
 public:
  NVector<Chunk> Expand(const NVector<Scale>& part_sizes) {
    return part_sizes.Map<Chunk>(
        [&] (const Scale& psize) {
          return Chunk::Constant(psize, closure.val);
        });
  }
  std::string Name() const {
    std::stringstream ss;
    ss << ":const=" << closure.val;
    return ss.str();
  }
};

///////////////////////////////////////////////////
// Compute functions
///////////////////////////////////////////////////

class MatMultOp : public SharedComputeFnWithClosure<MatMultClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    CHECK_EQ(inputs.size(), 2) << "MatMultOp takes 2 inputs";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no partition allowed";
    CHECK_EQ(inputs[1].Size().Prod(), 1) << "no partition allowed";
    auto& a = inputs[0].ToVector()[0];
    auto& b = inputs[1].ToVector()[0];
    CHECK_EQ(a.Size().NumDims(), 2) << "MatMultOp only performs on 2D data";
    CHECK_EQ(b.Size().NumDims(), 2) << "MatMultOp only performs on 2D data";
    NVector<Chunk> c({1, 1});
    c[{0, 0}] = Chunk::Compute({a, b}, {Scale{a.Size(0), b.Size(1)}}, new MatMultOp)[0];
    return {c};
  }
  std::string Name() const {
    return "*";
  }
};

class TransOp : public SharedComputeFnWithClosure<TransposeClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    CHECK_EQ(inputs.size(), 1) << "TransOp takes 1 input";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no partition allowed";
    auto& a = inputs[0].ToVector()[0];
    CHECK_EQ(a.Size().NumDims(), 2) << "TransOp only performs on 2D data";
    NVector<Chunk> res({1, 1});
    res[{0, 0}] = Chunk::Compute({a}, {Scale{a.Size(1), a.Size(0)}}, new TransOp)[0];
    return {res};
  }
  std::string Name() const {
    return "trans";
  }
};

class ReductionOp : public SharedComputeFnWithClosure<ReductionClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    CHECK_EQ(inputs.size(), 1) << "ReductionOp takes 1 input";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no partition allowed";
    auto& a = inputs[0].ToVector()[0];
    ReductionOp* op = new ReductionOp;
    op->closure = closure;
    auto rstsize = a.Size();
    for (auto i: closure.dims_to_reduce) {
      rstsize[i] = 1;
    }
    NVector<Chunk> res(Scale::Constant(a.Size().NumDims(), 1));
    res[Scale::Origin(a.Size().NumDims())] = Chunk::Compute({a}, {rstsize}, op)[0];
    return {res};
  }
  std::string Name() const {
   switch (closure.type) {
     case SUM:
       return "sum";
     case MAX:
       return "max";
   }
   return "reduction N/A";
  }
};

class MaxIndexOp : public SharedComputeFnWithClosure<MaxIndexClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    CHECK_EQ(inputs.size(), 1) << "MaxIndexOp takes 1 input";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no partition allowed";
    auto& a = inputs[0].ToVector()[0];
    MaxIndexOp* op = new MaxIndexOp;
    op->closure = closure;
    auto size = a.Size();
    size[closure.dim] = 1;
    NVector<Chunk> res(Scale::Constant(a.Size().NumDims(), 1));
    res[Scale::Origin(a.Size().NumDims())] = Chunk::Compute({a}, {size}, op)[0];
    return {res};
  }
  std::string Name() const {
    return "max index";
  }
};

class ElewiseOp : public SharedComputeFnWithClosure<ElewiseClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    NVector<Chunk> ret = inputs[0].Map<Chunk>(
        [&] (const Chunk& ch) {
          ElewiseOp* elewise_op = new ElewiseOp;
          elewise_op->closure = closure;
          return Chunk::Compute({ch}, {ch.Size()}, elewise_op)[0];
        }
      );
    return {ret};
  }
  std::string Name() const {
    switch(closure.type) {
      case EXP:      return "exp";
      case LN:       return "ln";
      case SIGMOID:  return "sigmoid";
      case NEGATIVE: return "-";
    };
    return "NA";
  }
};

class ArithmeticOp : public SharedComputeFnWithClosure<ArithmeticClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    const NVector<Chunk>& a = inputs[0];
    const NVector<Chunk>& b = inputs[1];
    const NVector<Chunk>& ret = NVector<Chunk>::ZipMap(a, b,
        [&] (const Chunk& c1, const Chunk& c2) {
          assert(c1.Size() == c2.Size());
          ArithmeticOp* arith_op = new ArithmeticOp;
          arith_op->closure = closure;
          return Chunk::Compute({c1, c2}, {c1.Size()}, arith_op)[0];
        }
      );
    return {ret};
  }
  std::string Name() const {
    switch(closure.type) {
      case ADD:   return "+";
      case SUB:   return "-";
      case MULT:  return ".*";
      case DIV:   return "./";
    };
    return "NA";
  }
};

class ArithmeticConstOp : public SharedComputeFnWithClosure<ArithmeticConstClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    const NVector<Chunk>& a = inputs[0];
    const NVector<Chunk>& ret = a.Map<Chunk>(
        [&] (const Chunk& c) {
          ArithmeticConstOp* aconst_op = new ArithmeticConstOp;
          aconst_op->closure = closure;
          return Chunk::Compute({c}, {c.Size()}, aconst_op)[0];
        }
      );
    return {ret};
  }
  std::string Name() const {
    std::stringstream ss;
    if(closure.side == 0) { // left
      ss << closure.val;
    }
    switch(closure.type) {
      case ADD:   ss << "+."; break;
      case SUB:   ss << "-."; break;
      case MULT:  ss << ".*"; break;
      case DIV:   ss << "./"; break;
    };
    if(closure.side == 1) { // right
      ss << closure.val;
    }
    return ss.str();
  }
};

class NormArithmeticOp: public SharedComputeFnWithClosure<NormArithmeticClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    const NVector<Chunk>& lhs = inputs[0];
    const NVector<Chunk>& rhs = inputs[1];
    NVector<Chunk> res(lhs.Size());
    // TODO How to verify that the parition is the same on dimensions that don't need to be replicated?
    // Let's put this work into kernel for now
    auto iterator_max = lhs.Size();
    auto iterator = Scale::Origin(iterator_max.NumDims());
    do {
      auto iterator_rhs = iterator;
      for (auto i: closure.dims_to_replicate) {
        iterator_rhs[i] = 0;
      }
      NormArithmeticOp* op = new NormArithmeticOp;
      op->closure = closure;
      res[iterator] = Chunk::Compute({lhs[iterator], rhs[iterator_rhs]}, {lhs[iterator].Size()}, op)[0];
    } while (iterator.IncrOne(iterator_max));
    return {res};
  }
  std::string Name() const {
    std::stringstream ss;
    switch (closure.type) {
      case ADD:
        ss << "+";
        break;
      case SUB:
        ss << "-";
        break;
      case MULT:
        ss << ".*";
        break;
      case DIV:
        ss << "./";
        break;
    }
    ss << " norm";
    return ss.str();
  }
};

}

