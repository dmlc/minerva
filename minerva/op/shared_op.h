#pragma once
#include "op/shared_fn.h"
#include "logical.h"
#include "physical.h"
#include "closure.h"
#include "impl/bundle.h"
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
    CHECK_EQ(part_sizes.Size().Prod(), 1) << "no partition allowed";
    RandnOp* op = new RandnOp(*this);
    auto size = part_sizes.ToVector()[0];
    NVector<Chunk> res(Scale::Constant(size.NumDims(), 1));
    res[Scale::Origin(size.NumDims())] = Chunk::Compute({}, {size}, op)[0];
    return {res};
  }
  std::string Name() const {
    return ":randn";
  }
};

class FillOp : public SharedDataGenFnWithClosure<FillClosure> {
 public:
  NVector<Chunk> Expand(const NVector<Scale>& part_sizes) {
    CHECK_EQ(part_sizes.Size().Prod(), 1) << "no partition allowed";
    FillOp* op = new FillOp(*this);
    auto size = part_sizes.ToVector()[0];
    NVector<Chunk> res(Scale::Constant(size.NumDims(), 1));
    res[Scale::Origin(size.NumDims())] = Chunk::Compute({}, {size}, op)[0];
    return {res};
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
    CHECK_EQ(a.Size(1), b.Size(0)) << "size mismatch";
    NVector<Chunk> c({1, 1});
    MatMultOp* op = new MatMultOp();
    op->device_id = device_id;
    c[{0, 0}] = Chunk::Compute({a, b}, {Scale{a.Size(0), b.Size(1)}}, op)[0];
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
    TransOp* op = new TransOp();
    op->device_id = device_id;
    res[{0, 0}] = Chunk::Compute({a}, {Scale{a.Size(1), a.Size(0)}}, op)[0];
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
    op->device_id = device_id;
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
    op->device_id = device_id;
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
    CHECK_EQ(inputs.size(), 1) << "ElewiseOp takes 1 input";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no parition allowed";
    auto& a = inputs[0].ToVector()[0];
    ElewiseOp* elewise_op = new ElewiseOp;
    elewise_op->closure = closure;
    elewise_op->device_id = device_id;
    NVector<Chunk> res(Scale::Constant(a.Size().NumDims(), 1));
    res[Scale::Origin(a.Size().NumDims())] = Chunk::Compute({a}, {a.Size()}, elewise_op)[0];
    return {res};
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
    CHECK_EQ(inputs.size(), 2) << "ArithmeticOp takes 2 inputs";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no partition allowed";
    CHECK_EQ(inputs[1].Size().Prod(), 1) << "no partition allowed";
    auto& a = inputs[0].ToVector()[0];
    auto& b = inputs[1].ToVector()[0];
    CHECK_EQ(a.Size(), b.Size()) << "size mismatch";
    ArithmeticOp* arith_op = new ArithmeticOp;
    arith_op->closure = closure;
    arith_op->device_id = device_id;
    NVector<Chunk> res(Scale::Constant(a.Size().NumDims(), 1));
    res[Scale::Origin(a.Size().NumDims())] = Chunk::Compute({a, b}, {a.Size()}, arith_op)[0];
    return {res};
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
    CHECK_EQ(inputs.size(), 1) << "ArithmeticConstOp takes 1 input";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no parition allowed";
    auto& a = inputs[0].ToVector()[0];
    ArithmeticConstOp* aconst_op = new ArithmeticConstOp;
    aconst_op->closure = closure;
    aconst_op->device_id = device_id;
    NVector<Chunk> res(Scale::Constant(a.Size().NumDims(), 1));
    res[Scale::Origin(a.Size().NumDims())] = Chunk::Compute({a}, {a.Size()}, aconst_op)[0];
    return {res};
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
    CHECK_EQ(inputs.size(), 2) << "NormArithmeticOp takes 2 inputs";
    CHECK_EQ(inputs[0].Size().Prod(), 1) << "no parition allowed";
    CHECK_EQ(inputs[1].Size().Prod(), 1) << "no parition allowed";
    auto& a = inputs[0].ToVector()[0]; // Normalizee
    auto& b = inputs[1].ToVector()[0]; // Normalizer
    for (size_t i = 0; i < a.Size().NumDims(); ++i) {
      if (closure.dims_to_replicate.Contains(i)) {
        CHECK_EQ(b.Size(i), 1) << "size mismatch";
      } else {
        CHECK_EQ(a.Size(i), b.Size(i)) << "size mismatch";
      }
    }
    NormArithmeticOp* op = new NormArithmeticOp;
    op->closure = closure;
    op->device_id = device_id;
    NVector<Chunk> res(Scale::Constant(a.Size().NumDims(), 1));
    res[Scale::Origin(a.Size().NumDims())] = Chunk::Compute({a, b}, {a.Size()}, op)[0];
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

