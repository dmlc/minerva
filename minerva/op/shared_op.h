#pragma once
#include <sstream>
#include <vector>

#include "logical.h"
#include "physical.h"
#include "closure.h"
#include "impl/bundle.h"

namespace minerva {

template<class C>
class SharedDataGenFnWithClosure :
  public LogicalDataGenFn, public PhysicalDataGenFn, public ClosureTrait<C> {
 public:
  void Execute(DataShard output, IMPL_TYPE impl_type) {
    FnBundle<C>::Call(output, ClosureTrait<C>::closure, impl_type);
  }
};

template<class C>
class SharedComputeFnWithClosure :
  public LogicalComputeFn, public PhysicalComputeFn, public ClosureTrait<C> {
 public:
  void Execute(DataList& inputs, DataList& outputs, IMPL_TYPE impl_type) {
    FnBundle<C>::Call(inputs, outputs, ClosureTrait<C>::closure, impl_type);
  }
};

///////////////////////////////////////////////////
// Data generate functions
///////////////////////////////////////////////////
class RandnOp : public SharedDataGenFnWithClosure<RandnClosure> {
 public:
  Chunk Expand(const Scale& size) {
    return Chunk::Randn(size, closure.mu, closure.var);
  }
  std::string Name() const {
    return ":randn";
  }
};

class FillOp : public SharedDataGenFnWithClosure<FillClosure> {
 public:
  Chunk Expand(const Scale& size) {
    return Chunk::Constant(size, closure.val);
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
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    NVector<Chunk> a = inputs[0];
    NVector<Chunk> b = inputs[1];
    // validity
    assert(a.Size(1) == b.Size(0));
    int m = a.Size(0);
    int n = b.Size(1);
    int k = a.Size(1);
    // matmult
    NVector<Chunk> c({m, n});
    for(int i = 0 ; i < m; ++i) {
      for(int j = 0; j < n; ++j) {
        for(int l = 0; l < k; ++l) {
          if(l == 0) {
            c[{i, j}] = a[{i, l}] * b[{l, j}];
          }
          else {
            c[{i, j}] += a[{i, l}] * b[{l, j}];
          }
        }
      }
    }
    return {c};
  }
  std::string Name() const {
    return "*";
  }
};

class TransOp : public SharedComputeFnWithClosure<TransposeClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    NVector<Chunk> in = inputs[0];
    assert(in.Size().NumDims() == 2);
    int row = in.Size(0), col = in.Size(1);
    NVector<Chunk> rst({col, row});
    for(int i = 0; i < row; ++i)
      for(int j = 0; j < col; ++j)
        rst[{j, i}] = in[{i, j}].Trans();
    return {rst};
  }
  std::string Name() const {
    return "trans";
  }
};

class ElewiseOp : public SharedComputeFnWithClosure<ElewiseClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
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
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    NVector<Chunk> a = inputs[0], b = inputs[1];
    NVector<Chunk> ret = NVector<Chunk>::ZipMap(a, b,
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
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    NVector<Chunk>& a = inputs[0];
    NVector<Chunk> ret = a.Map<Chunk>(
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
      case ADD:   ss << "+"; break;
      case SUB:   ss << "-"; break;
      case MULT:  ss << ".*"; break;
      case DIV:   ss << "./"; break;
    };
    if(closure.side == 1) { // right
      ss << closure.val;
    }
    return ss.str();
  }
};

} // end of namespace minerva
