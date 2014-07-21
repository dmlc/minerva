#pragma once

#include <sstream>
#include "shared.h"
#include "closure.h"

namespace minerva {

///////////////////////////////////////////////////
// Data generate functions
///////////////////////////////////////////////////
class RandnOp : public SharedDataGenFn, public ClosureTrait<RandnClosure> {
 public:
  NVector<Chunk> Expand(const Scale& size) {
    NVector<Scale> partsizes = size.EquallySplit(closure.numparts);
    NVector<Chunk> rst_chunks = partsizes.Map<Chunk>(
      [&] (const Scale& size)->Chunk {
        return Chunk::Randn(size, closure.mu, closure.var);
      });
    return rst_chunks;
  }
  std::string Name() const {
    return ":randn";
  }
};

class FillOp : public SharedDataGenFn, public ClosureTrait<FillClosure> {
 public:
  NVector<Chunk> Expand(const Scale& size) {
    NVector<Scale> partsizes = size.EquallySplit(closure.numparts);
    NVector<Chunk> rst_chunks = partsizes.Map<Chunk>(
      [&] (const Scale& size)->Chunk {
        return Chunk::Constant(size, closure.val);
      });
    return rst_chunks;
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

class MatMultOp : public SharedComputeFn {
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
        int row = a[{i, 0}].Size(0);
        int col = b[{0, j}].Size(1);
        c[{i, j}] = Chunk::Constant({row, col}, 0.0);
        for(int l = 0; l < k; ++l) {
          c[{i, j}] += a[{i, l}] * b[{l, j}];
        }
      }
    }
    return {c};
  }
  std::string Name() const {
    return "*";
  }
};

class TransOp : public SharedComputeFn {
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

class ElewiseOp : public SharedComputeFn,
  public ClosureTrait<ElewiseClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    assert(false);
    return std::vector<NVector<Chunk>>();
  }
  std::string Name() const {
    switch(closure.type) {
      case EXP:      return "exp";
      case LN:       return "ln";
      case SIGMOID:  return "sigmoid";
      case NEGATIVE: return "-";
    };
  }
};

class ArithmeticOp : public SharedComputeFn,
  public ClosureTrait<ArithmeticClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    NVector<Chunk> a = inputs[0], b = inputs[1];
    NVector<Chunk> ret = NVector<Chunk>::ZipMap(a, b,
        [&] (const Chunk& c1, const Chunk& c2) {
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
  }
};

class ArithmeticConstOp : public SharedComputeFn,
  public ClosureTrait<ArithmeticConstClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    NVector<Chunk>& a = inputs[0];
    NVector<Chunk> ret = a.Map<Chunk>(
        [&] (const Chunk& c) {
          ArithmeticConstOp* arith_const_op = new ArithmeticConstOp;
          arith_const_op->closure = closure;
          return Chunk::Compute({c}, {c.Size()}, arith_const_op)[0];
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
