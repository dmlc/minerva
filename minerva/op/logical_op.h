#pragma once

#include <sstream>
#include "dag/logical.h"
#include "closure.h"

namespace minerva {

class MatMultLogic : public LogicalOp {
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
    std::vector<NVector<Chunk>> rst;
    rst.push_back(c);
    return rst;
  }
  std::string Name() const {
    return "*";
  }
};

class RandnLogic : public LogicalOpWithClosure<RandnClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
  std::string Name() const {
    return "randn";
  }
};

class TransLogic : public LogicalOp {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
  std::string Name() const {
    return "trans";
  }
};

class ElewiseLogic : public LogicalOpWithClosure<ElewiseClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
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

class ArithmicLogic : public LogicalOpWithClosure<ArithmicClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
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

class ArithmicConstLogic : public LogicalOpWithClosure<ArithmicConstClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
  std::string Name() const {
    std::stringstream ss;
    if(closure.side == 0) { // left
      ss << closure.val;
    }
    switch(closure.type) {
      case ADD:   ss << "+";
      case SUB:   ss << "-";
      case MULT:  ss << ".*";
      case DIV:   ss << "./";
    };
    if(closure.side == 1) { // right
      ss << closure.val;
    }
    return ss.str();
  }
};

} // end of namespace minerva
