#pragma once

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
};

class RandnLogic : public LogicalOpWithClosure<RandnClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
};

class TransLogic : public LogicalOp {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
};

class ElewiseLogic : public LogicalOpWithClosure<ElewiseClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
};

class ArithmicLogic : public LogicalOpWithClosure<ArithmicClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
};

class ArithmicConstLogic : public LogicalOpWithClosure<ArithmicConstClosure> {
  std::vector<NVector<Chunk>> Expand(std::vector<NVector<Chunk>> inputs) {
    //TODO
    return std::vector<NVector<Chunk>>();
  }
};

} // end of namespace minerva
