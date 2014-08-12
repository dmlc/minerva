#pragma once
#include <sstream>
#include <vector>
#include <glog/logging.h>
#include "op/shared_fn.h"
#include "logical.h"
#include "physical.h"
#include "closure.h"
#include "impl/bundle.h"

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
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
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

class ReductionOp : public SharedComputeFnWithClosure<ReductionClosure> {
 public:
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    LOG(INFO) << "ReductionOp::Expand";
    CHECK_EQ(inputs.size(), 1) << "Reduction #input wrong";
    NVector<Chunk> individual_reduce = inputs[0].Map<Chunk>(
      [&] (Chunk ch) {
        return ch.Reduce(closure.dims_to_reduce, closure.type);
      }
    );
    Chunk merged = Chunk::Merge(individual_reduce);
    Scale merged_size = merged.Size();
    Scale num_rst_parts = Scale::Constant(merged_size.NumDims(), 1);
    NVector<Chunk> rst(num_rst_parts);
    rst[num_rst_parts - 1] = merged.Reduce(closure.dims_to_reduce, closure.type);
    return {rst};
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
    LOG(INFO) << "MaxIndexOp::Expand";
    CHECK_EQ(inputs.size(), 1) << "MaxIndex #input wrong";
    Chunk merged;
    if (inputs[0].Length() != 1) {
      // Merge first
      merged = Chunk::Merge(inputs[0]);
    } else {
      merged = inputs[0][0];
    }
    MaxIndexOp* op = new MaxIndexOp;
    op->closure = closure;
    auto size = merged.Size();
    for (auto i: dims) {
      size[i] = 1;
    }

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
  std::vector<NVector<Chunk>> Expand(const std::vector<NVector<Chunk>>& inputs) {
    const NVector<Chunk>& a = inputs[0];
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

