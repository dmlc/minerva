#pragma once

#include "common/scale.h"

namespace minerva {

/*enum ClosureType {
  ARITHMETIC = 0,
  ARITHMETIC_CONST,
  ELEWISE,
  MAT_MULT,
  REDUCTION,
  CONV,
  // TODO how to generate this when adding new closure types ?
};*/

enum ArithmeticType {
  ADD = 0,
  SUB,
  MULT,
  DIV,
};

enum ElewiseType {
  EXP = 0,
  LN,
  SIGMOID,
  NEGATIVE,
};

enum ReductionType {
  SUM = 0,
  MAX,
};

struct AssembleClosure {
};

struct ArithmeticClosure {
  ArithmeticType type;
};

struct ArithmeticConstClosure {
  ArithmeticType type;
  float val;
  int side; // 0 is left const, 1 is right const
};

struct ElewiseClosure {
  ElewiseType type;
};

struct MatMultClosure {
};

struct TransposeClosure {
};

struct ReductionClosure {
  ReductionType type;
  Scale dims_to_reduce;
};

struct MaxIndexClosure {
  int dim;
};

struct NormArithmeticClosure {
  ArithmeticType type;
  Scale dims_to_replicate;
};

struct ConvInfo {
  int numfilters;
  Scale filtersize, stride, paddingsize;
};

struct RandnClosure {
  float mu, var;
};

struct FillClosure {
  float val;
};

struct SplitClosure {
};

} // end of namespace minerva
