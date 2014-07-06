#pragma once

#include "common/scale.h"

namespace minerva {

enum ClosureType {
  ARITHMIC = 0,
  ARITHMIC_CONST,
  ELEWISE,
  MAT_MULT,
  REDUCTION,
  CONV,
};

enum ArithmicType {
  ADD = 0,
  SUB,
  MULT,
  DIV,
};

enum ElewiseType {
  EXP = 0,
  LN,
  SIGMOID,
};

enum ReductionType {
  SUM = 0,
  MAX,
};

struct ArithmicClosure {
  ArithmicType type;
};

struct ArithmicConstClosure {
  ArithmicType type;
  float val;
  int side; // 0 is left, 1 is right
};

struct ElewiseClosure {
  ElewiseType type;
};

struct MatMultClosure {
};

struct ReductionClosure {
  ReductionType type;
  Scale dims_to_reduce;
};

struct ConvolutionClosure {
};

} // end of namespace minerva
