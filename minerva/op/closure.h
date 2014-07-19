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
  // TODO how to generate this when adding new closure types ?
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
  NEGATIVE,
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

struct ConvInfo {
  int numfilters;
  Scale filtersize, stride, paddingsize;
};

struct RandnClosure {
  float mu, var;
  Scale numparts;
};

struct FillClosure {
  float val;
  Scale numparts;
};

} // end of namespace minerva

