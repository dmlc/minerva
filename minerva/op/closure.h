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

class Closure {
 public:
  virtual bool Equal(Closure* other) = 0;
  // TODO abstract serial/deserial interface
};

struct ArithmicClosure : public Closure {
  ArithmicType type;
  bool Equal(Closure* other) {
    ArithmicClosure* ac = dynamic_cast<ArithmicClosure*>(other);
    return ac == NULL ? false : type == ac->type;
  }
};

struct ArithmicConstClosure : public Closure {
  ArithmicType type;
  float val;
  int side; // 0 is left, 1 is right
  bool Equal(Closure* other) {
    ArithmicConstClosure* acc = dynamic_cast<ArithmicConstClosure*>(other);
    return acc == NULL ? false : 
      type == acc->type && val == acc->val && side == acc->side;
  }
};

struct ElewiseClosure {
  ElewiseType type;
  bool Equal(Closure* other) {
    ElewiseClosure* ec = dynamic_cast<ElewiseClosure*>(other);
    return ec == NULL ? false : type == ec->type;
  }
};

struct MatMultClosure {
  bool Equal(Closure* other) {
    MatMultClosure* mc = dynamic_cast<MatMultClosure*>(other);
    return mc != NULL;
  }
};

struct ReductionClosure {
  ReductionType type;
  Scale dims_to_reduce;
  bool Equal(Closure* other) {
    ReductionClosure* rc = dynamic_cast<ReductionClosure*>(other);
    return rc == NULL ? false :
      type == rc->type && dims_to_reduce == rc->dims_to_reduce;
  }
};

struct ConvInfo {
  int numfilters;
  Scale filtersize, stride, paddingsize;
  bool Equal(Closure* other) {
    ConvInfo* ci = dynamic_cast<ConvInfo*>(other);
    return ci == NULL ? false :
      numfilters == ci->numfilters && filtersize == ci->filtersize &&
      stride == ci->stride && paddingsize == ci->paddingsize;
  }
};

} // end of namespace minerva
