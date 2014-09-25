#pragma once

#include "common/scale.h"

namespace minerva {

enum class ArithmeticType {
  kAdd = 0,
  kSub,
  kMult,
  kDiv,
};

enum class ElewiseType {
  kExp = 0,
  kLn,
  kSigmoid,
  kNegative,
};

enum class ReductionType {
  kSum = 0,
  kMax,
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

}  // end of namespace minerva

