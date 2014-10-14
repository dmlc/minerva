#pragma once
#include "common/scale.h"
#include "narray/conv_closure.h"

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

struct RandnClosure {
  float mu, var;
};

struct FillClosure {
  float val;
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

struct ElewiseClosure {
  ElewiseType type;
};

struct ArithmeticClosure {
  ArithmeticType type;
};

struct ArithmeticConstClosure {
  ArithmeticType type;
  float val;
  int side; // 0 is left const, 1 is right const
};

struct NormArithmeticClosure {
  ArithmeticType type;
  Scale dims_to_replicate;
};

typedef ConvInfo ConvClosure;

typedef PoolingInfo PoolingClosure;

struct SoftmaxClosure {
  SoftmaxAlgorithm algorithm;
};

struct ActivationClosure {
  ActivationAlgorithm algorithm;
};

}  // end of namespace minerva

