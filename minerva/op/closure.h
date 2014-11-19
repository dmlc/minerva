#pragma once
#include <memory>
#include "common/scale.h"
#include "narray/convolution_info.h"

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

struct ArrayLoaderClosure {
  std::shared_ptr<float> data;
};

struct RandnClosure {
  float mu, var;
};

struct RandBernoulliClosure {
  float p;
};

struct FillClosure {
  float val;
};

struct MatMultClosure {
};

struct TransposeClosure {
};

struct ReshapeClosure {
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

template<int i> struct ConvClosure {
  int pad_height;
  int pad_width;
  int stride_vertical;
  int stride_horizontal;
};

typedef ConvClosure<0> ConvForwardClosure;

typedef ConvClosure<1> ConvBackwardDataClosure;

typedef ConvClosure<2> ConvBackwardFilterClosure;

struct ConvBackwardBiasClosure {
};

template<int i> struct SoftmaxClosure {
  SoftmaxAlgorithm algorithm;
};

typedef SoftmaxClosure<0> SoftmaxForwardClosure;

typedef SoftmaxClosure<1> SoftmaxBackwardClosure;

template<int i> struct ActivationClosure {
  ActivationAlgorithm algorithm;
};

typedef ActivationClosure<0> ActivationForwardClosure;

typedef ActivationClosure<1> ActivationBackwardClosure;

template<int i> struct PoolingClosure {
  PoolingInfo::Algorithm algorithm;
  int height;
  int width;
  int stride_vertical;
  int stride_horizontal;
};

typedef PoolingClosure<0> PoolingForwardClosure;

typedef PoolingClosure<1> PoolingBackwardClosure;

}  // end of namespace minerva

