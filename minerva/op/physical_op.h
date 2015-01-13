#pragma once
#include "op/physical.h"
#include "op/physical_fn.h"
#include "op/closure.h"
#include "op/impl/bundle.h"
#include <sstream>
#include <vector>
#include <glog/logging.h>

namespace minerva {

// Data generate functions

class ArrayLoaderOp : public PhyDataGenFnWithClosure<ArrayLoaderClosure> {
 public:
  std::string Name() const {
    return ":array loader";
  }
};

class RandnOp : public PhyDataGenFnWithClosure<RandnClosure> {
 public:
  std::string Name() const {
    return ":normal";
  }
};

class RandBernoulliOp : public PhyDataGenFnWithClosure<RandBernoulliClosure> {
 public:
  std::string Name() const {
    return ":bernoulli";
  }
};

class FillOp : public PhyDataGenFnWithClosure<FillClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << ":const=" << closure.val;
    return ss.str();
  }
};

// Compute functions
class MatMultOp : public PhyComputeFnWithClosure<MatMultClosure> {
 public:
  std::string Name() const {
    return "*";
  }
};

class TransOp : public PhyComputeFnWithClosure<TransposeClosure> {
 public:
  std::string Name() const {
    return "trans";
  }
};

class ReductionOp : public PhyComputeFnWithClosure<ReductionClosure> {
 public:
  std::string Name() const {
   switch (closure.type) {
     case ReductionType::kSum:
       return "sum";
     case ReductionType::kMax:
       return "max";
   }
   return "reduction N/A";
  }
};

class MaxIndexOp : public PhyComputeFnWithClosure<MaxIndexClosure> {
 public:
  std::string Name() const {
    return "max index";
  }
};

class ReshapeOp : public PhyComputeFnWithClosure<ReshapeClosure> {
 public:
  std::string Name() const {
    return "reshape";
  }
};

class ElewiseOp : public PhyComputeFnWithClosure<ElewiseClosure> {
 public:
  std::string Name() const {
    switch(closure.type) {
      case ElewiseType::kExp:      return "exp";
      case ElewiseType::kLn:       return "ln";
      case ElewiseType::kNegative: return "-";
    };
    return "NA";
  }
};

class ArithmeticOp : public PhyComputeFnWithClosure<ArithmeticClosure> {
 public:
  std::string Name() const {
    switch(closure.type) {
      case ArithmeticType::kAdd:   return "+";
      case ArithmeticType::kSub:   return "-";
      case ArithmeticType::kMult:  return ".*";
      case ArithmeticType::kDiv:   return "./";
    };
    return "NA";
  }
};

class ArithmeticConstOp : public PhyComputeFnWithClosure<ArithmeticConstClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    if(closure.side == 0) { // left
      ss << closure.val;
    }
    switch(closure.type) {
      case ArithmeticType::kAdd:   ss << "+."; break;
      case ArithmeticType::kSub:   ss << "-."; break;
      case ArithmeticType::kMult:  ss << ".*"; break;
      case ArithmeticType::kDiv:   ss << "./"; break;
    };
    if(closure.side == 1) { // right
      ss << closure.val;
    }
    return ss.str();
  }
};

class NormArithmeticOp : public PhyComputeFnWithClosure<NormArithmeticClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.type) {
      case ArithmeticType::kAdd:
        ss << "+";
        break;
      case ArithmeticType::kSub:
        ss << "-";
        break;
      case ArithmeticType::kMult:
        ss << ".*";
        break;
      case ArithmeticType::kDiv:
        ss << "./";
        break;
    }
    ss << " norm";
    return ss.str();
  }
};

class SigmoidForwardOp : public PhyComputeFnWithClosure<SigmoidForwardClosure> {
 public:
  std::string Name() const {
    return "sigmoid forward";
  }
};

class SigmoidBackwardOp : public PhyComputeFnWithClosure<SigmoidBackwardClosure> {
 public:
  std::string Name() const {
    return "sigmoid backward";
  }
};

class ReluForwardOp : public PhyComputeFnWithClosure<ReluForwardClosure> {
 public:
  std::string Name() const {
    return "relu forward";
  }
};

class ReluBackwardOp : public PhyComputeFnWithClosure<ReluBackwardClosure> {
 public:
  std::string Name() const {
    return "relu backward";
  }
};

class TanhForwardOp : public PhyComputeFnWithClosure<TanhForwardClosure> {
 public:
  std::string Name() const {
    return "tanh forward";
  }
};

class TanhBackwardOp : public PhyComputeFnWithClosure<TanhBackwardClosure> {
 public:
  std::string Name() const {
    return "tanh backward";
  }
};

class ConvForwardOp : public PhyComputeFnWithClosure<ConvForwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv ff";
    return ss.str();
  }
};

class ConvBackwardDataOp : public PhyComputeFnWithClosure<ConvBackwardDataClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv bp data";
    return ss.str();
  }
};

class ConvBackwardFilterOp : public PhyComputeFnWithClosure<ConvBackwardFilterClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv bp filter";
    return ss.str();
  }
};

class ConvBackwardBiasOp : public PhyComputeFnWithClosure<ConvBackwardBiasClosure> {
 public:
  std::string Name() const {
    return "conv bp bias";
  }
};

class SoftmaxForwardOp : public PhyComputeFnWithClosure<SoftmaxForwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case SoftmaxAlgorithm::kInstance:
        return "instance softmax ff";
      case SoftmaxAlgorithm::kChannel:
        return "channel softmax ff";
    }
    return "unknown softmax ff";
  }
};

class SoftmaxBackwardOp : public PhyComputeFnWithClosure<SoftmaxBackwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case SoftmaxAlgorithm::kInstance:
        return "instance softmax bp";
      case SoftmaxAlgorithm::kChannel:
        return "channel softmax bp";
    }
    return "unknown softmax bp";
  }
};

class ActivationForwardOp : public PhyComputeFnWithClosure<ActivationForwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case ActivationAlgorithm::kSigmoid:
        return "sigmoid ff";
      case ActivationAlgorithm::kRelu:
        return "relu ff";
      case ActivationAlgorithm::kTanh:
        return "tanh ff";
    }
    return "unknown activation ff";
  }
};

class ActivationBackwardOp : public PhyComputeFnWithClosure<ActivationBackwardClosure> {
 public:
  std::string Name() const {
    switch (closure.algorithm) {
      case ActivationAlgorithm::kSigmoid:
        return "sigmoid bp";
      case ActivationAlgorithm::kRelu:
        return "relu bp";
      case ActivationAlgorithm::kTanh:
        return "tanh bp";
    }
    return "unknown activation bp";
  }
};

class PoolingForwardOp : public PhyComputeFnWithClosure<PoolingForwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.algorithm) {
      case PoolingInfo::Algorithm::kMax:
        ss << "max pooling ff";
        break;
      case PoolingInfo::Algorithm::kAverage:
        ss << "average pooling ff";
        break;
    }
    ss << " " << closure.height << "*" << closure.width;
    ss << " stride:" << closure.stride_horizontal << "*" << closure.stride_vertical;
    return ss.str();
  }
};

class PoolingBackwardOp : public PhyComputeFnWithClosure<PoolingBackwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.algorithm) {
      case PoolingInfo::Algorithm::kMax:
        ss << "max pooling bp";
        break;
      case PoolingInfo::Algorithm::kAverage:
        ss << "average pooling bp";
        break;
    }
    ss << " " << closure.height << "*" << closure.width;
    ss << " stride:" << closure.stride_horizontal << "*" << closure.stride_vertical;
    return ss.str();
  }
};

/*
class LRNOp : public PhyDataGenFnWithClosure<LRNClosure> {
 public:
  std::string Name() const {
    return "LRN";
  }
};
*/

}  // namespace minerva

