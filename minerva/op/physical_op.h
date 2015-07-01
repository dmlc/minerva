#pragma once
#include "op/physical.h"
#include "op/physical_fn.h"
#include "op/closure.h"
#include "op/impl/bundle.h"
#include <sstream>
#include <vector>
#include <dmlc/logging.h>

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
class SyncWithPSOp : public ComputeFnWithClosure<SyncWithPSClosure> {
public:
  std::string Name() const {
    return std::string(":sync with ps on layer") + closure.layer_name;
  }
};

class MatMultOp : public ComputeFnWithClosure<MatMultClosure> {
 public:
  std::string Name() const {
    return "*";
  }
};

class TransOp : public ComputeFnWithClosure<TransposeClosure> {
 public:
  std::string Name() const {
    return "trans";
  }
};

class ReductionExceptDimOp : public ComputeFnWithClosure<ReductionExceptDimClosure> {
 public:
  std::string Name() const {
   switch (closure.type) {
     case ReductionType::kSum:
       return "sum";
     case ReductionType::kMax:
       return "max";
   }
   return "reduction except Dim N/A";
  }
};

class ReductionOp : public ComputeFnWithClosure<ReductionClosure> {
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

class ReductionWithReshapeOp : public ComputeFnWithClosure<ReductionWithReshapeClosure> {
 public:
  std::string Name() const {
   switch (closure.type) {
     case ReductionType::kSum:
       return "sum with reshape";
     case ReductionType::kMax:
       return "max with reshape";
   }
   return "reduction with reshape N/A";
  }
};

class MaxIndexOp : public ComputeFnWithClosure<MaxIndexClosure> {
 public:
  std::string Name() const {
    return "max index";
  }
};

class ReshapeOp : public ComputeFnWithClosure<ReshapeClosure> {
 public:
  std::string Name() const {
    return "reshape";
  }
};

class PowOp : public ComputeFnWithClosure<PowClosure> {
 public:
  std::string Name() const {
    return "pow";
  }
};

class ElewiseOp : public ComputeFnWithClosure<ElewiseClosure> {
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

class ArithmeticOp : public ComputeFnWithClosure<ArithmeticClosure> {
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

class ArithmeticConstOp : public ComputeFnWithClosure<ArithmeticConstClosure> {
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

class NormExceptDimArithmeticOp : public ComputeFnWithClosure<NormExceptDimArithmeticClosure> {
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
    ss << " norm except dim";
    return ss.str();
  }
};

class NormArithmeticOp : public ComputeFnWithClosure<NormArithmeticClosure> {
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

class SigmoidForwardOp : public ComputeFnWithClosure<SigmoidForwardClosure> {
 public:
  std::string Name() const {
    return "sigmoid forward";
  }
};

class SigmoidBackwardOp : public ComputeFnWithClosure<SigmoidBackwardClosure> {
 public:
  std::string Name() const {
    return "sigmoid backward";
  }
};

class ReluForwardOp : public ComputeFnWithClosure<ReluForwardClosure> {
 public:
  std::string Name() const {
    return "relu forward";
  }
};

class ReluBackwardOp : public ComputeFnWithClosure<ReluBackwardClosure> {
 public:
  std::string Name() const {
    return "relu backward";
  }
};

class TanhForwardOp : public ComputeFnWithClosure<TanhForwardClosure> {
 public:
  std::string Name() const {
    return "tanh forward";
  }
};

class TanhBackwardOp : public ComputeFnWithClosure<TanhBackwardClosure> {
 public:
  std::string Name() const {
    return "tanh backward";
  }
};

class ConvForwardOp : public ComputeFnWithClosure<ConvForwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv ff";
    return ss.str();
  }
};

class ConvBackwardDataOp : public ComputeFnWithClosure<ConvBackwardDataClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv bp data";
    return ss.str();
  }
};

class ConvBackwardFilterOp : public ComputeFnWithClosure<ConvBackwardFilterClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_height << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_vertical << "*" << closure.stride_horizontal;
    ss << " conv bp filter";
    return ss.str();
  }
};

class ConvBackwardBiasOp : public ComputeFnWithClosure<ConvBackwardBiasClosure> {
 public:
  std::string Name() const {
    return "conv bp bias";
  }
};

class SoftmaxForwardOp : public ComputeFnWithClosure<SoftmaxForwardClosure> {
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

class SoftmaxBackwardOp : public ComputeFnWithClosure<SoftmaxBackwardClosure> {
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

class ActivationForwardOp : public ComputeFnWithClosure<ActivationForwardClosure> {
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

class ActivationBackwardOp : public ComputeFnWithClosure<ActivationBackwardClosure> {
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

class PoolingForwardOp : public ComputeFnWithClosure<PoolingForwardClosure> {
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

class PoolingBackwardOp : public ComputeFnWithClosure<PoolingBackwardClosure> {
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

class LRNForwardOp : public ComputeFnWithClosure<LRNForwardClosure> {
 public:
  std::string Name() const {
    return "LRN Forward";
  }
};

class LRNBackwardOp : public ComputeFnWithClosure<LRNBackwardClosure> {
 public:
  std::string Name() const {
    return "LRN Backward";
  }
};

class ConcatOp : public ComputeFnWithClosure<ConcatClosure> {
 public:
  std::string Name() const {
    return "Concat";
  }
};

class SliceOp : public ComputeFnWithClosure<SliceClosure> {
 public:
  std::string Name() const {
    return "Slice";
  }
};

class IndexOp : public ComputeFnWithClosure<IndexClosure> {
 public:
  std::string Name() const {
    return "Index";
  }
};

class SelectOp : public ComputeFnWithClosure<SelectClosure> {
 public:
  std::string Name() const {
    return "Select";
  }
};

}  // namespace minerva

