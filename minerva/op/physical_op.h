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
class RandnOp : public PhyDataGenFnWithClosure<RandnClosure> {
 public:
  std::string Name() const {
    return ":randn";
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

class ElewiseOp : public PhyComputeFnWithClosure<ElewiseClosure> {
 public:
  std::string Name() const {
    switch(closure.type) {
      case ElewiseType::kExp:      return "exp";
      case ElewiseType::kLn:       return "ln";
      case ElewiseType::kSigmoid:  return "sigmoid";
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

class ConvForwardOp : public PhyComputeFnWithClosure<ConvForwardClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    ss << "pad:" << closure.pad_width << "*" << closure.pad_width;
    ss << " stride:" << closure.stride_horizontal << "*" << closure.stride_vertical;
    ss << " conv ff";
    return ss.str();
  }
};

}

