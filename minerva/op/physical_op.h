#pragma once
#include "op/physical.h"
#include "op/closure.h"
#include "op/impl/bundle.h"
#include "device/device_info.h"
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
     case SUM:
       return "sum";
     case MAX:
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
      case EXP:      return "exp";
      case LN:       return "ln";
      case SIGMOID:  return "sigmoid";
      case NEGATIVE: return "-";
    };
    return "NA";
  }
};

class ArithmeticOp : public PhyComputeFnWithClosure<ArithmeticClosure> {
 public:
  std::string Name() const {
    switch(closure.type) {
      case ADD:   return "+";
      case SUB:   return "-";
      case MULT:  return ".*";
      case DIV:   return "./";
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
      case ADD:   ss << "+."; break;
      case SUB:   ss << "-."; break;
      case MULT:  ss << ".*"; break;
      case DIV:   ss << "./"; break;
    };
    if(closure.side == 1) { // right
      ss << closure.val;
    }
    return ss.str();
  }
};

class NormArithmeticOp: public PhyComputeFnWithClosure<NormArithmeticClosure> {
 public:
  std::string Name() const {
    std::stringstream ss;
    switch (closure.type) {
      case ADD:
        ss << "+";
        break;
      case SUB:
        ss << "-";
        break;
      case MULT:
        ss << ".*";
        break;
      case DIV:
        ss << "./";
        break;
    }
    ss << " norm";
    return ss.str();
  }
};

}

