#pragma once
#include "narray/narray.h"

namespace minerva {

class Elewise {
 public:
  static NArray Mult(NArray, NArray);
  static NArray Exp(NArray);
  static NArray Ln(NArray);
  static NArray Sigmoid(NArray);
};

}

