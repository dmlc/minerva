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

NArray operator+(NArray, NArray);
NArray operator-(NArray, NArray);
NArray operator/(NArray, NArray);
NArray operator+(float, NArray);
NArray operator-(float, NArray);
NArray operator*(float, NArray);
NArray operator/(float, NArray);
NArray operator+(NArray, float);
NArray operator-(NArray, float);
NArray operator*(NArray, float);
NArray operator/(NArray, float);

}

