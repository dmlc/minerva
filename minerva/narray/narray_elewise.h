#pragma once
#include "narray/narray.h"

namespace minerva {

class Elewise {
 public:
  static NArray Mult(const NArray&, const NArray&);
  static NArray Exp(const NArray&);
  static NArray Ln(const NArray&);
  static NArray Sigmoid(const NArray&);
};

NArray operator+(const NArray&, const NArray&);
NArray operator-(const NArray&, const NArray&);
NArray operator/(const NArray&, const NArray&);
NArray operator+(float, const NArray&);
NArray operator-(float, const NArray&);
NArray operator*(float, const NArray&);
NArray operator/(float, const NArray&);
NArray operator+(const NArray&, float);
NArray operator-(const NArray&, float);
NArray operator*(const NArray&, float);
NArray operator/(const NArray&, float);

}

