#pragma once
#include "narray/narray.h"

namespace minerva {

// TODO yutian: better use namespace
class Elewise {
 public:
  static NArray Mult(const NArray&, const NArray&);
  static NArray Exp(const NArray&);
  static NArray Ln(const NArray&);
  static NArray Pow(const NArray&, float exponent);
  static NArray SigmoidForward(const NArray&);
  static NArray SigmoidBackward(const NArray& diff, const NArray& top, const NArray& bottom);
  static NArray ReluForward(const NArray&);
  static NArray ReluBackward(const NArray& diff, const NArray& top, const NArray& bottom);
  static NArray TanhForward(const NArray&);
  static NArray TanhBackward(const NArray& diff, const NArray& top, const NArray& bottom);
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

