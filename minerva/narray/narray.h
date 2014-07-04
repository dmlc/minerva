#pragma once

#include <cstdlib>
#include <vector>
#include <initializer_list>

namespace minerva {

class Vector;
class NArray;
class Elewise;
class Reduction;

class Vector {
 public:
   Vector(int );
   template<typename T> Vector(std::initializer_list<T> init) {}
};

class Elewise {
 public:
  static NArray Add(const NArray&, const NArray&);
  static NArray Minus(const NArray&, const NArray&);
  static NArray Mult(const NArray&, const NArray&);
  static NArray Div(const NArray&, const NArray&);

  static NArray Exp(const NArray& );
  static NArray Ln(const NArray& );
  static NArray Sigmoid(const NArray&, const NArray&);
};

class NArray {
  friend class Elewise;
  friend class Reduction;
 public:
  static NArray Constant(const Vector& size, float val);
  static NArray Randn(const Vector& size, float mu, float var);
 public:
  // element-wise
  friend NArray operator + (const NArray&, const NArray&);
  friend NArray operator - (const NArray&, const NArray&);
  friend NArray operator / (const NArray&, const NArray&);
  friend NArray operator + (float, const NArray&);
  friend NArray operator - (float, const NArray&);
  friend NArray operator * (float, const NArray&);
  friend NArray operator / (float, const NArray&);
  friend NArray operator + (const NArray&, float);
  friend NArray operator - (const NArray&, float);
  friend NArray operator * (const NArray&, float);
  friend NArray operator / (const NArray&, float);
  // matmult
  friend NArray operator * (const NArray&, const NArray&);
  // lazy reductions
  NArray Sum(const Vector& dims);
  NArray Max(const Vector& dims);
  NArray MaxIndex(const Vector& dims);
  // non-lazy reductions
  float Sum();
  float Max();
  int CountZero();
  // shape
  size_t Size(size_t dim);
  NArray Tile(const Vector& times);
  NArray Reshape(const Vector& dims);
  NArray Trans();
};

} // end of namespace minerva
