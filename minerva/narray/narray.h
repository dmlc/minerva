#pragma once

#include <cstdlib>
#include <vector>
#include <initializer_list>

#include "common/scale.h"
#include "dag/logical.h"

namespace minerva {

class NArray;
class Elewise;
class Reduction;
class Convolution;

class Elewise {
 public:
  static NArray Add(NArray, NArray);
  static NArray Minus(NArray, NArray);
  static NArray Mult(NArray&, NArray);
  static NArray Div(NArray, NArray);

  static NArray Exp(NArray );
  static NArray Ln(NArray );
  static NArray Sigmoid(NArray );
};

struct ConvInfo {
  int numfilters;
  Scale filtersize, stride, paddingsize;
};

class Convolution {
 public:
  static NArray ConvFF(NArray act, NArray weight, const ConvInfo& convinfo);
  static NArray ConvBP(NArray sen, NArray weight, const ConvInfo& convinfo);
  static NArray GetGrad(NArray act_low, NArray sen_high, const ConvInfo& convinfo);
};

class NArray {
  friend class Elewise;
  friend class Reduction;
  friend class Convolution;
 public:
  static NArray Constant(const Scale& size, float val);
  static NArray Randn(const Scale& size, float mu, float var);
  NArray();
 public:
  // element-wise
  friend NArray operator + (NArray, NArray);
  friend NArray operator - (NArray, NArray);
  friend NArray operator / (NArray, NArray);
  friend NArray operator + (float, NArray);
  friend NArray operator - (float, NArray);
  friend NArray operator * (float, NArray);
  friend NArray operator / (float, NArray);
  friend NArray operator + (NArray, float);
  friend NArray operator - (NArray, float);
  friend NArray operator * (NArray, float);
  friend NArray operator / (NArray, float);
  void operator += (NArray );
  void operator -= (NArray );
  void operator *= (NArray );
  void operator /= (NArray );
  void operator += (float );
  void operator -= (float );
  void operator *= (float );
  void operator /= (float );
  // matmult
  friend NArray operator * (NArray, NArray);
  // lazy reductions
  NArray Sum(const Scale& dims);
  NArray Max(const Scale& dims);
  NArray MaxIndex(const Scale& dims);
  // non-lazy reductions
  float Sum();
  float Max();
  int CountZero();
  // shape
  Scale Size();
  int Size(int dim);
  NArray Tile(const Scale& times);
  NArray Reshape(const Scale& dims);
  NArray Trans();

 private:
  NArray(LogicalDataNode* node);
  LogicalDataNode* data_node_;
};

} // end of namespace minerva
